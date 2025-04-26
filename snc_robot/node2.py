#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, CameraInfo
from visualization_msgs.msg import Marker # publisher definition
from std_msgs.msg import String, Header, Float32MultiArray
from typing import Union
from geometry_msgs.msg import Point, PoseStamped, PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs 
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException, TransformException
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from rclpy.duration import Duration
import math 
from find_object_2d.msg import ObjectsStamped
import os
from tf2_geometry_msgs import do_transform_pose

FIND_OBJECT_ID_TO_HAZARD_ID = {
    1: 1, 
    2: 2, 
    3: 3, 
    4: 4, 
    5: 5, 
    6: 6, 
    7: 7, 
    8: 8, 
    9: 9, 
    10: 10, 
    11: 11, 
    12: 12,
}

HAZARD_ID_TO_NAME = {
    1: "Explosive", 2: "Flammable Gas", 3: "Non-Flammable Gas", 4: "Dangerous When Wet",
    5: "Flammable Solid", 6: "Spontaneously Combustible", 7: "Oxidizer", 8: "Organic Peroxide",
    9: "Inhalation Hazard", 10: "Poison", 11: "Radioactive", 12: "Corrosive"
}

class HazardMarkerDetector(Node):
    """
    listens for hazard markers detected by find_object_2d and sensor data.
    Currently focused on diagnosing why find_object_2d messages are not received.
    Placeholder functions exist for future 3D estimation and TF transformation.
    """
    def __init__(self):
        # initializes node, parameters, subscribers, publishers 
        super().__init__('hazard_marker_detector')

        # parameters
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('depth_image_topic', '/camera/depth/image_raw')
        self.declare_parameter('laser_scan_topic', '/scan')
        self.declare_parameter('find_object_topic', '/find_object/objects')
        self.declare_parameter('hazard_marker_topic', '/hazards')
        self.declare_parameter('coords_topic', '/hazard_coords')
        self.declare_parameter('status_topic', '/snc_status')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('camera_optical_frame', 'camera_color_optical_frame')
        self.declare_parameter('marker_publish_dist_threshold', 0.2)

        # parameter values
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.depth_image_topic = self.get_parameter('depth_image_topic').value
        self.laser_scan_topic = self.get_parameter('laser_scan_topic').value
        self.find_object_topic = '/objectsStamped'
        self.hazard_marker_topic = self.get_parameter('hazard_marker_topic').value
        self.status_topic = self.get_parameter('status_topic').value
        self.coords_topic = self.get_parameter('coords_topic').value
        self.map_frame = self.get_parameter('map_frame').value
        self.camera_optical_frame = self.get_parameter('camera_optical_frame').value

        # initialization
        self.bridge = CvBridge()
        self.tf_buffer = Buffer() 
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.camera_intrinsics = None
        self.last_depth_image = None
        self.last_depth_header = None
        
        # track detected hazard markers to avoid duplicates
        self.detected_hazards = {}  # dict to store hazard_id -> position
        
        # QoS Profiles
        qos_reliable = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        qos_sensor = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=5)
        qos_cam_info = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1, durability=QoSDurabilityPolicy.VOLATILE)

        # subscriptions 
        self.get_logger().info(f"Subscribing to Camera Info: {self.camera_info_topic}")
        self.cam_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, qos_cam_info)

        self.get_logger().info(f"Subscribing to Depth Image: {self.depth_image_topic}")
        self.depth_subscription = self.create_subscription(Image, self.depth_image_topic, self.depth_callback, qos_sensor)

        # crucial subscription
        self.get_logger().info(f"Attempting to subscribe to Find Object: {self.find_object_topic}")
        self.find_object_sub = self.create_subscription(
            ObjectsStamped,
            self.find_object_topic,
            self.find_object_callback,
            10
        )

        # publishers 
        self.get_logger().info(f"Publisher initialized for Hazard Markers: {self.hazard_marker_topic}")
        self.marker_publisher = self.create_publisher(Marker, self.hazard_marker_topic, qos_reliable)

        self.get_logger().info(f"Publisher initialized for Status: {self.status_topic}")
        self.status_publisher = self.create_publisher(String, self.status_topic, qos_reliable)
        
        self.get_logger().info(f"Publisher initialized for Hazard Coordinates: {self.coords_topic}")
        self.coords_publisher = self.create_publisher(String, self.coords_topic, qos_reliable)

        self.get_logger().info('Hazard Marker Detector Node Initialized.')
        self.publish_status("Initializing and waiting for find_object messages...")

    def publish_status(self, status_text):
        # helper function to publish a status message and log it
        msg = String()
        msg.data = f"Node2: {status_text}"
        self.status_publisher.publish(msg)
        self.get_logger().info(f"Status: {status_text}")

    def camera_info_callback(self, msg: CameraInfo):
        # stores camera intrinsic parameters when received
        if self.camera_intrinsics is None:
            self.get_logger().info("Received camera intrinsics.")
            self.camera_intrinsics = { 'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5], 'width': msg.width, 'height': msg.height }
            self.publish_status("Camera info received")

    def depth_callback(self, msg: Image):
        # stores the latest depth image
        self.get_logger().debug('Received depth image message.')
        try:
            self.last_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.last_depth_header = msg.header
        except Exception as e:
            self.get_logger().error(f'CvBridge Error converting depth image: {e}')

    # def find_object_callback(self, msg: ObjectsStamped):
    #     """
    #     Callback for find_object_2d detections.
    #     Extracts object coordinates, estimates 3D position, transforms to map frame, and publishes the hazard marker.
    #     """
    #     # log receipt of the object detection message
    #     self.get_logger().info(f"Received {len(msg.objects.data) // 12} objects detected!")

    #     if len(msg.objects.data) == 0:
    #         self.get_logger().info("Empty detection message received.")
    #         return

    #     # loop through all detected objects (every 12th entry in the msg.objects.data)
    #     for i in range(0, len(msg.objects.data), 12):  # Each object has 12 values

    #         object_id = int(msg.objects.data[i])
    #         self.get_logger().info(f"Processing object with ID: {object_id}")
        
    #         # extract the homography matrix for this object detection (values 2 to 11)
    #         h = np.array(msg.objects.data[i + 2: i + 11]).reshape(3, 3)

    #         # compute pixel position (u, v) from homography (center of object)
    #         self.get_logger().info(f"Homography matrix: {h}")
    #         px_homo = np.dot(h, np.array([0, 0, 1]))  # Object center in image space
    #         self.get_logger().info(f"Homogeneous coords: {px_homo}")
    #         pixel_x = px_homo[0] / px_homo[2]
    #         pixel_y = px_homo[1] / px_homo[2]

    #         self.get_logger().info(f"Detected object at pixel: ({pixel_x}, {pixel_y})")

    #         # estimate the 3D position using depth information
    #         point_camera = self.estimate_3d_position(int(pixel_x), int(pixel_y), msg.header)
    #         if point_camera:
    #             self.get_logger().info(f"Estimated 3D position: {point_camera}")

    #             # transform to map frame using TF
    #             map_point = self.transform_to_map(point_camera, msg.header)
    #             if map_point:
    #                 self.get_logger().info(f"Map-relative position: x={map_point.x}, y={map_point.y}, z={map_point.z}")

    #                 # get hazard ID from the message data (directly mapped from find_object)
    #                 hazard_id = msg.objects.data[i]  # Use the first value of the 12 values for object ID
    #                 hazard_name = HAZARD_ID_TO_NAME.get(hazard_id, "Unknown")

    #                 # save the hazard marker position to file
    #                 self.save_hazard_marker_position(hazard_id, hazard_name, map_point)

    #                 # publish the marker at the map position
    #                 self.publish_marker(map_point, hazard_id, hazard_name)
    #             else:
    #                 self.get_logger().warning(f"Failed to transorm point to map frame")

    #         # if no valid point found, continue checking for other objects
    #         else:
    #             self.get_logger().warning(f"Could not estimate 3D position for object {i // 12 + 1}.")

    #             # if we can't get depth, try to estimate position in a different way
    #             self.publish_status(f"Detected {hazard_name} but could not determine position")

    #     # debugging to check how many objects were detected in the message
    #     if msg.objects.data:
    #         num_objects = len(msg.objects.data) // 12 
    #         self.get_logger().info(f"  Message contained {num_objects} detected object(s).")


    def find_object_callback(self, msg):
        """
        Callback for find_object_2d detections.
        Extracts object coordinates, estimates 3D position, transforms to map frame, and publishes the hazard marker.
        """
        if not msg.objects.data:
            self.get_logger().info("Empty detection message received.")
            self.publish_status("No hazards detected")
            return

        self.get_logger().info(f"Received {len(msg.objects.data) // 12} objects detected!")

        for i in range(0, len(msg.objects.data), 12):
            object_id = int(msg.objects.data[i])
            hazard_name = HAZARD_ID_TO_NAME.get(object_id, "Unknown")
            
            self.get_logger().info(f"Processing object with ID: {object_id}")
            
            # Extract the homography matrix for this object
            homography_matrix = np.array(msg.objects.data[i + 2: i + 11]).reshape(3, 3)
            self.get_logger().info(f"Homography matrix: {homography_matrix}")
            
            # IMPORTANT: Proper extraction of pixel coordinates from homography matrix
            # The homography matrix maps from object coordinates to image coordinates
            
            # For find_object_2d, the object center is at (0,0,1) in homogeneous coordinates
            # We need to extract translation elements (3rd column) and scale by the homogeneous factor
            
            if self.camera_intrinsics is None:
                self.get_logger().error("Camera intrinsics not available!")
                continue
                
            # Get image dimensions
            width = int(self.camera_intrinsics['width'])
            height = int(self.camera_intrinsics['height'])
            
            # Construct a proper coordinate for the object in 3D space (in the camera frame)
            # Based on testing, find_object_2d's comments suggest that the object is at (355, 235) 
            # according to the warning message in your logs
            pixel_x = int(homography_matrix[0, 2] * width / homography_matrix[2, 2])
            pixel_y = int(homography_matrix[1, 2] * height / homography_matrix[2, 2])
            
            # Ensure values are within the image bounds
            pixel_x = max(0, min(width-1, pixel_x))
            pixel_y = max(0, min(height-1, pixel_y))
            
            # If the calculated pixel coordinates are still near (0,0), use a fixed position
            if pixel_x < 10 and pixel_y < 10:
                # The homography matrix might be in normalized coordinates (0-1)
                # Try a different interpretation
                self.get_logger().warn(f"Calculated coordinates too close to origin: ({pixel_x}, {pixel_y})")
                
                # Try extracting coordinates differently
                # Some implementations use the homography matrix differently
                try:
                    # Check the first values in the homography - they might be the image width
                    if homography_matrix[0, 0] > 100:  # If it's a large value like 800
                        self.get_logger().info(f"Homography appears to contain image width: {homography_matrix[0, 0]}")
                        # Try different combinations from the homography
                        candidates = [
                            (homography_matrix[0, 2], homography_matrix[1, 2]),  # Direct translation
                            (homography_matrix[2, 0], homography_matrix[2, 1]),  # Bottom row
                            (homography_matrix[0, 2] / homography_matrix[2, 2], 
                            homography_matrix[1, 2] / homography_matrix[2, 2])  # Normalized
                        ]
                        
                        # Log all candidates
                        for idx, (x, y) in enumerate(candidates):
                            self.get_logger().info(f"Coordinate candidate {idx+1}: ({x}, {y})")
                        
                        # Check if the 3rd row, first two values might contain useful coordinates
                        row3_scale = abs(homography_matrix[2, 2])
                        if row3_scale > 0:
                            pixel_x = int(abs(homography_matrix[2, 0]) * width / row3_scale)
                            pixel_y = int(abs(homography_matrix[2, 1]) * height / row3_scale)
                            self.get_logger().info(f"Trying third row coordinates: ({pixel_x}, {pixel_y})")
                    
                    # Validate again
                    pixel_x = max(10, min(width-10, pixel_x))
                    pixel_y = max(10, min(height-10, pixel_y))
                except Exception as e:
                    self.get_logger().error(f"Error in alternate coordinate calculation: {e}")
                    # Fallback to image center if all else fails
                    pixel_x = width // 2
                    pixel_y = height // 2
            
            self.get_logger().info(f"Using pixel coordinates: ({pixel_x}, {pixel_y})")
            
            # Now attempt 3D position estimation with the better coordinates
            position_camera = self.estimate_3d_position(pixel_x, pixel_y, msg.header)
            
            if position_camera:
                try:
                    # Transform to map frame
                    map_point = self.transform_to_map(position_camera, msg.header)
                    
                    if map_point:
                        self.get_logger().info(f"Transformed position in map frame: {map_point}")
                        
                        # Save hazard marker position and publish marker
                        self.save_hazard_marker_position(object_id, hazard_name, map_point)
                        self.publish_marker(map_point, object_id, hazard_name)
                        
                        # Publish status
                        self.publish_status(f"Detected {hazard_name} at position: x={map_point.x:.2f}, y={map_point.y:.2f}")
                    else:
                        self.get_logger().error("Failed to transform point to map frame")
                        self.publish_status(f"Detected {hazard_name} but failed to transform to map")
                except Exception as e:
                    self.get_logger().error(f"Error in transformation: {e}")
                    self.publish_status(f"Detected {hazard_name} but error in transformation")
            else:
                self.get_logger().warning(f"Could not estimate 3D position for {hazard_name}")
                self.publish_status(f"Detected {hazard_name} but could not determine position")

    # def find_object_callback(self, msg):
    #     """
    #     Callback for find_object_2d detections.
    #     Extracts object coordinates, estimates 3D position, transforms to map frame, and publishes the hazard marker.
    #     """
    #     if not msg.objects.data:
    #         self.get_logger().info("Empty detection message received.")
    #         self.publish_status("No hazards detected")
    #         return

    #     self.get_logger().info(f"Received {len(msg.objects.data) // 12} objects detected!")

    #     for i in range(0, len(msg.objects.data), 12):
    #         object_id = int(msg.objects.data[i])
    #         self.get_logger().info(f"Processing object with ID: {object_id}")

    #         hazard_name = HAZARD_ID_TO_NAME.get(object_id, "Unknown")
            
    #         # === Homography matrix calculation ===
    #         homography_matrix = np.array(msg.objects.data[i + 2: i + 11]).reshape(3, 3)
    #         self.get_logger().info(f"Homography matrix: {homography_matrix}")
            
    #         # Calculate the center point of the object in normalized coordinates
    #         homogeneous_coords = np.dot(homography_matrix, np.array([0, 0, 1]))
    #         self.get_logger().info(f"Homogeneous coords: {homogeneous_coords}")
            
    #         # Get the image dimensions
    #         if self.camera_intrinsics is None:
    #             self.get_logger().error("Camera intrinsics not available!")
    #             continue
                
    #         width = self.camera_intrinsics['width']
    #         height = self.camera_intrinsics['height']
            
    #         # Get pixel coordinates
    #         if homogeneous_coords[2] != 0:
    #             pixel_x = int(width // 2)  # Use center of image as fallback
    #             pixel_y = int(height // 2)
                
    #             self.get_logger().info(f"Using center position: ({pixel_x}, {pixel_y})")
                
    #             # Attempt to estimate 3D position
    #             position_camera = self.estimate_3d_position(pixel_x, pixel_y, msg.header)
                
    #             if position_camera:
    #                 try:
    #                     # Transform from camera frame to map frame
    #                     map_point = self.transform_to_map(position_camera, msg.header)
                        
    #                     if map_point:
    #                         self.get_logger().info(f"Transformed 3D position in map frame: {map_point}")
                            
    #                         # Save hazard marker position and publish marker
    #                         self.save_hazard_marker_position(object_id, hazard_name, map_point)
    #                         self.publish_marker(map_point, object_id, hazard_name)
    #                     else:
    #                         self.get_logger().error("Failed to transform point to map frame")
                    
    #                 except Exception as e:
    #                     self.get_logger().error(f"Error transforming point to map frame: {e}")
    #                     self.publish_status(f"Detected {hazard_name} but error transforming to map frame")
    #             else:
    #                 self.get_logger().warning(f"Could not estimate 3D position for object {object_id}.")
    #                 self.publish_status(f"Detected {hazard_name} but could not determine position")
    #         else:
    #             self.get_logger().error("Invalid homogeneous coordinates (division by zero)")
    #             continue

    # def find_object_callback(self, msg):
    #     """
    #     Callback for find_object_2d detections.
    #     Extracts object coordinates, estimates 3D position, transforms to map frame, and publishes the hazard marker.
    #     """
    #     if not msg.objects.data:
    #         self.get_logger().info("Empty detection message received.")
    #         self.publish_status("No hazards detected")
    #         return

    #     self.get_logger().info(f"Received {len(msg.objects.data) // 12} objects detected!")
        
    #     for i in range(0, len(msg.objects.data), 12):
    #         object_id = int(msg.objects.data[i])
    #         self.get_logger().info(f"Processing object with ID: {object_id}")

    #         hazard_name = HAZARD_ID_TO_NAME.get(object_id, "Unknown")
            
    #         # === Homography matrix calculation ===
    #         homography_matrix = np.array(msg.objects.data[i + 2: i + 11]).reshape(3, 3)
    #         homogeneous_coords = np.dot(homography_matrix, np.array([0, 0, 1]))

    #         # Normalize
    #         pixel_x_norm = homogeneous_coords[0] / homogeneous_coords[2]
    #         pixel_y_norm = homogeneous_coords[1] / homogeneous_coords[2]

    #         # Scale to real pixel coordinates
    #         if self.last_depth_image is not None:
    #             h, w = self.last_depth_image.shape[:2]
    #             pixel_x = int(pixel_x_norm * w)
    #             pixel_y = int(pixel_y_norm * h)
    #         else:
    #             self.get_logger().error("No depth image available yet!")
    #             continue

    #         self.get_logger().info(f"Detected object at pixel: ({pixel_x}, {pixel_y})")

    #         # === 3D position estimation ===
    #         position_camera = self.estimate_3d_position(pixel_x, pixel_y, msg.header)
            
    #         if position_camera:
    #             try:
    #                 # Transform from camera frame to map frame
    #                 position_map = self.tf_buffer.transform(
    #                     PointStamped(header=Header(frame_id="camera_link"), point=position_camera),
    #                     "map",
    #                     timeout=Duration(seconds=0.5)
    #                 )

    #                 self.get_logger().info(f"Transformed 3D position in map frame: {position_map}")

    #                 # ✅ Save hazard marker using the correct function
    #                 self.save_hazard_marker_position(object_id, hazard_name, position_map.point)

    #             except Exception as e:
    #                 self.get_logger().error(f"Error transforming point to map frame: {e}")
    #                 self.publish_status(f"Detected {hazard_name} but error transforming to map frame")

    #         else:
    #             self.get_logger().warning(f"Could not estimate 3D position for object ID {object_id}")


    #     """
    #     for i in range(0, len(msg.objects.data), 12):
    #         object_id = int(msg.objects.data[i])
    #         self.get_logger().info(f"Processing object with ID: {object_id}")

    #         h = np.array(msg.objects.data[i + 2: i + 11]).reshape(3, 3)
    #         self.get_logger().info(f"Homography matrix: {h}")

    #         px_homo = np.dot(h, np.array([0, 0, 1]))
    #         self.get_logger().info(f"Homogeneous coords: {px_homo}")
            
    #         pixel_x = px_homo[0] / px_homo[2]
    #         pixel_y = px_homo[1] / px_homo[2]
    #         self.get_logger().info(f"Detected object at pixel: ({pixel_x}, {pixel_y})")

    #         # 🔵 Always extract hazard name early
    #         hazard_id = msg.objects.data[i]
    #         hazard_name = HAZARD_ID_TO_NAME.get(hazard_id, "Unknown")

    #         # Estimate 3D point
    #         point_camera = self.estimate_3d_position(int(pixel_x), int(pixel_y), msg.header)
            
    #         if point_camera:
    #             self.get_logger().info(f"Estimated 3D position in camera frame: {point_camera}")
    #             try:
    #                 point_map = self.tf_buffer.transform(
    #                     PointStamped(header=Header(frame_id="camera_link"), point=point_camera),
    #                     "map",
    #                     timeout=Duration(seconds=0.5)
    #                 )
    #                 self.get_logger().info(f"Transformed 3D position in map frame: {point_map}")

    #                 # Save the hazard marker
    #                 # self.save_hazard_marker(hazard_name, point_map.point)
    #                 self.save_hazard_marker_position(hazard_id, hazard_name, point_map.point)

    #             except Exception as e:
    #                 self.get_logger().error(f"Error transforming point to map frame: {e}")
    #                 self.publish_status(f"Detected {hazard_name} but error transforming to map frame")

    #         else:
    #             self.get_logger().warning(f"Could not estimate 3D position for object {i // 12 + 1}.")
    #             self.publish_status(f"Detected {hazard_name} but could not determine position")
    #     """

    # def save_hazard_marker_position(self, hazard_id, hazard_name, position):
    #     """
    #     Publishes the hazard marker position to a topic that can be echoed.
    #     """
    #     # create a unique key for the hazard (could be based on position and ID)
    #     marker_key = f"{hazard_id}"
        
    #     # check if we've already detected this hazard (using approximate position)
    #     already_detected = False
    #     for key, pos in self.detected_hazards.items():
    #         if key == marker_key:
    #             # check if the positions are similar (within 0.2m)
    #             dist = math.sqrt((pos[0] - position.x)**2 + 
    #                              (pos[1] - position.y)**2 + 
    #                              (pos[2] - position.z)**2)
    #             if dist < 0.2:  # already detected within 20cm
    #                 already_detected = True
    #                 break
        
    #     # create the formatted data string for the coordinate
    #     coord_str = f"ID: {hazard_id}, {hazard_name}, x: {position.x:.4f}, y: {position.y:.4f}, z: {position.z:.4f}"
        
    #     # always publish to topic for real-time access
    #     coord_msg = String()
    #     coord_msg.data = coord_str
    #     self.coords_publisher.publish(coord_msg)
        
    #     # if new detection, save it 
    #     if not already_detected:
    #         # store in detected hazards dictionary
    #         self.detected_hazards[marker_key] = (position.x, position.y, position.z)
            
    #         # print to terminal with more visibility
    #         terminal_output = f"NEW HAZARD MARKER DETECTED - ID: {hazard_id} ({hazard_name}) at position: x={position.x:.4f}, y={position.y:.4f}, z={position.z:.4f}"
    #         print("\n" + "="*80)
    #         print(terminal_output)
    #         print("="*80 + "\n")
    #         self.get_logger().info(f"New hazard detected: {coord_str}")
    #     else:
    #         # log continued visibility of already detected hazard
    #         self.get_logger().debug(f"Already detected hazard seen again: {coord_str}")

    def save_hazard_marker_position(self, hazard_id, hazard_name, position):
        """
        Publishes the hazard marker position to a topic that can be echoed.
        """

        # Check if position is None
        if position is None:
            self.get_logger().warning(f"Cannot save hazard marker for ID {hazard_id}: 3D position estimation failed.")
            return  # Stop here if position is not valid

        # create a unique key for the hazard (could be based on position and ID)
        marker_key = f"{hazard_id}"
        
        # check if we've already detected this hazard (using approximate position)
        already_detected = False
        for key, pos in self.detected_hazards.items():
            if key == marker_key:
                # check if the positions are similar (within 0.2m)
                dist = math.sqrt((pos[0] - position.x)**2 + 
                                (pos[1] - position.y)**2 + 
                                (pos[2] - position.z)**2)
                if dist < 0.2:  # already detected within 20cm
                    already_detected = True
                    break

        # create the formatted data string for the coordinate
        coord_str = f"ID: {hazard_id}, {hazard_name}, x: {position.x:.4f}, y: {position.y:.4f}, z: {position.z:.4f}"
        
        # always publish to topic for real-time access
        coord_msg = String()
        coord_msg.data = coord_str
        self.coords_publisher.publish(coord_msg)
        
        # if new detection, save it 
        if not already_detected:
            self.detected_hazards[marker_key] = (position.x, position.y, position.z)

            terminal_output = f"NEW HAZARD MARKER DETECTED - ID: {hazard_id} ({hazard_name}) at position: x={position.x:.4f}, y={position.y:.4f}, z={position.z:.4f}"
            print("\n" + "="*80)
            print(terminal_output)
            print("="*80 + "\n")
            self.get_logger().info(f"New hazard detected: {coord_str}")
        else:
            self.get_logger().debug(f"Already detected hazard seen again: {coord_str}")

    def estimate_3d_position(self, pixel_x: int, pixel_y: int, header: Header) -> Union[Point, None]:
        """
        Estimates the 3D coordinates of a point in the camera's optical frame,
        given its pixel coordinates (x, y). Prefers using depth camera data.
        Returns a Point object or None if estimation fails.
        """
        pixel_x = int(round(pixel_x))  
        pixel_y = int(round(pixel_y))          
        # log entry into the function for debugging flow
        self.get_logger().info(f"Attempting to estimate 3D position for pixel ({pixel_x}, {pixel_y})")

        # check prerequisites: We need both depth data and camera calibration
        if self.camera_intrinsics is None:
            self.get_logger().warning("Cannot estimate 3D position: Camera intrinsics not available.")
            return None
        
        if self.last_depth_image is None:
            self.get_logger().warning("Cannot estimate 3D position: Depth image not available.")
            self.publish_status("Waiting for depth image") 
            return None
            
        if self.last_depth_image is not None: 
            self.get_logger().info("Attempting estimation using depth image...")
            h, w = self.last_depth_image.shape[:2] # get depth image dimensions

        # check if the detected pixel coordinates are within the image bounds
        if 0 <= pixel_y < h and 0 <= pixel_x < w:
            # access the depth value at the specific pixel
            region_size = 5
            y_min = max(0, pixel_y - region_size)
            y_max = min(h, pixel_y + region_size)
            x_min = max(0, pixel_x - region_size)
            x_max = min(w, pixel_x + region_size)            
            region = self.last_depth_image[y_min:y_max, x_min:x_max]

            depth_value = self.last_depth_image[pixel_y, pixel_x]
            
            # Print depth value type and value for debugging
            self.get_logger().info(f"Depth value type: {type(depth_value)}, Value: {depth_value}")
            
            valid_depths = region[region > 0]
            if len(valid_depths) > 0:
                depth_value = np.median(valid_depths)

                # convert depth value to meters based on its type
                if isinstance(depth_value, np.uint16): 
                    depth_meters = float(depth_value) / 1000.0
                    self.get_logger().info(f"  Depth value (uint16) at ({pixel_x}, {pixel_y}): {depth_value} -> {depth_meters:.3f}m")
                elif isinstance(depth_value, (np.float32, np.float64, float)): 
                    # Handle float64 depth values - likely already in meters
                    depth_meters = float(depth_value)
                    self.get_logger().info(f"  Depth value (float) at ({pixel_x}, {pixel_y}): {depth_meters:.3f}m")
                else:
                    # handle unexpected depth types
                    self.get_logger().error(f"  Unexpected depth value type: {type(depth_value)}, value: {depth_value}. Cannot interpret depth.")
                    self.publish_status("Error: Unexpected depth format")
                    return None

                # Validate depth range
                if not np.isfinite(depth_meters):
                    self.get_logger().warning(f"  Invalid depth value (NaN or Inf): {depth_meters}")
                    return None
                    
                if 0.1 < depth_meters < 10.0:
                    fx = self.camera_intrinsics['fx']
                    fy = self.camera_intrinsics['fy']
                    cx = self.camera_intrinsics['cx']
                    cy = self.camera_intrinsics['cy']

                    # calculate 3d coords  
                    x_cam = (pixel_x - cx) * depth_meters / fx
                    y_cam = (pixel_y - cy) * depth_meters / fy
                    z_cam = depth_meters

                    self.get_logger().info(f"  Calculated 3D position: x={x_cam:.3f}, y={y_cam:.3f}, z={z_cam:.3f}")
                    return Point(x=x_cam, y=y_cam, z=z_cam)
                else:
                    self.get_logger().warning(f"  Depth value {depth_meters:.3f}m is outside valid range (0.1m - 10.0m)")
            else:
                self.get_logger().warning(f"  No valid depth values found in region around ({pixel_x}, {pixel_y})")

        else:
            self.get_logger().warning(f"  Pixel ({pixel_x}, {pixel_y}) is outside image bounds ({w}x{h})")

        self.get_logger().warning("Could not get valid depth information for the detected object")
        self.publish_status("3D estimation failed (check logs)")
        return None

    # def estimate_3d_position(self, pixel_x: int, pixel_y: int, header: Header) -> Union[Point, None]:
    #     """
    #     Estimates the 3D coordinates of a point in the camera's optical frame,
    #     given its pixel coordinates (x, y). Prefers using depth camera data.
    #     Returns a Point object or None if estimation fails.
    #     """
    #     pixel_x = int(round(pixel_x))  
    #     pixel_y = int(round(pixel_y))          
    #     # log entry into the function for debugging flow
    #     self.get_logger().info(f"Attempting to estimate 3D position for pixel ({pixel_x}, {pixel_y})")

    #     # check prerequisites: We need both depth data and camera calibration
    #     if self.camera_intrinsics is None:
    #         self.get_logger().warning("Cannot estimate 3D position: Camera intrinsics not available.")
    #         # self.publish_status("Waiting for camera info") # Update status
    #         return None
        
    #     if self.last_depth_image is None:
    #         self.get_logger().warning("Cannot estimate 3D position: Depth image not available.")
    #         self.publish_status("Waiting for depth image") 
    #         return None
            
    #     if self.last_depth_image is not None: 
    #         self.get_logger().info("Attempting estimation using depth image...")
    #         h, w = self.last_depth_image.shape[:2] # det depth image dimensions

    #     # check if the detected pixel coordinates are within the image bounds
    #     if 0 <= pixel_y < h and 0 <= pixel_x < w:
    #         # access the depth value at the specific pixel
    #         region_size = 5
    #         y_min = max(0, pixel_y - region_size)
    #         y_max = min(h, pixel_y + region_size)
    #         x_min = max(0, pixel_x - region_size)
    #         x_max = min(w, pixel_x + region_size)            
    #         region = self.last_depth_image[y_min:y_max, x_min:x_max]

    #         depth_value = self.last_depth_image[pixel_y, pixel_x]
            
    #         valid_depths = region[region > 0]
    #         if len(valid_depths) > 0:
    #             depth_value = np.median(valid_depths)

    #             # convert depth value to meters.
    #             if isinstance(depth_value, np.uint16): 
    #                 depth_meters = float(depth_value) / 1000.0
    #                 self.get_logger().info(f"  Depth value (uint16) at ({pixel_x}, {pixel_y}): {depth_value} -> {depth_meters:.3f}m")
    #             elif isinstance(depth_value, np.float32): # Check if it's likely meters
    #                 depth_meters = float(depth_value)
    #                 self.get_logger().info(f"  Depth value (float32) at ({pixel_x}, {pixel_y}): {depth_meters:.3f}m")
    #             else:
    #                 # handle unexpected depth types
    #                 self.get_logger().error(f"  Unexpected depth value type: {type(depth_value)}. Cannot interpret depth.")
    #                 self.publish_status("Error: Unexpected depth format")
    #                 return None

    #             if 0.1 < depth_meters < 10.0:
    #                 fx = self.camera_intrinsics['fx']
    #                 fy = self.camera_intrinsics['fy']
    #                 cx = self.camera_intrinsics['cx']
    #                 cy = self.camera_intrinsics['cy']

    #                 # calculate 3d coods  
    #                 x_cam = (pixel_x - cx) * depth_meters / fx
    #                 y_cam = (pixel_y - cy) * depth_meters / fy
    #                 z_cam = depth_meters

    #                 return Point(x=x_cam, y=y_cam, z=z_cam)

    #     self.get_logger().warning("Could not get valid depth information for the detected object")
    #     self.publish_status("3D estimation failed (check logs)")
    #     return None

    def transform_to_map(self, point_in_camera: Point, header: Header) -> Union[Point,None]:
        """
        Transforms 3D coordinates from the camera frame to the map frame using tf2.
        Returns the transformed position in the map frame.
        """
        try:
            camera_pose = PoseStamped()
            camera_pose.header.stamp = self.get_clock().now().to_msg()
            camera_pose.header.frame_id = self.camera_optical_frame
            camera_pose.pose.position = point_in_camera
            camera_pose.pose.orientation.w = 1.0

            self.get_logger().info(f"Looking up transform from {self.camera_optical_frame} to {self.map_frame}")
            
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.camera_optical_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            map_pose = do_transform_pose(camera_pose, transform)
            result_point = map_pose.pose.position

            self.get_logger().info(f"Transformed point: x={result_point.x:.3f}, y={result_point.y:.3f}, z={result_point.z:.3f}")
            return result_point

        except Exception as e:
            self.get_logger().error(f"Error during transform_to_map: {e}")
            return None
        # try:
        #     # create a PoseStamped message for the camera frame
        #     camera_pose = PoseStamped()
        #     camera_pose.header.stamp = self.get_clock().now().to_msg()
        #     camera_pose.header.frame_id = self.camera_optical_frame
        #     camera_pose.pose.position = point_in_camera
        #     camera_pose.pose.orientation.w = 1.0  # No rotation in this case
            
        #     self.get_logger().info(f"Looking up transform from {camera_pose.header.frame_id} to {self.map_frame}")
            
        #     try:
        #         transform = self.tf_buffer.lookup_transform(
        #             self.map_frame,
        #             camera_pose.header.frame_id,
        #             rclpy.time.Time(),
        #             timeout=rclpy.duration.Duration(seconds=1.0)
        #         )
        #         self.get_logger().info(f"Transform found successfully")
        #     except Exception as e:
        #         self.get_logger().error(f"Transform lookup error: {e}")
        #         return None

        #     # create a result point manually using the transform
        #     try:
        #         result_point = Point()
                
        #         # apply the transform directly instead of using do_transform_pose
        #         # this skips the problematic tf2_geometry_msgs.do_transform_pose step
        #         # xxtract the translation from the transform
        #         result_point.x = transform.transform.translation.x + point_in_camera.x
        #         result_point.y = transform.transform.translation.y + point_in_camera.y
        #         result_point.z = transform.transform.translation.z + point_in_camera.z
                    
        #         self.get_logger().info(f"Map-relative position: x={result_point.x:.3f}, y={result_point.y:.3f}, z={result_point.z:.3f}")
        #         return result_point
                
        #     except Exception as e:
        #         self.get_logger().error(f"Manual transform error: {e}")
        #         self.get_logger().info(f"Transform: {transform}")
        #         self.get_logger().info(f"Point in camera: {point_in_camera}")
        #         return None
                
        # except Exception as e:
        #     self.get_logger().error(f"Unexpected error in transform_to_map: {e}")
        #     return None    
    
    def publish_marker(self, position_in_map, marker_id, marker_name):
        """Publishes a visualization_msgs/Marker."""

        marker_id_int = int(marker_id)
        self.get_logger().info(f"Publishing marker for {marker_name} (ID: {marker_id_int}) at map coordinates: x={position_in_map.x:.3f}, y={position_in_map.y:.3f}, z={position_in_map.z:.3f}")

        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "hazard_markers"
        marker.id = marker_id_int
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = position_in_map
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.color.a = 0.8

        # color assignment
        if "explosive" in marker_name.lower() or "flammable" in marker_name.lower() or "oxidizer" in marker_name.lower():
            marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0 # Red
        elif "gas" in marker_name.lower():
             marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 1.0 # Blue
        elif "corrosive" in marker_name.lower():
             marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 0.0 # Yellow
        elif "poison" in marker_name.lower() or "inhalation" in marker_name.lower():
             marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 0.0 # Black
        elif "radioactive" in marker_name.lower():
             marker.color.r = 1.0; marker.color.g = 0.5; marker.color.b = 0.0 # Orange
        else:
            marker.color.r = 0.5; marker.color.g = 0.0; marker.color.b = 0.5 # Purple

        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
        
        # log just before publishing
        self.get_logger().info(f"Sending marker to publisher...")
        self.marker_publisher.publish(marker)
        self.get_logger().info(f"Marker published successfully!")


# main execution block
def main(args=None):
    rclpy.init(args=args)
    node = HazardMarkerDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl-C detected, shutting down node cleanly.")
    except Exception as e:
        node.get_logger().error(f"Node errored out: {e}")
    finally:
        node.get_logger().info("Destroying node...")
        node.destroy_node()
        rclpy.shutdown()
        print("HazardMarkerDetector node shutdown complete.")

if __name__ == '__main__':
    main()