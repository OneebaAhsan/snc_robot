#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, CameraInfo
from visualization_msgs.msg import Marker
from std_msgs.msg import String, Header, Float32MultiArray
from typing import Union, List, Tuple, Optional
from geometry_msgs.msg import Point, PoseStamped, PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs 
from tf2_ros import Buffer, TransformListener
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from rclpy.duration import Duration
import math 
from find_object_2d.msg import ObjectsStamped

FIND_OBJECT_ID_TO_HAZARD_ID = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12,
}

HAZARD_ID_TO_NAME = {
    1: "Explosive", 2: "Flammable Gas", 3: "Non-Flammable Gas", 4: "Dangerous When Wet",
    5: "Flammable Solid", 6: "Spontaneously Combustible", 7: "Oxidizer", 8: "Organic Peroxide",
    9: "Inhalation Hazard", 10: "Poison", 11: "Radioactive", 12: "Corrosive"
}

class HazardMarkerDetector(Node):
    """
    Listens for hazard markers detected by find_object_2d and uses laser scan data
    to determine their positions in the map frame.
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
        self.last_laser_scan = None
        
        # Track detected hazard markers to avoid duplicates
        self.detected_hazards = {}  # dict to store hazard_id -> position
        
        # QoS Profiles
        qos_reliable = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        qos_sensor = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=5)
        qos_cam_info = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1, durability=QoSDurabilityPolicy.VOLATILE)

        # subscriptions 
        self.get_logger().info(f"Subscribing to Camera Info: {self.camera_info_topic}")
        self.cam_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, qos_cam_info)

        self.get_logger().info(f"Subscribing to Laser Scan: {self.laser_scan_topic}")
        self.laser_subscription = self.create_subscription(LaserScan, self.laser_scan_topic, self.laser_callback, qos_sensor)

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

        # Periodically publish a test marker for debugging
        self.test_marker_timer = self.create_timer(5.0, self.publish_test_marker)

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

    def laser_callback(self, msg: LaserScan):
        # stores the latest laser scan
        self.last_laser_scan = msg
        self.get_logger().debug(f'Received laser scan message with frame_id: {msg.header.frame_id}')

    def publish_test_marker(self):
        # Publish a static test marker to verify markers are visible in RViz
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "test_markers"
        marker.id = 999
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.5
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime = rclpy.duration.Duration(seconds=10).to_msg()
        
        self.marker_publisher.publish(marker)
        self.get_logger().debug("Published test marker")

    def find_object_callback(self, msg: ObjectsStamped):
        """
        Callback for find_object_2d detections.
        Processes detected objects, estimates their positions using laser data,
        and publishes markers at those positions.
        """
        if not msg.objects.data:
            self.get_logger().info("Empty detection message received.")
            self.publish_status("No hazards detected")
            return

        # Check if we have laser data and camera intrinsics
        if self.last_laser_scan is None:
            self.get_logger().warning("No laser scan data available yet.")
            self.publish_status("Waiting for laser scan data")
            return
        if self.camera_intrinsics is None:
            self.get_logger().warning("Camera intrinsics not available yet.")
            self.publish_status("Waiting for camera intrinsics")
            return

        self.get_logger().info(f"Received {len(msg.objects.data) // 12} objects detected!")

        for i in range(0, len(msg.objects.data), 12):
            object_id = int(msg.objects.data[i])
            hazard_name = HAZARD_ID_TO_NAME.get(object_id, "Unknown")
            
            self.get_logger().info(f"Processing object with ID: {object_id}")
            
            # Extract bounding box info from the homography matrix
            h = np.array(msg.objects.data[i + 2: i + 11]).reshape(3, 3)
            
            # Compute the bounding box center by applying the homography to the reference object's center
            # Assume the reference object is centered at (0, 0) with a size of 100x100 pixels
            ref_center = np.array([50, 50, 1])  # Homogeneous coordinates
            image_center = h @ ref_center  # Apply homography
            image_center = image_center / image_center[2]  # Normalize homogeneous coordinates
            image_center_x = image_center[0]
            image_center_y = image_center[1]
            
            self.get_logger().info(f"Object center in image: x={image_center_x:.2f}, y={image_center_y:.2f}")
            
            # Use camera intrinsics to compute the angle
            fx = self.camera_intrinsics['fx']
            cx = self.camera_intrinsics['cx']
            cy = self.camera_intrinsics['cy']
            fy = self.camera_intrinsics['fy']
            
            try:
                # Create a point in the camera frame (at a unit distance along the ray)
                point_camera = PointStamped()
                point_camera.header.frame_id = self.camera_optical_frame
                point_camera.header.stamp = self.get_clock().now().to_msg()
                point_camera.point.x = 1.0  # Unit distance along the ray
                point_camera.point.y = (image_center_x - cx) / fx  # Using pinhole camera model
                point_camera.point.z = (image_center_y - cy) / fy
                
                # Transform the point to the laser frame
                point_laser_dir = self.tf_buffer.transform(
                    point_camera,
                    self.last_laser_scan.header.frame_id,
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
                
                # Compute the angle in the laser frame
                angle = np.arctan2(point_laser_dir.point.y, point_laser_dir.point.x)
                self.get_logger().info(f"Computed angle in laser frame: {angle:.2f} radians")
                
                # Map the angle to a laser scan index
                index = int((angle - self.last_laser_scan.angle_min) / self.last_laser_scan.angle_increment)
                
                # Check if index is within valid range
                if 0 <= index < len(self.last_laser_scan.ranges):
                    # Get distance and apply a small correction factor
                    raw_distance = self.last_laser_scan.ranges[index]
                    distance = raw_distance * 0.95  # Apply a small correction factor
                    
                    # Validate distance
                    if not np.isfinite(distance) or not (self.last_laser_scan.range_min <= distance <= self.last_laser_scan.range_max):
                        self.get_logger().warn(f'Invalid range for object {object_id}: {distance}m')
                        self.publish_status(f"Detected {hazard_name} but invalid distance")
                        self.place_fallback_marker(object_id, hazard_name)
                        continue
                    
                    # Create point in laser frame
                    point_laser = PointStamped()
                    point_laser.header.frame_id = self.last_laser_scan.header.frame_id
                    point_laser.header.stamp = self.get_clock().now().to_msg()
                    point_laser.point.x = distance * np.cos(angle)
                    point_laser.point.y = distance * np.sin(angle)
                    point_laser.point.z = 0.1  # Lower height for more accurate placement
                    
                    self.get_logger().info(f"Estimated position in laser frame: x={point_laser.point.x:.2f}, y={point_laser.point.y:.2f}")
                    
                    # Transform to map frame and publish marker
                    try:
                        if self.tf_buffer.can_transform('map', point_laser.header.frame_id, rclpy.time.Time(), 
                                                    timeout=rclpy.duration.Duration(seconds=1.0)):
                            
                            point_map = self.tf_buffer.transform(
                                point_laser, 
                                'map', 
                                timeout=rclpy.duration.Duration(seconds=1.0)
                            )
                            
                            self.get_logger().info(f"Transformed to map: x={point_map.point.x:.2f}, y={point_map.point.y:.2f}, z={point_map.point.z:.2f}")
                            
                            # Save and publish marker
                            if object_id not in self.detected_hazards:
                                self.save_hazard_marker_position(object_id, hazard_name, point_map.point)
                                self.publish_marker(point_map.point, object_id, hazard_name)
                            else:
                                self.get_logger().info(f"Object ID {object_id} already detected, skipping new marker")
                        else:
                            self.get_logger().warn(f"Transform to map frame not ready, using fallback")
                            self.place_fallback_marker(object_id, hazard_name)
                    
                    except Exception as e:
                        self.get_logger().error(f"Error in transformation: {e}")
                        self.publish_status(f"Error processing {hazard_name}")
                        self.place_fallback_marker(object_id, hazard_name)
                else:
                    self.get_logger().warn(f"Object position {index} is outside laser scan range [0, {len(self.last_laser_scan.ranges)-1}]")
                    self.publish_status(f"Detected {hazard_name} but outside scan range")
                    self.place_fallback_marker(object_id, hazard_name)
            
            except Exception as e:
                self.get_logger().error(f"Error in angle computation or transform: {e}")
                self.publish_status(f"Error processing {hazard_name}")
                self.place_fallback_marker(object_id, hazard_name)


    def place_fallback_marker(self, object_id, hazard_name):
        """Place marker at a fixed location in front of the robot as fallback"""
        try:
            # Get robot position in map
            robot_transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            # Create a point 1m in front of robot
            point_map = PointStamped()
            point_map.header.frame_id = 'map'
            point_map.header.stamp = self.get_clock().now().to_msg()
            
            # Extract robot position
            x = robot_transform.transform.translation.x
            y = robot_transform.transform.translation.y
            z = robot_transform.transform.translation.z
            
            # Extract orientation quaternion
            qx = robot_transform.transform.rotation.x
            qy = robot_transform.transform.rotation.y
            qz = robot_transform.transform.rotation.z
            qw = robot_transform.transform.rotation.w
            
            # Convert quaternion to yaw angle
            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            # Place point 0.7m in front of robot
            point_map.point.x = x + 0.7 * math.cos(yaw)
            point_map.point.y = y + 0.7 * math.sin(yaw)
            point_map.point.z = 0.1  # Just above ground
            
            self.get_logger().info(f"Using fallback position: x={point_map.point.x:.2f}, y={point_map.point.y:.2f}, z={point_map.point.z:.2f}")
            
            if object_id not in self.detected_hazards:
                self.save_hazard_marker_position(object_id, hazard_name, point_map.point)
                self.publish_marker(point_map.point, object_id, hazard_name)
            else:
                self.get_logger().info(f"Object ID {object_id} already detected, skipping fallback marker")
        
        except Exception as e:
            self.get_logger().error(f"Fallback positioning also failed: {e}")
            self.publish_status(f"Could not place marker for {hazard_name}")

    # def find_object_callback(self, msg: ObjectsStamped):
    #     """
    #     Callback for find_object_2d detections.
    #     Processes detected objects, estimates their positions using laser data,
    #     and publishes markers at those positions.
    #     """
    #     if not msg.objects.data:
    #         self.get_logger().info("Empty detection message received.")
    #         self.publish_status("No hazards detected")
    #         return

    #     # Check if we have laser data
    #     if self.last_laser_scan is None:
    #         self.get_logger().warning("No laser scan data available yet.")
    #         self.publish_status("Waiting for laser scan data")
    #         return

    #     self.get_logger().info(f"Received {len(msg.objects.data) // 12} objects detected!")

    #     for i in range(0, len(msg.objects.data), 12):
    #         object_id = int(msg.objects.data[i])
    #         hazard_name = HAZARD_ID_TO_NAME.get(object_id, "Unknown")
            
    #         self.get_logger().info(f"Processing object with ID: {object_id}")
            
    #         # Extract bounding box info from the homography matrix
    #         h = np.array(msg.objects.data[i + 2: i + 11]).reshape(3, 3)
            
    #         # Get bounding box center and width
    #         bbox_x = h[0, 2]  # X coordinate from homography
    #         bbox_width = h[0, 0] / 10  # Width estimation from homography scale
            
    #         self.get_logger().info(f"Object bounding box x: {bbox_x}, width: {bbox_width}")
            
    #         # Convert to normalized position in image (0-1)
    #         image_width = 800.0  # Default width from find_object_2d
    #         if self.camera_intrinsics:
    #             image_width = self.camera_intrinsics['width']
                
    #         image_center_x = bbox_x + bbox_width / 2.0
    #         normalized_x = image_center_x / image_width
            
    #         # Map normalized position to laser scan angle with offset
    #         angle = self.last_laser_scan.angle_min + normalized_x * (self.last_laser_scan.angle_max - self.last_laser_scan.angle_min) + 0.1
    #         index = int((angle - self.last_laser_scan.angle_min) / self.last_laser_scan.angle_increment)
            
    #         # Check if index is within valid range
    #         if 0 <= index < len(self.last_laser_scan.ranges):
    #             distance = self.last_laser_scan.ranges[index]
                
    #             # Validate distance
    #             if not np.isfinite(distance) or not (self.last_laser_scan.range_min <= distance <= self.last_laser_scan.range_max):
    #                 self.get_logger().warn(f'Invalid range for object {object_id}: {distance}m')
    #                 self.publish_status(f"Detected {hazard_name} but invalid distance")
    #                 continue
                
    #             # Create point in laser frame
    #             point_laser = PointStamped()
    #             point_laser.header.frame_id = self.last_laser_scan.header.frame_id
    #             # Use current time for transform to avoid timestamp issues
    #             point_laser.header.stamp = self.get_clock().now().to_msg()
    #             point_laser.point.x = distance * np.cos(angle)
    #             point_laser.point.y = distance * np.sin(angle)
    #             point_laser.point.z = 0.3  # Slightly elevated position for visibility
                
    #             self.get_logger().info(f"Estimated position in laser frame: x={point_laser.point.x:.2f}, y={point_laser.point.y:.2f}")
                
    #             self.transform_and_place_marker(object_id, hazard_name, point_laser)
    #         else:
    #             self.get_logger().warn(f"Object position {index} is outside laser scan range [0, {len(self.last_laser_scan.ranges)-1}]")
    #             self.publish_status(f"Detected {hazard_name} but outside scan range")


    def transform_and_place_marker(self, object_id, hazard_name, point_laser):
        """Transform point and publish marker with improved error handling"""
        try:
            # Try to use the transform with latest available data
            if self.tf_buffer.can_transform('map', point_laser.header.frame_id, rclpy.time.Time(), 
                                        timeout=rclpy.duration.Duration(seconds=1.0)):
                
                point_map = self.tf_buffer.transform(
                    point_laser, 
                    'map', 
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
                
                self.get_logger().info(f"Transformed to map: x={point_map.point.x:.2f}, y={point_map.point.y:.2f}, z={point_map.point.z:.2f}")
                
                # Apply position adjustment
                point_map.point.y += 0.3
                
                # Save and publish marker
                if object_id not in self.detected_hazards:
                    self.save_hazard_marker_position(object_id, hazard_name, point_map.point)
                    self.publish_marker(point_map.point, object_id, hazard_name)
                else:
                    self.get_logger().info(f"Object ID {object_id} already detected, skipping new marker")
                
            else:
                # FALLBACK: Use direct placement relative to robot base
                try:
                    robot_transform = self.tf_buffer.lookup_transform(
                        'map',
                        'base_link',
                        rclpy.time.Time(),  # Use latest available transform
                        timeout=rclpy.duration.Duration(seconds=2.0)
                    )
                    
                    # Create a point in map frame directly
                    point_map = PointStamped()
                    point_map.header.frame_id = 'map'
                    point_map.header.stamp = self.get_clock().now().to_msg()
                    
                    # Extract robot position and orientation
                    x = robot_transform.transform.translation.x
                    y = robot_transform.transform.translation.y
                    z = robot_transform.transform.translation.z
                    
                    # Extract orientation quaternion for yaw
                    qx = robot_transform.transform.rotation.x
                    qy = robot_transform.transform.rotation.y
                    qz = robot_transform.transform.rotation.z
                    qw = robot_transform.transform.rotation.w
                    
                    # Convert quaternion to yaw angle
                    siny_cosp = 2.0 * (qw * qz + qx * qy)
                    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
                    yaw = math.atan2(siny_cosp, cosy_cosp)
                    
                    # Place point 0.5m in front of robot with offset
                    point_map.point.x = x + 0.5 * math.cos(yaw)
                    point_map.point.y = y + 0.5 * math.sin(yaw) + 0.3
                    point_map.point.z = z + 0.3
                    
                    self.get_logger().info(f"Using fallback position: x={point_map.point.x:.2f}, y={point_map.point.y:.2f}, z={point_map.point.z:.2f}")
                    
                    # Only save/publish if this is a new detection
                    if object_id not in self.detected_hazards:
                        self.save_hazard_marker_position(object_id, hazard_name, point_map.point)
                        self.publish_marker(point_map.point, object_id, hazard_name)
                    else:
                        self.get_logger().info(f"Object ID {object_id} already detected, skipping new marker")
                
                except Exception as e:
                    self.get_logger().error(f"Fallback positioning also failed: {e}")
                    self.publish_status(f"Detected {hazard_name} but couldn't position marker")
        
        except Exception as e:
            self.get_logger().error(f"Error in transformation: {e}")
            self.publish_status(f"Error processing {hazard_name}")

    def save_hazard_marker_position(self, hazard_id, hazard_name, position):
        """
        Publishes the hazard marker position to a topic and tracks detections to avoid duplicates.
        """
        # Check if position is None
        if position is None:
            self.get_logger().warning(f"Cannot save hazard marker for ID {hazard_id}: Position is None")
            return
            
        # Create a unique key for the hazard
        marker_key = f"{hazard_id}"
        
        # Check if we've already detected this hazard (using approximate position)
        already_detected = False
        for key, pos in self.detected_hazards.items():
            if key == marker_key:
                # Check if the positions are similar (within 0.2m)
                dist = math.sqrt((pos[0] - position.x)**2 + 
                                (pos[1] - position.y)**2 + 
                                (pos[2] - position.z)**2)
                if dist < 0.2:  # already detected within 20cm
                    already_detected = True
                    break

        # Create the formatted data string for the coordinate
        coord_str = f"ID: {hazard_id}, {hazard_name}, x: {position.x:.4f}, y: {position.y:.4f}, z: {position.z:.4f}"
        
        # Always publish to topic for real-time access
        coord_msg = String()
        coord_msg.data = coord_str
        self.coords_publisher.publish(coord_msg)
        
        # If new detection, save it and provide more feedback
        if not already_detected:
            self.detected_hazards[marker_key] = (position.x, position.y, position.z)

            # Print visibility indication
            terminal_output = f"NEW HAZARD MARKER DETECTED - ID: {hazard_id} ({hazard_name}) at position: x={position.x:.4f}, y={position.y:.4f}, z={position.z:.4f}"
            print("\n" + "="*80)
            print(terminal_output)
            print("="*80 + "\n")
            self.get_logger().info(f"New hazard detected: {coord_str}")
        else:
            self.get_logger().debug(f"Already detected hazard seen again: {coord_str}")

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
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 1.0  # Fully opaque for better visibility

        # Color assignment based on hazard type
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

        # Set a long lifetime so markers stay visible
        marker.lifetime = rclpy.duration.Duration(seconds=3600).to_msg()
        
        self.get_logger().info(f"Sending marker to publisher...")
        self.marker_publisher.publish(marker)
        self.get_logger().info(f"Marker published successfully!")

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

################################################

# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image, LaserScan, CameraInfo
# from visualization_msgs.msg import Marker
# from std_msgs.msg import String, Header, Float32MultiArray
# from typing import Union, List, Tuple, Optional
# from geometry_msgs.msg import Point, PoseStamped, PointStamped
# from cv_bridge import CvBridge
# import cv2
# import numpy as np
# import tf2_ros
# import tf2_geometry_msgs 
# from tf2_ros import Buffer, TransformListener
# from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
# from rclpy.duration import Duration
# import math 
# from find_object_2d.msg import ObjectsStamped

# FIND_OBJECT_ID_TO_HAZARD_ID = {
#     1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12,
# }

# HAZARD_ID_TO_NAME = {
#     1: "Explosive", 2: "Flammable Gas", 3: "Non-Flammable Gas", 4: "Dangerous When Wet",
#     5: "Flammable Solid", 6: "Spontaneously Combustible", 7: "Oxidizer", 8: "Organic Peroxide",
#     9: "Inhalation Hazard", 10: "Poison", 11: "Radioactive", 12: "Corrosive"
# }

# class HazardMarkerDetector(Node):
#     """
#     Listens for hazard markers detected by find_object_2d and uses laser scan data
#     to determine their positions in the map frame.
#     """
#     def __init__(self):
#         # initializes node, parameters, subscribers, publishers 
#         super().__init__('hazard_marker_detector')

#         # parameters
#         self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
#         self.declare_parameter('depth_image_topic', '/camera/depth/image_raw')
#         self.declare_parameter('laser_scan_topic', '/scan')
#         self.declare_parameter('find_object_topic', '/find_object/objects')
#         self.declare_parameter('hazard_marker_topic', '/hazards')
#         self.declare_parameter('coords_topic', '/hazard_coords')
#         self.declare_parameter('status_topic', '/snc_status')
#         self.declare_parameter('map_frame', 'map')
#         self.declare_parameter('camera_optical_frame', 'camera_color_optical_frame')
#         self.declare_parameter('marker_publish_dist_threshold', 0.2)

#         # parameter values
#         self.camera_info_topic = self.get_parameter('camera_info_topic').value
#         self.depth_image_topic = self.get_parameter('depth_image_topic').value
#         self.laser_scan_topic = self.get_parameter('laser_scan_topic').value
#         self.find_object_topic = '/objectsStamped'
#         self.hazard_marker_topic = self.get_parameter('hazard_marker_topic').value
#         self.status_topic = self.get_parameter('status_topic').value
#         self.coords_topic = self.get_parameter('coords_topic').value
#         self.map_frame = self.get_parameter('map_frame').value
#         self.camera_optical_frame = self.get_parameter('camera_optical_frame').value

#         # initialization
#         self.bridge = CvBridge()
#         self.tf_buffer = Buffer()
#         self.tf_listener = TransformListener(self.tf_buffer, self)
#         self.camera_intrinsics = None
#         self.last_laser_scan = None
        
#         # Track detected hazard markers to avoid duplicates
#         self.detected_hazards = {}  # dict to store hazard_id -> position
        
#         # QoS Profiles
#         qos_reliable = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=10)
#         qos_sensor = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=5)
#         qos_cam_info = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1, durability=QoSDurabilityPolicy.VOLATILE)

#         # subscriptions 
#         self.get_logger().info(f"Subscribing to Camera Info: {self.camera_info_topic}")
#         self.cam_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, qos_cam_info)

#         self.get_logger().info(f"Subscribing to Laser Scan: {self.laser_scan_topic}")
#         self.laser_subscription = self.create_subscription(LaserScan, self.laser_scan_topic, self.laser_callback, qos_sensor)

#         # crucial subscription
#         self.get_logger().info(f"Attempting to subscribe to Find Object: {self.find_object_topic}")
#         self.find_object_sub = self.create_subscription(
#             ObjectsStamped,
#             self.find_object_topic,
#             self.find_object_callback,
#             10
#         )

#         # publishers 
#         self.get_logger().info(f"Publisher initialized for Hazard Markers: {self.hazard_marker_topic}")
#         self.marker_publisher = self.create_publisher(Marker, self.hazard_marker_topic, qos_reliable)

#         self.get_logger().info(f"Publisher initialized for Status: {self.status_topic}")
#         self.status_publisher = self.create_publisher(String, self.status_topic, qos_reliable)
        
#         self.get_logger().info(f"Publisher initialized for Hazard Coordinates: {self.coords_topic}")
#         self.coords_publisher = self.create_publisher(String, self.coords_topic, qos_reliable)

#         self.get_logger().info('Hazard Marker Detector Node Initialized.')
#         self.publish_status("Initializing and waiting for find_object messages...")

#     def publish_status(self, status_text):
#         # helper function to publish a status message and log it
#         msg = String()
#         msg.data = f"Node2: {status_text}"
#         self.status_publisher.publish(msg)
#         self.get_logger().info(f"Status: {status_text}")

#     def camera_info_callback(self, msg: CameraInfo):
#         # stores camera intrinsic parameters when received
#         if self.camera_intrinsics is None:
#             self.get_logger().info("Received camera intrinsics.")
#             self.camera_intrinsics = { 'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5], 'width': msg.width, 'height': msg.height }
#             self.publish_status("Camera info received")

#     def laser_callback(self, msg: LaserScan):
#         # stores the latest laser scan
#         self.last_laser_scan = msg
#         self.get_logger().debug(f'Received laser scan message with frame_id: {msg.header.frame_id}')

#     def find_object_callback(self, msg: ObjectsStamped):
#         """
#         Callback for find_object_2d detections.
#         Processes detected objects, estimates their positions using laser data,
#         and publishes markers at those positions.
#         """
#         if not msg.objects.data:
#             self.get_logger().info("Empty detection message received.")
#             self.publish_status("No hazards detected")
#             return

#         # Check if we have laser data
#         if self.last_laser_scan is None:
#             self.get_logger().warning("No laser scan data available yet.")
#             self.publish_status("Waiting for laser scan data")
#             return

#         self.get_logger().info(f"Received {len(msg.objects.data) // 12} objects detected!")

#         for i in range(0, len(msg.objects.data), 12):
#             object_id = int(msg.objects.data[i])
#             hazard_name = HAZARD_ID_TO_NAME.get(object_id, "Unknown")
            
#             self.get_logger().info(f"Processing object with ID: {object_id}")
            
#             # Extract bounding box info from the homography matrix
#             h = np.array(msg.objects.data[i + 2: i + 11]).reshape(3, 3)
            
#             # Get bounding box center and width
#             # We're extracting this from the homography matrix's components
#             # These values help determine the object's position in the image
#             bbox_x = h[0, 2]  # X coordinate from homography
#             bbox_width = h[0, 0] / 10  # Width estimation from homography scale
            
#             self.get_logger().info(f"Object bounding box x: {bbox_x}, width: {bbox_width}")
            
#             # Convert to normalized position in image (0-1)
#             image_width = 800.0  # Default width from find_object_2d
#             if self.camera_intrinsics:
#                 image_width = self.camera_intrinsics['width']
                
#             image_center_x = bbox_x + bbox_width / 2.0
#             normalized_x = image_center_x / image_width
            
#             # Map normalized position to laser scan angle
#             angle = self.last_laser_scan.angle_min + normalized_x * (self.last_laser_scan.angle_max - self.last_laser_scan.angle_min)
#             index = int((angle - self.last_laser_scan.angle_min) / self.last_laser_scan.angle_increment)
            
#             # Check if index is within valid range
#             if 0 <= index < len(self.last_laser_scan.ranges):
#                 distance = self.last_laser_scan.ranges[index]
                
#                 # Validate distance
#                 if not np.isfinite(distance) or not (self.last_laser_scan.range_min <= distance <= self.last_laser_scan.range_max):
#                     self.get_logger().warn(f'Invalid range for object {object_id}: {distance}m')
#                     self.publish_status(f"Detected {hazard_name} but invalid distance")
#                     continue
                
#                 # Create point in laser frame
#                 point_laser = PointStamped()
#                 point_laser.header.frame_id = self.last_laser_scan.header.frame_id
#                 point_laser.header.stamp = self.last_laser_scan.header.stamp
#                 point_laser.point.x = distance * np.cos(angle)
#                 point_laser.point.y = distance * np.sin(angle)
#                 point_laser.point.z = 0.3  # Slightly elevated position for visibility
                
#                 self.get_logger().info(f"Estimated position in laser frame: x={point_laser.point.x:.2f}, y={point_laser.point.y:.2f}")
                
#                 # Transform to map frame with robust error handling
#                 try:
#                     # First, try the standard transform approach with increased timeout
#                     if self.tf_buffer.can_transform('map', point_laser.header.frame_id, rclpy.time.Time(), 
#                                                 timeout=rclpy.duration.Duration(seconds=1.0)):
                        
#                         # using latest available transform instead of the exact timestamp
#                         point_laser.header.stamp = self.get_clock().now().to_msg()   
#                         point_map = self.tf_buffer.transform(
#                             point_laser, 
#                             'map', 
#                             timeout=rclpy.duration.Duration(seconds=1.0)
#                         )
                        
#                         self.get_logger().info(f"Transformed to map: x={point_map.point.x:.2f}, y={point_map.point.y:.2f}, z={point_map.point.z:.2f}")
                        
#                         # Add offset to adjust marker position (shift right on the map)
#                         point_map.point.y += 0.3
                        
#                         # Save hazard marker position and publish marker
#                         # Only save/publish if this is a new detection or we want to update
#                         if object_id not in self.detected_hazards:
#                             self.save_hazard_marker_position(object_id, hazard_name, point_map.point)
#                             self.publish_marker(point_map.point, object_id, hazard_name)
#                         else:
#                             self.get_logger().info(f"Object ID {object_id} already detected, skipping new marker")
                            
#                     else:
#                         # FALLBACK: If transform isn't available, use a fixed offset from the robot's base frame
#                         self.get_logger().warn(f"Transform to map frame not ready, using fallback positioning")
                        
#                         # Try to get the robot's position in the map
#                         try:
#                             robot_transform = self.tf_buffer.lookup_transform(
#                                 'map',
#                                 'base_link',  # Assuming this is the robot's base frame
#                                 rclpy.time.Time(),
#                                 timeout=rclpy.duration.Duration(seconds=2.0)   # increased time out from 1.0 to 2.0
#                             )
                            
#                             # Create a point in map frame directly
#                             point_map = PointStamped()
#                             point_map.header.frame_id = 'map'
#                             point_map.header.stamp = self.get_clock().now().to_msg()
                            
#                             # Extract robot position and orientation
#                             x = robot_transform.transform.translation.x
#                             y = robot_transform.transform.translation.y
#                             z = robot_transform.transform.translation.z
                            
#                             # Extract orientation quaternion
#                             qx = robot_transform.transform.rotation.x
#                             qy = robot_transform.transform.rotation.y
#                             qz = robot_transform.transform.rotation.z
#                             qw = robot_transform.transform.rotation.w
                            
#                             # Convert quaternion to yaw angle
#                             siny_cosp = 2.0 * (qw * qz + qx * qy)
#                             cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
#                             yaw = math.atan2(siny_cosp, cosy_cosp)
                            
#                             # Place point 0.5m in front of robot
#                             point_map.point.x = x + 0.5 * math.cos(yaw)
#                             point_map.point.y = y + 0.5 * math.sin(yaw) + 0.3  # Add offset to shift right
#                             point_map.point.z = z + 0.3  # Slightly above ground
                            
#                             self.get_logger().info(f"Using fallback position: x={point_map.point.x:.2f}, y={point_map.point.y:.2f}, z={point_map.point.z:.2f}")
                            
#                             # Only save/publish if this is a new detection
#                             if object_id not in self.detected_hazards:
#                                 self.save_hazard_marker_position(object_id, hazard_name, point_map.point)
#                                 self.publish_marker(point_map.point, object_id, hazard_name)
#                             else:
#                                 self.get_logger().info(f"Object ID {object_id} already detected, skipping new marker")
                        
#                         except Exception as e:
#                             self.get_logger().error(f"Fallback positioning also failed: {e}")
#                             self.publish_status(f"Detected {hazard_name} but couldn't position marker")
                
#                 except Exception as e:
#                     self.get_logger().error(f"Error in transformation: {e}")
#                     self.publish_status(f"Error processing {hazard_name}")
                    
#             else:
#                 self.get_logger().warn(f"Object position {index} is outside laser scan range [0, {len(self.last_laser_scan.ranges)-1}]")
#                 self.publish_status(f"Detected {hazard_name} but outside scan range")

#     # def find_object_callback(self, msg: ObjectsStamped):
#     #     """
#     #     Callback for find_object_2d detections.
#     #     Processes detected objects, estimates their positions using laser data,
#     #     and publishes markers at those positions.
#     #     """
#     #     if not msg.objects.data:
#     #         self.get_logger().info("Empty detection message received.")
#     #         self.publish_status("No hazards detected")
#     #         return

#     #     # Check if we have laser data
#     #     if self.last_laser_scan is None:
#     #         self.get_logger().warning("No laser scan data available yet.")
#     #         self.publish_status("Waiting for laser scan data")
#     #         return

#     #     self.get_logger().info(f"Received {len(msg.objects.data) // 12} objects detected!")

#     #     for i in range(0, len(msg.objects.data), 12):
#     #         object_id = int(msg.objects.data[i])
#     #         hazard_name = HAZARD_ID_TO_NAME.get(object_id, "Unknown")
            
#     #         self.get_logger().info(f"Processing object with ID: {object_id}")
            
#     #         # Extract bounding box info from the homography matrix
#     #         h = np.array(msg.objects.data[i + 2: i + 11]).reshape(3, 3)
            
#     #         # Get bounding box center and width
#     #         # We're extracting this from the homography matrix's components
#     #         # These values help determine the object's position in the image
#     #         bbox_x = h[0, 2]  # X coordinate from homography
#     #         bbox_width = h[0, 0] / 10  # Width estimation from homography scale
            
#     #         self.get_logger().info(f"Object bounding box x: {bbox_x}, width: {bbox_width}")
            
#     #         # Convert to normalized position in image (0-1)
#     #         image_width = 800.0  # Default width from find_object_2d
#     #         if self.camera_intrinsics:
#     #             image_width = self.camera_intrinsics['width']
                
#     #         image_center_x = bbox_x + bbox_width / 2.0
#     #         ########## test on real world. adjusting widhth, remove if useless
#     #         image_width = self.camera_intrinsics['width'] if self.camera_intrinsics else 800.0   
#     #         normalized_x = image_center_x / image_width
            
#     #         # Map normalized position to laser scan angle
#     #         # angle = self.last_laser_scan.angle_min + normalized_x * (self.last_laser_scan.angle_max - self.last_laser_scan.angle_min)
#     #         # Add a small correction factor (positive moves right, negative moves left)
#     #         # angle = self.last_laser_scan.angle_min + normalized_x * (self.last_laser_scan.angle_max - self.last_laser_scan.angle_min) + 0.1  # Add offset here 
#     #         # Where you calculate the angle in find_object_callback
#     #         # ====== test on real world to check 
#     #         angle = self.last_laser_scan.angle_min + normalized_x * (self.last_laser_scan.angle_max - self.last_laser_scan.angle_min) + 0.2  
#     #         index = int((angle - self.last_laser_scan.angle_min) / self.last_laser_scan.angle_increment)
            
#     #         # Check if index is within valid range
#     #         if 0 <= index < len(self.last_laser_scan.ranges):
#     #             distance = self.last_laser_scan.ranges[index]
                
#     #             # Validate distance
#     #             if not np.isfinite(distance) or not (self.last_laser_scan.range_min <= distance <= self.last_laser_scan.range_max):
#     #                 self.get_logger().warn(f'Invalid range for object {object_id}: {distance}m')
#     #                 self.publish_status(f"Detected {hazard_name} but invalid distance")
#     #                 continue
                
#     #             # Create point in laser frame
#     #             point_laser = PointStamped()
#     #             point_laser.header.frame_id = self.last_laser_scan.header.frame_id
#     #             point_laser.header.stamp = self.last_laser_scan.header.stamp
#     #             point_laser.point.x = distance * np.cos(angle) + 0.05
#     #             point_laser.point.x = distance * np.cos(angle)
#     #             point_laser.point.y = distance * np.sin(angle)
#     #             point_laser.point.z = 0.3  # Slightly elevated position for visibility
                
#     #             self.get_logger().info(f"Estimated position in laser frame: x={point_laser.point.x:.2f}, y={point_laser.point.y:.2f}")
                
#     #             # Transform to map frame
#     #             try:
#     #                 if self.tf_buffer.can_transform('map', point_laser.header.frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)):
#     #                     point_map = self.tf_buffer.transform(point_laser, 'map', timeout=rclpy.duration.Duration(seconds=0.1))
                        
#     #                     self.get_logger().info(f"Transformed to map: x={point_map.point.x:.2f}, y={point_map.point.y:.2f}, z={point_map.point.z:.2f}")

#     #                     #### test on real world. for adjusing marker position
#     #                     point_map.point.y += 0.3  # Adjust Y coordinate to shift right in the map frame
                        
#     #                     # Save hazard marker position and publish marker
#     #                     self.save_hazard_marker_position(object_id, hazard_name, point_map.point)
#     #                     self.publish_marker(point_map.point, object_id, hazard_name)
#     #                 else:
#     #                     self.get_logger().warn(f"Transform to map frame not ready for object {object_id}")
#     #                     self.publish_status(f"Detected {hazard_name} but transform not ready")
#     #             except Exception as e:
#     #                 self.get_logger().error(f"Error transforming to map frame: {e}")
#     #                 self.publish_status(f"Detected {hazard_name} but error in transformation")
#     #         else:
#     #             self.get_logger().warn(f"Object position {index} is outside laser scan range [0, {len(self.last_laser_scan.ranges)-1}]")
#     #             self.publish_status(f"Detected {hazard_name} but outside scan range")

#     def save_hazard_marker_position(self, hazard_id, hazard_name, position):
#         """
#         Publishes the hazard marker position to a topic and tracks detections to avoid duplicates.
#         """
#         # Check if position is None
#         if position is None:
#             self.get_logger().warning(f"Cannot save hazard marker for ID {hazard_id}: Position is None")
#             return
            
#         # Create a unique key for the hazard
#         marker_key = f"{hazard_id}"
        
#         # Check if we've already detected this hazard (using approximate position)
#         already_detected = False
#         for key, pos in self.detected_hazards.items():
#             if key == marker_key:
#                 # Check if the positions are similar (within 0.2m)
#                 dist = math.sqrt((pos[0] - position.x)**2 + 
#                                 (pos[1] - position.y)**2 + 
#                                 (pos[2] - position.z)**2)
#                 if dist < 0.2:  # already detected within 20cm
#                     already_detected = True
#                     break

#         # Create the formatted data string for the coordinate
#         coord_str = f"ID: {hazard_id}, {hazard_name}, x: {position.x:.4f}, y: {position.y:.4f}, z: {position.z:.4f}"
        
#         # Always publish to topic for real-time access
#         coord_msg = String()
#         coord_msg.data = coord_str
#         self.coords_publisher.publish(coord_msg)
        
#         # If new detection, save it and provide more feedback
#         if not already_detected:
#             self.detected_hazards[marker_key] = (position.x, position.y, position.z)

#             # Print visibility indication
#             terminal_output = f"NEW HAZARD MARKER DETECTED - ID: {hazard_id} ({hazard_name}) at position: x={position.x:.4f}, y={position.y:.4f}, z={position.z:.4f}"
#             print("\n" + "="*80)
#             print(terminal_output)
#             print("="*80 + "\n")
#             self.get_logger().info(f"New hazard detected: {coord_str}")
#         else:
#             self.get_logger().debug(f"Already detected hazard seen again: {coord_str}")

#     def publish_marker(self, position_in_map, marker_id, marker_name):
#         """Publishes a visualization_msgs/Marker."""
#         marker_id_int = int(marker_id)
#         self.get_logger().info(f"Publishing marker for {marker_name} (ID: {marker_id_int}) at map coordinates: x={position_in_map.x:.3f}, y={position_in_map.y:.3f}, z={position_in_map.z:.3f}")

#         marker = Marker()
#         marker.header.frame_id = self.map_frame
#         marker.header.stamp = self.get_clock().now().to_msg()
#         marker.ns = "hazard_markers"
#         marker.id = marker_id_int
#         marker.type = Marker.SPHERE
#         marker.action = Marker.ADD
#         marker.pose.position = position_in_map
#         marker.pose.orientation.w = 1.0
#         marker.scale.x = 0.3
#         marker.scale.y = 0.3
#         marker.scale.z = 0.3
#         marker.color.a = 0.8

#         # Color assignment based on hazard type
#         if "explosive" in marker_name.lower() or "flammable" in marker_name.lower() or "oxidizer" in marker_name.lower():
#             marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0 # Red
#         elif "gas" in marker_name.lower():
#              marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 1.0 # Blue
#         elif "corrosive" in marker_name.lower():
#              marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 0.0 # Yellow
#         elif "poison" in marker_name.lower() or "inhalation" in marker_name.lower():
#              marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 0.0 # Black
#         elif "radioactive" in marker_name.lower():
#              marker.color.r = 1.0; marker.color.g = 0.5; marker.color.b = 0.0 # Orange
#         else:
#             marker.color.r = 0.5; marker.color.g = 0.0; marker.color.b = 0.5 # Purple

#         # marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
#         marker.lifetime = rclpy.duration.Duration(seconds=3600).to_msg()  # 1 hour
        
#         self.get_logger().info(f"Sending marker to publisher...")
#         self.marker_publisher.publish(marker)
#         self.get_logger().info(f"Marker published successfully!")

# def main(args=None):
#     rclpy.init(args=args)
#     node = HazardMarkerDetector()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info("Ctrl-C detected, shutting down node cleanly.")
#     except Exception as e:
#         node.get_logger().error(f"Node errored out: {e}")
#     finally:
#         node.get_logger().info("Destroying node...")
#         node.destroy_node()
#         rclpy.shutdown()
#         print("HazardMarkerDetector node shutdown complete.")

# if __name__ == '__main__':
#     main()


########################################################3



# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image, LaserScan, CameraInfo
# from visualization_msgs.msg import Marker
# from std_msgs.msg import String, Header, Float32MultiArray
# from typing import Union, List, Tuple, Optional
# from geometry_msgs.msg import Point, PoseStamped, PointStamped
# from cv_bridge import CvBridge
# import cv2
# import numpy as np
# import tf2_ros
# import tf2_geometry_msgs 
# from tf2_ros import Buffer, TransformListener
# from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
# from rclpy.duration import Duration
# import math 
# from find_object_2d.msg import ObjectsStamped

# FIND_OBJECT_ID_TO_HAZARD_ID = {
#     1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12,
# }

# HAZARD_ID_TO_NAME = {
#     1: "Explosive", 2: "Flammable Gas", 3: "Non-Flammable Gas", 4: "Dangerous When Wet",
#     5: "Flammable Solid", 6: "Spontaneously Combustible", 7: "Oxidizer", 8: "Organic Peroxide",
#     9: "Inhalation Hazard", 10: "Poison", 11: "Radioactive", 12: "Corrosive"
# }

# class HazardMarkerDetector(Node):
#     """
#     Listens for hazard markers detected by find_object_2d and uses laser scan data
#     to determine their positions in the map frame.
#     """
#     def __init__(self):
#         # initializes node, parameters, subscribers, publishers 
#         super().__init__('hazard_marker_detector')

#         # parameters
#         self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
#         self.declare_parameter('depth_image_topic', '/camera/depth/image_raw')
#         self.declare_parameter('laser_scan_topic', '/scan')
#         self.declare_parameter('find_object_topic', '/find_object/objects')
#         self.declare_parameter('hazard_marker_topic', '/hazards')
#         self.declare_parameter('coords_topic', '/hazard_coords')
#         self.declare_parameter('status_topic', '/snc_status')
#         self.declare_parameter('map_frame', 'map')
#         self.declare_parameter('camera_optical_frame', 'camera_color_optical_frame')
#         self.declare_parameter('marker_publish_dist_threshold', 0.2)

#         # parameter values
#         self.camera_info_topic = self.get_parameter('camera_info_topic').value
#         self.depth_image_topic = self.get_parameter('depth_image_topic').value
#         self.laser_scan_topic = self.get_parameter('laser_scan_topic').value
#         self.find_object_topic = '/objectsStamped'
#         self.hazard_marker_topic = self.get_parameter('hazard_marker_topic').value
#         self.status_topic = self.get_parameter('status_topic').value
#         self.coords_topic = self.get_parameter('coords_topic').value
#         self.map_frame = self.get_parameter('map_frame').value
#         self.camera_optical_frame = self.get_parameter('camera_optical_frame').value

#         # initialization
#         self.bridge = CvBridge()
#         self.tf_buffer = Buffer()
#         self.tf_listener = TransformListener(self.tf_buffer, self)
#         self.camera_intrinsics = None
#         self.last_laser_scan = None
        
#         # Track detected hazard markers to avoid duplicates
#         self.detected_hazards = {}  # dict to store hazard_id -> position
        
#         # QoS Profiles
#         qos_reliable = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=10)
#         qos_sensor = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=5)
#         qos_cam_info = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1, durability=QoSDurabilityPolicy.VOLATILE)

#         # subscriptions 
#         self.get_logger().info(f"Subscribing to Camera Info: {self.camera_info_topic}")
#         self.cam_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, qos_cam_info)

#         self.get_logger().info(f"Subscribing to Laser Scan: {self.laser_scan_topic}")
#         self.laser_subscription = self.create_subscription(LaserScan, self.laser_scan_topic, self.laser_callback, qos_sensor)

#         # crucial subscription
#         self.get_logger().info(f"Attempting to subscribe to Find Object: {self.find_object_topic}")
#         self.find_object_sub = self.create_subscription(
#             ObjectsStamped,
#             self.find_object_topic,
#             self.find_object_callback,
#             10
#         )

#         # publishers 
#         self.get_logger().info(f"Publisher initialized for Hazard Markers: {self.hazard_marker_topic}")
#         self.marker_publisher = self.create_publisher(Marker, self.hazard_marker_topic, qos_reliable)

#         self.get_logger().info(f"Publisher initialized for Status: {self.status_topic}")
#         self.status_publisher = self.create_publisher(String, self.status_topic, qos_reliable)
        
#         self.get_logger().info(f"Publisher initialized for Hazard Coordinates: {self.coords_topic}")
#         self.coords_publisher = self.create_publisher(String, self.coords_topic, qos_reliable)

#         self.get_logger().info('Hazard Marker Detector Node Initialized.')
#         self.publish_status("Initializing and waiting for find_object messages...")

#     def publish_status(self, status_text):
#         # helper function to publish a status message and log it
#         msg = String()
#         msg.data = f"Node2: {status_text}"
#         self.status_publisher.publish(msg)
#         self.get_logger().info(f"Status: {status_text}")

#     def camera_info_callback(self, msg: CameraInfo):
#         # stores camera intrinsic parameters when received
#         if self.camera_intrinsics is None:
#             self.get_logger().info("Received camera intrinsics.")
#             self.camera_intrinsics = { 'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5], 'width': msg.width, 'height': msg.height }
#             self.publish_status("Camera info received")

#     def laser_callback(self, msg: LaserScan):
#         # stores the latest laser scan
#         self.last_laser_scan = msg
#         self.get_logger().debug(f'Received laser scan message with frame_id: {msg.header.frame_id}')

#     def find_object_callback(self, msg: ObjectsStamped):
#         """
#         Callback for find_object_2d detections.
#         Processes detected objects, estimates their positions using laser data,
#         and publishes markers at those positions.
#         """
#         if not msg.objects.data:
#             self.get_logger().info("Empty detection message received.")
#             self.publish_status("No hazards detected")
#             return

#         # Check if we have laser data
#         if self.last_laser_scan is None:
#             self.get_logger().warning("No laser scan data available yet.")
#             self.publish_status("Waiting for laser scan data")
#             return

#         self.get_logger().info(f"Received {len(msg.objects.data) // 12} objects detected!")

#         for i in range(0, len(msg.objects.data), 12):
#             object_id = int(msg.objects.data[i])
#             hazard_name = HAZARD_ID_TO_NAME.get(object_id, "Unknown")
            
#             self.get_logger().info(f"Processing object with ID: {object_id}")
            
#             # Extract bounding box info from the homography matrix
#             h = np.array(msg.objects.data[i + 2: i + 11]).reshape(3, 3)
            
#             # Get bounding box center and width
#             # We're extracting this from the homography matrix's components
#             # These values help determine the object's position in the image
#             bbox_x = h[0, 2]  # X coordinate from homography
#             bbox_width = h[0, 0] / 10  # Width estimation from homography scale
            
#             self.get_logger().info(f"Object bounding box x: {bbox_x}, width: {bbox_width}")
            
#             # Convert to normalized position in image (0-1)
#             image_width = 800.0  # Default width from find_object_2d
#             if self.camera_intrinsics:
#                 image_width = self.camera_intrinsics['width']
                
#             image_center_x = bbox_x + bbox_width / 2.0
#             normalized_x = image_center_x / image_width
            
#             # Map normalized position to laser scan angle
#             angle = self.last_laser_scan.angle_min + normalized_x * (self.last_laser_scan.angle_max - self.last_laser_scan.angle_min)
#             index = int((angle - self.last_laser_scan.angle_min) / self.last_laser_scan.angle_increment)
            
#             # Check if index is within valid range
#             if 0 <= index < len(self.last_laser_scan.ranges):
#                 distance = self.last_laser_scan.ranges[index]
                
#                 # Validate distance
#                 if not np.isfinite(distance) or not (self.last_laser_scan.range_min <= distance <= self.last_laser_scan.range_max):
#                     self.get_logger().warn(f'Invalid range for object {object_id}: {distance}m')
#                     self.publish_status(f"Detected {hazard_name} but invalid distance")
#                     continue
                
#                 # Create point in laser frame
#                 point_laser = PointStamped()
#                 point_laser.header.frame_id = self.last_laser_scan.header.frame_id
#                 point_laser.header.stamp = self.last_laser_scan.header.stamp
#                 point_laser.point.x = distance * np.cos(angle)
#                 point_laser.point.y = distance * np.sin(angle)
#                 point_laser.point.z = 0.3  # Slightly elevated position for visibility
                
#                 self.get_logger().info(f"Estimated position in laser frame: x={point_laser.point.x:.2f}, y={point_laser.point.y:.2f}")
                
#                 # Transform to map frame
#                 try:
#                     if self.tf_buffer.can_transform('map', point_laser.header.frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)):
#                         point_map = self.tf_buffer.transform(point_laser, 'map', timeout=rclpy.duration.Duration(seconds=0.1))
                        
#                         self.get_logger().info(f"Transformed to map: x={point_map.point.x:.2f}, y={point_map.point.y:.2f}, z={point_map.point.z:.2f}")
                        
#                         # Save hazard marker position and publish marker
#                         self.save_hazard_marker_position(object_id, hazard_name, point_map.point)
#                         self.publish_marker(point_map.point, object_id, hazard_name)
#                     else:
#                         self.get_logger().warn(f"Transform to map frame not ready for object {object_id}")
#                         self.publish_status(f"Detected {hazard_name} but transform not ready")
#                 except Exception as e:
#                     self.get_logger().error(f"Error transforming to map frame: {e}")
#                     self.publish_status(f"Detected {hazard_name} but error in transformation")
#             else:
#                 self.get_logger().warn(f"Object position {index} is outside laser scan range [0, {len(self.last_laser_scan.ranges)-1}]")
#                 self.publish_status(f"Detected {hazard_name} but outside scan range")

#     def save_hazard_marker_position(self, hazard_id, hazard_name, position):
#         """
#         Publishes the hazard marker position to a topic and tracks detections to avoid duplicates.
#         """
#         # Check if position is None
#         if position is None:
#             self.get_logger().warning(f"Cannot save hazard marker for ID {hazard_id}: Position is None")
#             return
            
#         # Create a unique key for the hazard
#         marker_key = f"{hazard_id}"
        
#         # Check if we've already detected this hazard (using approximate position)
#         already_detected = False
#         for key, pos in self.detected_hazards.items():
#             if key == marker_key:
#                 # Check if the positions are similar (within 0.2m)
#                 dist = math.sqrt((pos[0] - position.x)**2 + 
#                                 (pos[1] - position.y)**2 + 
#                                 (pos[2] - position.z)**2)
#                 if dist < 0.2:  # already detected within 20cm
#                     already_detected = True
#                     break

#         # Create the formatted data string for the coordinate
#         coord_str = f"ID: {hazard_id}, {hazard_name}, x: {position.x:.4f}, y: {position.y:.4f}, z: {position.z:.4f}"
        
#         # Always publish to topic for real-time access
#         coord_msg = String()
#         coord_msg.data = coord_str
#         self.coords_publisher.publish(coord_msg)
        
#         # If new detection, save it and provide more feedback
#         if not already_detected:
#             self.detected_hazards[marker_key] = (position.x, position.y, position.z)

#             # Print visibility indication
#             terminal_output = f"NEW HAZARD MARKER DETECTED - ID: {hazard_id} ({hazard_name}) at position: x={position.x:.4f}, y={position.y:.4f}, z={position.z:.4f}"
#             print("\n" + "="*80)
#             print(terminal_output)
#             print("="*80 + "\n")
#             self.get_logger().info(f"New hazard detected: {coord_str}")
#         else:
#             self.get_logger().debug(f"Already detected hazard seen again: {coord_str}")

#     def publish_marker(self, position_in_map, marker_id, marker_name):
#         """Publishes a visualization_msgs/Marker."""
#         marker_id_int = int(marker_id)
#         self.get_logger().info(f"Publishing marker for {marker_name} (ID: {marker_id_int}) at map coordinates: x={position_in_map.x:.3f}, y={position_in_map.y:.3f}, z={position_in_map.z:.3f}")

#         marker = Marker()
#         marker.header.frame_id = self.map_frame
#         marker.header.stamp = self.get_clock().now().to_msg()
#         marker.ns = "hazard_markers"
#         marker.id = marker_id_int
#         marker.type = Marker.SPHERE
#         marker.action = Marker.ADD
#         marker.pose.position = position_in_map
#         marker.pose.orientation.w = 1.0
#         marker.scale.x = 0.3
#         marker.scale.y = 0.3
#         marker.scale.z = 0.3
#         marker.color.a = 0.8

#         # Color assignment based on hazard type
#         if "explosive" in marker_name.lower() or "flammable" in marker_name.lower() or "oxidizer" in marker_name.lower():
#             marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0 # Red
#         elif "gas" in marker_name.lower():
#              marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 1.0 # Blue
#         elif "corrosive" in marker_name.lower():
#              marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 0.0 # Yellow
#         elif "poison" in marker_name.lower() or "inhalation" in marker_name.lower():
#              marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 0.0 # Black
#         elif "radioactive" in marker_name.lower():
#              marker.color.r = 1.0; marker.color.g = 0.5; marker.color.b = 0.0 # Orange
#         else:
#             marker.color.r = 0.5; marker.color.g = 0.0; marker.color.b = 0.5 # Purple

#         marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
        
#         self.get_logger().info(f"Sending marker to publisher...")
#         self.marker_publisher.publish(marker)
#         self.get_logger().info(f"Marker published successfully!")

# def main(args=None):
#     rclpy.init(args=args)
#     node = HazardMarkerDetector()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info("Ctrl-C detected, shutting down node cleanly.")
#     except Exception as e:
#         node.get_logger().error(f"Node errored out: {e}")
#     finally:
#         node.get_logger().info("Destroying node...")
#         node.destroy_node()
#         rclpy.shutdown()
#         print("HazardMarkerDetector node shutdown complete.")

# if __name__ == '__main__':
#     main()