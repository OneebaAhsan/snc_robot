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
import os

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
    Listens for hazard markers detected by find_object_2d and sensor data.
    Uses laser scanner data to determine the position of detected hazard markers.
    """
    def __init__(self):
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
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('laser_frame', 'laser')
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
        self.base_frame = self.get_parameter('base_frame').value
        self.laser_frame = self.get_parameter('laser_frame').value

        # initialization
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.camera_intrinsics = None
        self.last_depth_image = None
        self.last_depth_header = None
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
        """Stores the latest laser scan and its frame ID"""
        self.get_logger().debug('Received laser scan message.')
        self.last_laser_scan = msg
        self.laser_frame = msg.header.frame_id
        self.get_logger().debug(f'Laser frame ID: {self.laser_frame}')

    def find_object_callback(self, msg: ObjectsStamped):
        """
        Callback for find_object_2d detections.
        Extracts object ID and uses laser scan data to estimate position.
        """
        if not msg.objects.data:
            self.get_logger().info("Empty detection message received.")
            self.publish_status("No hazards detected")
            return

        self.get_logger().info(f"Received {len(msg.objects.data) // 12} objects detected!")

        # Check if we have laser data
        if self.last_laser_scan is None:
            self.get_logger().warning("No laser scan data available yet.")
            self.publish_status("Waiting for laser scan data")
            return

        for i in range(0, len(msg.objects.data), 12):
            object_id = int(msg.objects.data[i])
            hazard_name = HAZARD_ID_TO_NAME.get(object_id, "Unknown")
            
            self.get_logger().info(f"Processing object with ID: {object_id} ({hazard_name})")
            
            # Use laser to estimate the position
            # We'll assume the hazard is in front of the robot when detected
            position_map = self.estimate_position_from_laser(msg.header)
            
            if position_map:
                self.get_logger().info(f"Estimated map position: {position_map}")
                
                # Save hazard marker position and publish marker
                self.save_hazard_marker_position(object_id, hazard_name, position_map)
                self.publish_marker(position_map, object_id, hazard_name)
                
                # Publish status
                self.publish_status(f"Detected {hazard_name} at position: x={position_map.x:.2f}, y={position_map.y:.2f}, z={position_map.z:.2f}")
            else:
                self.get_logger().warning(f"Could not estimate position for {hazard_name}")
                self.publish_status(f"Detected {hazard_name} but could not determine position")

    def estimate_position_from_laser(self, header: Header) -> Optional[Point]:
        """
        Estimates position using laser scan data.
        Takes the closest point in front of the robot from the laser scan.
        """
        if self.last_laser_scan is None:
            self.get_logger().warning("No laser scan data available.")
            return None
            
        try:
            # Get laser scan data
            ranges = self.last_laser_scan.ranges
            angle_min = self.last_laser_scan.angle_min
            angle_increment = self.last_laser_scan.angle_increment
            
            # Get the laser frame directly from the scan message
            laser_frame = self.last_laser_scan.header.frame_id
            self.get_logger().info(f"Using laser frame ID: {laser_frame}")
            
            # Find the closest point in the forward direction
            # We'll use a sector in front of the robot (e.g., -30° to +30°)
            front_sector_width = math.pi / 3  # 60 degrees total (±30°)
            
            # Calculate the indices corresponding to this sector
            center_idx = len(ranges) // 2  # Middle of scan
            sector_size = int(front_sector_width / angle_increment / 2)
            start_idx = center_idx - sector_size
            end_idx = center_idx + sector_size
            
            start_idx = max(0, start_idx)
            end_idx = min(len(ranges) - 1, end_idx)
            
            # Find the closest valid reading in this sector
            closest_range = float('inf')
            closest_angle = 0.0
            
            for i in range(start_idx, end_idx + 1):
                r = ranges[i]
                # Check if the range is valid (not inf, not NaN, and within reasonable bounds)
                if r > 0.1 and r < 10.0 and math.isfinite(r):
                    if r < closest_range:
                        closest_range = r
                        closest_angle = angle_min + i * angle_increment
            
            if closest_range == float('inf'):
                self.get_logger().warning("No valid ranges found in front sector.")
                return None
                
            # Calculate the point in laser frame
            point_laser = Point()
            point_laser.x = closest_range * math.cos(closest_angle)
            point_laser.y = closest_range * math.sin(closest_angle)
            point_laser.z = 0.3  # Approximate height of the marker above ground
            
            self.get_logger().info(f"Closest point in laser frame: x={point_laser.x:.2f}, y={point_laser.y:.2f}, z={point_laser.z:.2f}, range={closest_range:.2f}m, angle={math.degrees(closest_angle):.1f}°")
            
            # Transform to map frame
            point_stamped = PointStamped()
            point_stamped.header.stamp = self.get_clock().now().to_msg()
            point_stamped.header.frame_id = laser_frame
            point_stamped.point = point_laser
            
            try:
                # Wait for the transform to be available
                self.tf_buffer.can_transform(
                    self.map_frame,
                    laser_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
                
                # Transform the point
                point_map_stamped = self.tf_buffer.transform(
                    point_stamped,
                    self.map_frame,
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
                
                return point_map_stamped.point
                
            except Exception as e:
                self.get_logger().error(f"Error transforming point to map frame: {e}")
                return None
                
        except Exception as e:
            self.get_logger().error(f"Error in estimate_position_from_laser: {e}")
            return None
        # """
        # Estimates position using laser scan data.
        # Takes the closest point in front of the robot from the laser scan.
        # """
        # if self.last_laser_scan is None:
        #     self.get_logger().warning("No laser scan data available.")
        #     return None
            
        # try:
        #     # Get laser scan data
        #     ranges = self.last_laser_scan.ranges
        #     angle_min = self.last_laser_scan.angle_min
        #     angle_increment = self.last_laser_scan.angle_increment
            
        #     # Find the closest point in the forward direction
        #     # We'll use a sector in front of the robot (e.g., -30° to +30°)
        #     front_sector_width = math.pi / 3  # 60 degrees total (±30°)
            
        #     # Calculate the indices corresponding to this sector
        #     center_idx = len(ranges) // 2  # Middle of scan
        #     sector_size = int(front_sector_width / angle_increment / 2)
        #     start_idx = center_idx - sector_size
        #     end_idx = center_idx + sector_size
            
        #     start_idx = max(0, start_idx)
        #     end_idx = min(len(ranges) - 1, end_idx)
            
        #     # Find the closest valid reading in this sector
        #     closest_range = float('inf')
        #     closest_angle = 0.0
            
        #     for i in range(start_idx, end_idx + 1):
        #         r = ranges[i]
        #         # Check if the range is valid (not inf, not NaN, and within reasonable bounds)
        #         if r > 0.1 and r < 10.0 and math.isfinite(r):
        #             if r < closest_range:
        #                 closest_range = r
        #                 closest_angle = angle_min + i * angle_increment
            
        #     if closest_range == float('inf'):
        #         self.get_logger().warning("No valid ranges found in front sector.")
        #         return None
                
        #     # Calculate the point in laser frame
        #     point_laser = Point()
        #     point_laser.x = closest_range * math.cos(closest_angle)
        #     point_laser.y = closest_range * math.sin(closest_angle)
        #     point_laser.z = 0.3  # Approximate height of the marker above ground
            
        #     self.get_logger().info(f"Closest point in laser frame: x={point_laser.x:.2f}, y={point_laser.y:.2f}, z={point_laser.z:.2f}, range={closest_range:.2f}m, angle={math.degrees(closest_angle):.1f}°")
            
        #     # Transform to map frame
        #     point_stamped = PointStamped()
        #     point_stamped.header.stamp = self.get_clock().now().to_msg()
        #     point_stamped.header.frame_id = self.laser_frame
        #     point_stamped.point = point_laser
            
        #     try:
        #         # Wait for the transform to be available
        #         self.tf_buffer.can_transform(
        #             self.map_frame,
        #             self.laser_frame,
        #             rclpy.time.Time(),
        #             timeout=rclpy.duration.Duration(seconds=1.0)
        #         )
                
        #         # Transform the point
        #         point_map_stamped = self.tf_buffer.transform(
        #             point_stamped,
        #             self.map_frame,
        #             timeout=rclpy.duration.Duration(seconds=1.0)
        #         )
                
        #         return point_map_stamped.point
                
        #     except Exception as e:
        #         self.get_logger().error(f"Error transforming point to map frame: {e}")
        #         return None
                
        # except Exception as e:
        #     self.get_logger().error(f"Error in estimate_position_from_laser: {e}")
        #     return None

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
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.color.a = 0.8

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

        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
        
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