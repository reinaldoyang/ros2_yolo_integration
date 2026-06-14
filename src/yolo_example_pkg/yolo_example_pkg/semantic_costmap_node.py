import base64
import binascii
import copy
import math
import os
from pathlib import Path

# Unity sends RFloat depth as EXR. OpenCV requires this flag before import.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import rclpy
import torch
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from tf2_ros import Buffer, TransformException, TransformListener
from ultralytics import YOLO


class SemanticCostmapNode(Node):
    def __init__(self):
        super().__init__("semantic_costmap_node")
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.declare_parameter("image_topic", "/camera/image/compressed")
        self.declare_parameter("depth_topic", "/camera/depth/compressed")
        self.declare_parameter("depth_is_compressed", True)
        self.declare_parameter("camera_info_topic", "/camera_info")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("grid_width", 200)
        self.declare_parameter("grid_height", 200)
        self.declare_parameter("resolution", 0.05)
        self.declare_parameter("pixel_sample_step", 8)
        self.declare_parameter("obstacle_inflation_radius", 0.15)
        self.declare_parameter("target_frame", "map")
        self.declare_parameter("camera_frame", "")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("max_depth", 4.0)
        self.declare_parameter("projection_mode", "depth")
        self.declare_parameter("ground_plane_z", 0.0)
        self.declare_parameter("rectify_obstacle_masks", True)
        self.declare_parameter("obstacle_depth_gate_margin", 0.40)
        self.declare_parameter("obstacle_depth_gate_percentile", 20.0)
        self.declare_parameter("enable_obstacle_memory", True)
        self.declare_parameter("obstacle_memory_decay_sec", 0.0)
        self.declare_parameter("obstacle_confirm_sec", 3.0)
        self.declare_parameter("obstacle_confirm_gap_sec", 1.0)
        self.declare_parameter("obstacle_clear_confirm_sec", 3.0)
        self.declare_parameter("obstacle_clear_gap_sec", 1.0)
        self.declare_parameter("semantic_memory_resolution", 0.05)

        self.image_topic = self.get_parameter("image_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.depth_is_compressed = bool(
            self.get_parameter("depth_is_compressed").value
        )
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.map_topic = self.get_parameter("map_topic").value
        self.grid_width = int(self.get_parameter("grid_width").value)
        self.grid_height = int(self.get_parameter("grid_height").value)
        self.resolution = float(self.get_parameter("resolution").value)
        self.pixel_sample_step = max(1, int(self.get_parameter("pixel_sample_step").value))
        self.obstacle_inflation_radius = float(
            self.get_parameter("obstacle_inflation_radius").value
        )
        self.target_frame = self.get_parameter("target_frame").value
        self.camera_frame_param = self.get_parameter("camera_frame").value
        self.conf_threshold = float(self.get_parameter("confidence_threshold").value)
        self.max_depth = float(self.get_parameter("max_depth").value)
        self.projection_mode = str(self.get_parameter("projection_mode").value)
        self.ground_plane_z = float(self.get_parameter("ground_plane_z").value)
        self.rectify_obstacle_masks = bool(
            self.get_parameter("rectify_obstacle_masks").value
        )
        self.obstacle_depth_gate_margin = float(
            self.get_parameter("obstacle_depth_gate_margin").value
        )
        self.obstacle_depth_gate_percentile = float(
            self.get_parameter("obstacle_depth_gate_percentile").value
        )
        self.enable_obstacle_memory = bool(
            self.get_parameter("enable_obstacle_memory").value
        )
        self.obstacle_memory_decay_sec = float(
            self.get_parameter("obstacle_memory_decay_sec").value
        )
        self.obstacle_confirm_sec = float(
            self.get_parameter("obstacle_confirm_sec").value
        )
        self.obstacle_confirm_gap_sec = float(
            self.get_parameter("obstacle_confirm_gap_sec").value
        )
        self.obstacle_clear_confirm_sec = float(
            self.get_parameter("obstacle_clear_confirm_sec").value
        )
        self.obstacle_clear_gap_sec = float(
            self.get_parameter("obstacle_clear_gap_sec").value
        )
        self.semantic_memory_resolution = float(
            self.get_parameter("semantic_memory_resolution").value
        )

        model_path = self.resolve_model_path("segmentation.pt")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {device}")
        self.seg_model = YOLO(model_path)
        self.seg_model.to(device)

        self.latest_depth = None
        self.latest_depth_frame = ""
        self.latest_camera_info = None
        self.latest_map_info = None
        self.latest_map_frame = self.target_frame
        self.last_depth_debug_log_time = 0.0
        self.obstacle_memory = {}
        self.pending_obstacle_memory = {}
        self.pending_clear_memory = {}
        self.last_observed_obstacle_cell_count = 0
        self.last_observed_clear_cell_count = 0
        self.last_obstacle_memory_log_time = 0.0
        self.initial_empty_obstacle_grid_published = False
        self.startup_obstacle_reset_done = False

        self.image_sub = self.create_subscription(
            CompressedImage, self.image_topic, self.image_callback, 1
        )
        depth_msg_type = CompressedImage if self.depth_is_compressed else Image
        self.depth_sub = self.create_subscription(
            depth_msg_type, self.depth_topic, self.depth_callback, 1
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, 1
        )
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, self.map_topic, self.map_callback, map_qos
        )
        self.grid_pub = self.create_publisher(OccupancyGrid, "/semantic_bridge_grid", 1)
        self.obstacle_grid_pub = self.create_publisher(
            OccupancyGrid, "/semantic_obstacle_grid", map_qos
        )

        self.get_logger().info(
            "Semantic costmap node ready: "
            f"image={self.image_topic}, depth={self.depth_topic}, "
            f"depth_is_compressed={self.depth_is_compressed}, "
            f"camera_info={self.camera_info_topic}, map={self.map_topic}, "
            "outputs=/semantic_bridge_grid,/semantic_obstacle_grid, "
            f"enable_obstacle_memory={self.enable_obstacle_memory}, "
            f"obstacle_inflation_radius={self.obstacle_inflation_radius:.2f}, "
            f"obstacle_memory_decay_sec={self.obstacle_memory_decay_sec:.1f}, "
            f"obstacle_confirm_sec={self.obstacle_confirm_sec:.1f}, "
            f"obstacle_clear_confirm_sec={self.obstacle_clear_confirm_sec:.1f}, "
            f"projection_mode={self.projection_mode}, "
            f"ground_plane_z={self.ground_plane_z:.2f}, "
            f"rectify_obstacle_masks={self.rectify_obstacle_masks}, "
            f"obstacle_depth_gate_margin={self.obstacle_depth_gate_margin:.2f}, "
            f"obstacle_depth_gate_percentile={self.obstacle_depth_gate_percentile:.1f}, "
            f"semantic_memory_resolution={self.semantic_memory_resolution:.3f}"
        )

    def resolve_model_path(self, filename):
        package_name = "yolo_example_pkg"
        candidate_dirs = [
            Path(get_package_share_directory(package_name)) / "models",
            Path(__file__).resolve().parent.parent / "models",
            Path.cwd() / package_name / "models",
        ]

        for parent in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
            candidate_dirs.extend(
                [
                    parent / package_name / "models",
                    parent / "src" / package_name / "models",
                ]
            )

        checked_paths = []
        for model_dir in candidate_dirs:
            model_path = model_dir / filename
            if model_path in checked_paths:
                continue
            checked_paths.append(model_path)
            if model_path.exists():
                self.get_logger().info(f"Using YOLO model: {model_path}")
                return str(model_path)

        checked = "\n  - ".join(str(path) for path in checked_paths)
        raise FileNotFoundError(
            f"Could not find YOLO model '{filename}'. Checked:\n  - {checked}"
        )

    def depth_callback(self, msg):
        if self.depth_is_compressed:
            self.depth_callback_compressed(msg)
            return

        self.depth_callback_raw(msg)

    def depth_callback_raw(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
            self.latest_depth_frame = msg.header.frame_id
            self.log_depth_debug("raw Image", msg.encoding, self.latest_depth)
        except Exception as exc:
            self.get_logger().warn(f"Could not convert depth image: {exc}")

    def depth_callback_compressed(self, msg):
        try:
            raw_data = bytes(msg.data)
            encoded_data, payload_type = self.decode_optional_base64(raw_data)
            np_arr = np.frombuffer(encoded_data, np.uint8)

            # Reuse the existing project approach of decoding compressed depth
            # with OpenCV, but add support for Unity's RFloat EXR payload.
            depth = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            if depth is None:
                self.get_logger().warn(
                    "Could not decode compressed depth image: "
                    f"format='{msg.format}', payload={payload_type}, "
                    f"data_len={len(raw_data)}, decoded_len={len(encoded_data)}, "
                    f"magic={encoded_data[:8].hex()}"
                )
                return

            if len(depth.shape) == 3:
                depth = depth[:, :, 0]

            self.latest_depth = self.normalize_depth_image(depth)
            self.latest_depth_frame = msg.header.frame_id
            self.log_depth_debug(
                payload_type,
                msg.format,
                self.latest_depth,
                data_len=len(raw_data),
                decoded_len=len(encoded_data),
            )
        except cv2.error as exc:
            self.get_logger().warn(
                "OpenCV failed to decode compressed depth. "
                "If this is EXR, make sure OpenCV has OpenEXR enabled. "
                f"format='{msg.format}', data_len={len(msg.data)}, error={exc}"
            )
        except Exception as exc:
            self.get_logger().warn(f"Could not convert compressed depth image: {exc}")

    def decode_optional_base64(self, data):
        if data.startswith(self.exr_magic()) or data.startswith(self.png_magic()):
            return data, "raw compressed bytes"

        stripped = data.strip()
        if not stripped:
            return data, "empty compressed bytes"

        try:
            stripped.decode("ascii")
            padded = stripped + (b"=" * ((4 - len(stripped) % 4) % 4))
            decoded = base64.b64decode(padded, validate=True)
            if decoded.startswith(self.exr_magic()) or decoded.startswith(self.png_magic()):
                return decoded, "base64 compressed bytes"
            return decoded, "base64 decoded bytes"
        except (UnicodeDecodeError, binascii.Error, ValueError):
            return data, "raw compressed bytes"

    def normalize_depth_image(self, depth):
        if not isinstance(depth, np.ndarray):
            return None

        if np.issubdtype(depth.dtype, np.integer):
            return depth.astype(np.float32) / 1000.0

        return depth.astype(np.float32)

    def log_depth_debug(self, payload_type, msg_format, depth, data_len=None, decoded_len=None):
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_depth_debug_log_time < 2.0:
            return
        self.last_depth_debug_log_time = now

        if depth is None:
            self.get_logger().warn(
                f"Depth decode failed: format='{msg_format}', payload={payload_type}"
            )
            return

        valid = depth[np.isfinite(depth) & (depth > 0.0)]
        if valid.size > 0:
            min_depth = float(np.min(valid))
            max_depth = float(np.max(valid))
            depth_range = f"valid_min={min_depth:.3f}m, valid_max={max_depth:.3f}m"
        else:
            depth_range = "no valid positive depth pixels"

        sizes = ""
        if data_len is not None:
            sizes += f", data_len={data_len}"
        if decoded_len is not None:
            sizes += f", decoded_len={decoded_len}"

        self.get_logger().info(
            "Decoded depth: "
            f"format='{msg_format}', payload={payload_type}{sizes}, "
            f"shape={depth.shape}, dtype={depth.dtype}, {depth_range}"
        )

    def exr_magic(self):
        return bytes([0x76, 0x2F, 0x31, 0x01])

    def png_magic(self):
        return bytes([137, 80, 78, 71, 13, 10, 26, 10])

    def camera_info_callback(self, msg):
        self.latest_camera_info = msg

    def map_callback(self, msg):
        new_map_info = copy.deepcopy(msg.info)
        new_map_frame = msg.header.frame_id or self.target_frame
        self.latest_map_info = new_map_info
        self.latest_map_frame = new_map_frame

        if not self.startup_obstacle_reset_done:
            self.clear_obstacle_memory("node started")
            self.publish_semantic_obstacle_grid()
            self.initial_empty_obstacle_grid_published = True
            self.startup_obstacle_reset_done = True
            return

    def clear_obstacle_memory(self, reason):
        self.obstacle_memory.clear()
        self.pending_obstacle_memory.clear()
        self.pending_clear_memory.clear()
        self.last_observed_obstacle_cell_count = 0
        self.last_observed_clear_cell_count = 0
        self.initial_empty_obstacle_grid_published = False
        self.get_logger().info(f"Semantic obstacle memory cleared: {reason}")

    def image_callback(self, msg):
        if self.latest_depth is None or self.latest_camera_info is None:
            self.get_logger().debug("Waiting for depth image and camera_info")
            return

        try:
            image = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
        except Exception as exc:
            self.get_logger().warn(f"Could not convert RGB image: {exc}")
            return

        camera_frame = self.get_camera_frame(msg)
        transform = self.lookup_transform(camera_frame)
        if transform is None:
            return

        try:
            # This debug node runs its own segmentation inference because the
            # existing segmentation topic is a blended visualization image, not
            # a clean class mask that can be projected into the map.
            results = self.seg_model(
                image, conf=self.conf_threshold, verbose=False
            )
        except Exception as exc:
            self.get_logger().warn(f"YOLO segmentation failed: {exc}")
            return

        robot_pose = self.lookup_robot_pose()
        if robot_pose is None:
            return

        grid, origin_x, origin_y = self.create_empty_grid(robot_pose)
        self.rasterize_segmentation(
            results, image.shape, transform, grid, origin_x, origin_y
        )
        self.publish_grid(grid, origin_x, origin_y)
        self.publish_semantic_obstacle_grid()

    def get_camera_frame(self, image_msg):
        if self.camera_frame_param:
            return self.camera_frame_param
        if image_msg.header.frame_id:
            return image_msg.header.frame_id
        if self.latest_depth_frame:
            return self.latest_depth_frame
        return self.latest_camera_info.header.frame_id

    def lookup_transform(self, camera_frame):
        try:
            return self.tf_buffer.lookup_transform(
                self.target_frame,
                camera_frame,
                Time(),
                timeout=Duration(seconds=0.1),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"TF lookup failed {self.target_frame} <- {camera_frame}: {exc}",
                throttle_duration_sec=2.0,
            )
            return None

    def lookup_robot_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                "base_footprint",
                Time(),
                timeout=Duration(seconds=0.1),
            )
            return (
                transform.transform.translation.x,
                transform.transform.translation.y,
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"TF lookup failed {self.target_frame} <- base_footprint: {exc}",
                throttle_duration_sec=2.0,
            )
            return None

    def create_empty_grid(self, robot_pose):
        grid = np.full((self.grid_height, self.grid_width), -1, dtype=np.int8)
        origin_x = robot_pose[0] - (self.grid_width * self.resolution / 2.0)
        origin_y = robot_pose[1] - (self.grid_height * self.resolution / 2.0)
        return grid, origin_x, origin_y

    def rasterize_segmentation(
        self, results, image_shape, transform, grid, origin_x, origin_y
    ):
        image_height, image_width = image_shape[:2]
        fx = float(self.latest_camera_info.k[0])
        fy = float(self.latest_camera_info.k[4])
        cx = float(self.latest_camera_info.k[2])
        cy = float(self.latest_camera_info.k[5])
        if fx <= 0.0 or fy <= 0.0:
            self.get_logger().warn("Invalid camera_info intrinsics")
            return

        rotation = self.quaternion_to_matrix(transform.transform.rotation)
        translation = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ],
            dtype=np.float64,
        )
        observed_obstacle_keys = set()
        observed_clear_keys = set()

        for result in results:
            if result.masks is None or result.boxes is None:
                continue

            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes
            for i, mask in enumerate(masks):
                class_id = int(boxes.cls[i])
                class_name = self.get_class_name(class_id)
                semantic_value = self.semantic_grid_value(class_name)
                if semantic_value is None:
                    continue

                mask_resized = cv2.resize(mask, (image_width, image_height)) > 0.5
                pixels_v, pixels_u = np.where(mask_resized)
                obstacle_cells = []
                obstacle_depth_limit = None
                if semantic_value == 100:
                    obstacle_depth_limit = self.get_obstacle_depth_limit(
                        pixels_u,
                        pixels_v,
                        image_width,
                        image_height,
                    )

                for v, u in zip(
                    pixels_v[:: self.pixel_sample_step],
                    pixels_u[:: self.pixel_sample_step],
                ):
                    z = self.get_depth_meters(u, v, image_width, image_height)
                    if z is None:
                        continue

                    if (
                        semantic_value == 100
                        and obstacle_depth_limit is not None
                        and z > obstacle_depth_limit
                    ):
                        continue

                    point_map = self.project_pixel_to_map(
                        u,
                        v,
                        image_width,
                        image_height,
                        fx,
                        fy,
                        cx,
                        cy,
                        rotation,
                        translation,
                        z,
                    )
                    if point_map is None:
                        continue

                    cell = self.map_point_to_cell(
                        point_map[0], point_map[1], origin_x, origin_y
                    )
                    if cell is None:
                        continue

                    if semantic_value == 100:
                        obstacle_cells.append(cell)
                    elif grid[cell[1], cell[0]] != 100:
                        grid[cell[1], cell[0]] = 0
                        observed_clear_keys.update(
                            self.get_memory_keys_for_point(
                                point_map[0], point_map[1], radius=0.0
                            )
                        )

                if semantic_value == 100 and obstacle_cells:
                    if self.rectify_obstacle_masks and len(obstacle_cells) >= 3:
                        marked_cells = self.mark_rectified_obstacle(
                            grid, obstacle_cells
                        )
                    else:
                        marked_cells = []
                        for col, row in obstacle_cells:
                            self.mark_obstacle(grid, col, row)
                            marked_cells.append((col, row))

                    for col, row in marked_cells:
                        x = origin_x + (col + 0.5) * self.resolution
                        y = origin_y + (row + 0.5) * self.resolution
                        observed_obstacle_keys.update(
                            self.get_memory_keys_for_point(x, y, radius=0.0)
                        )

        if observed_obstacle_keys:
            self.get_logger().debug(
                f"Observed semantic obstacle keys this frame: {len(observed_obstacle_keys)}"
            )
        if observed_clear_keys:
            self.get_logger().debug(
                f"Observed semantic clear keys this frame: {len(observed_clear_keys)}"
            )
        self.last_observed_obstacle_cell_count = len(observed_obstacle_keys)
        self.last_observed_clear_cell_count = len(observed_clear_keys)
        self.observe_obstacle_cells(observed_obstacle_keys)
        self.observe_clear_cells(observed_clear_keys)

    def get_class_name(self, class_id):
        names = self.seg_model.names
        if isinstance(names, dict):
            return str(names.get(class_id, class_id)).lower()
        if 0 <= class_id < len(names):
            return str(names[class_id]).lower()
        return str(class_id)

    def semantic_grid_value(self, class_name):
        if "bear" in class_name:
            return None
        if "bridge" in class_name or "whole_bridge" in class_name or "wall" in class_name:
            return 100
        if "road" in class_name:
            return 0
        return None

    def get_depth_meters(self, u, v, image_width, image_height):
        depth = self.latest_depth
        if depth is None:
            return None

        depth_height, depth_width = depth.shape[:2]
        depth_u = int(round(u * (depth_width - 1) / max(1, image_width - 1)))
        depth_v = int(round(v * (depth_height - 1) / max(1, image_height - 1)))
        if depth_u < 0 or depth_u >= depth_width or depth_v < 0 or depth_v >= depth_height:
            return None

        value = depth[depth_v, depth_u]
        if isinstance(value, np.ndarray):
            value = value[0]

        if np.issubdtype(depth.dtype, np.integer):
            z = float(value) / 1000.0
        else:
            z = float(value)

        if not math.isfinite(z) or z <= 0.0 or z > self.max_depth:
            return None
        return z

    def get_obstacle_depth_limit(self, pixels_u, pixels_v, image_width, image_height):
        if self.obstacle_depth_gate_margin < 0.0:
            return None

        valid_depths = []
        for v, u in zip(
            pixels_v[:: self.pixel_sample_step],
            pixels_u[:: self.pixel_sample_step],
        ):
            z = self.get_depth_meters(u, v, image_width, image_height)
            if z is not None:
                valid_depths.append(z)

        if not valid_depths:
            return None

        percentile = min(100.0, max(0.0, self.obstacle_depth_gate_percentile))
        near_depth = float(np.percentile(np.array(valid_depths), percentile))
        return near_depth + self.obstacle_depth_gate_margin

    def project_pixel_to_map(
        self,
        u,
        v,
        image_width,
        image_height,
        fx,
        fy,
        cx,
        cy,
        rotation,
        translation,
        z,
    ):
        x_norm = (u - cx) / fx
        y_norm = (v - cy) / fy

        if self.projection_mode == "ground_plane":
            # Instead of using each obstacle pixel's raw depth, intersect the
            # camera ray with the map ground plane. This makes bridge/wall
            # masks behave like a top-down floor projection, which is more
            # stable for semantic costmap debugging.
            ray_map = rotation @ np.array([x_norm, y_norm, 1.0], dtype=np.float64)
            if abs(ray_map[2]) < 1e-6:
                return None

            scale = (self.ground_plane_z - translation[2]) / ray_map[2]
            if not math.isfinite(scale) or scale <= 0.0 or scale > self.max_depth:
                return None

            return translation + ray_map * scale

        # Pinhole projection into the camera optical frame:
        # X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy.
        point_camera = np.array([x_norm * z, y_norm * z, z], dtype=np.float64)
        return rotation @ point_camera + translation

    def map_point_to_cell(self, x, y, origin_x, origin_y):
        col = int((x - origin_x) / self.resolution)
        row = int((y - origin_y) / self.resolution)
        if col < 0 or col >= self.grid_width or row < 0 or row >= self.grid_height:
            return None
        return col, row

    def mark_obstacle(self, grid, col, row):
        inflation_cells = int(math.ceil(self.obstacle_inflation_radius / self.resolution))
        for dy in range(-inflation_cells, inflation_cells + 1):
            for dx in range(-inflation_cells, inflation_cells + 1):
                if dx * dx + dy * dy > inflation_cells * inflation_cells:
                    continue
                c = col + dx
                r = row + dy
                if 0 <= c < self.grid_width and 0 <= r < self.grid_height:
                    grid[r, c] = 100

    def mark_rectified_obstacle(self, grid, obstacle_cells):
        # The raw projected pixels can be ragged because segmentation and depth
        # are noisy. Fit a minimum-area rectangle to the projected obstacle
        # footprint so bridge/wall regions appear as a stable top-down shape.
        points = np.array(obstacle_cells, dtype=np.float32)
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect).astype(np.int32)

        rect_mask = np.zeros(grid.shape, dtype=np.uint8)
        cv2.fillPoly(rect_mask, [box], 1)
        rows, cols = np.where(rect_mask > 0)

        marked_cells = []
        for row, col in zip(rows, cols):
            if 0 <= col < self.grid_width and 0 <= row < self.grid_height:
                self.mark_obstacle(grid, col, row)
                marked_cells.append((col, row))
        return marked_cells

    def get_obstacle_keys_for_point(self, x, y):
        return self.get_memory_keys_for_point(
            x, y, radius=self.obstacle_inflation_radius
        )

    def get_memory_keys_for_point(self, x, y, radius):
        if not self.enable_obstacle_memory or self.semantic_memory_resolution <= 0.0:
            return []

        center_x = int(round(x / self.semantic_memory_resolution))
        center_y = int(round(y / self.semantic_memory_resolution))
        inflation_keys = int(math.ceil(radius / self.semantic_memory_resolution))
        keys = []

        for dy in range(-inflation_keys, inflation_keys + 1):
            for dx in range(-inflation_keys, inflation_keys + 1):
                if dx * dx + dy * dy > inflation_keys * inflation_keys:
                    continue
                keys.append((center_x + dx, center_y + dy))
        return keys

    def observe_obstacle_cells(self, obstacle_cells):
        if not obstacle_cells:
            return

        stamp = self.get_clock().now().nanoseconds / 1e9
        for cell in obstacle_cells:
            self.update_pending_obstacle(cell, stamp)

    def update_pending_obstacle(self, cell, stamp):
        if cell in self.obstacle_memory:
            self.obstacle_memory[cell] = stamp
            return

        pending = self.pending_obstacle_memory.get(cell)
        if pending is None or stamp - pending["last_seen"] > self.obstacle_confirm_gap_sec:
            self.pending_obstacle_memory[cell] = {
                "first_seen": stamp,
                "last_seen": stamp,
            }
            return

        pending["last_seen"] = stamp
        if pending["last_seen"] - pending["first_seen"] >= self.obstacle_confirm_sec:
            self.obstacle_memory[cell] = stamp
            del self.pending_obstacle_memory[cell]

    def observe_clear_cells(self, clear_cells):
        if not clear_cells:
            return

        stamp = self.get_clock().now().nanoseconds / 1e9
        for cell in clear_cells:
            self.update_pending_clear(cell, stamp)

    def update_pending_clear(self, cell, stamp):
        if cell not in self.obstacle_memory and cell not in self.pending_obstacle_memory:
            self.pending_clear_memory.pop(cell, None)
            return

        pending = self.pending_clear_memory.get(cell)
        if pending is None or stamp - pending["last_seen"] > self.obstacle_clear_gap_sec:
            self.pending_clear_memory[cell] = {
                "first_seen": stamp,
                "last_seen": stamp,
            }
            return

        pending["last_seen"] = stamp
        if pending["last_seen"] - pending["first_seen"] >= self.obstacle_clear_confirm_sec:
            self.obstacle_memory.pop(cell, None)
            self.pending_obstacle_memory.pop(cell, None)
            del self.pending_clear_memory[cell]

    def map_point_to_static_map_cell(self, x, y):
        if self.latest_map_info is None:
            return None

        origin = self.latest_map_info.origin
        resolution = float(self.latest_map_info.resolution)
        if resolution <= 0.0:
            return None

        yaw = self.quaternion_to_yaw(origin.orientation)
        dx = x - origin.position.x
        dy = y - origin.position.y

        # Convert map-frame coordinates into the OccupancyGrid's fixed index
        # frame. Most maps have yaw=0, but this keeps the grid aligned if they do not.
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        col = int(math.floor(local_x / resolution))
        row = int(math.floor(local_y / resolution))

        if col < 0 or col >= self.latest_map_info.width:
            return None
        if row < 0 or row >= self.latest_map_info.height:
            return None
        return col, row

    def memory_key_to_point(self, key):
        return (
            key[0] * self.semantic_memory_resolution,
            key[1] * self.semantic_memory_resolution,
        )

    def prune_obstacle_memory(self):
        now = self.get_clock().now().nanoseconds / 1e9
        stale_pending_cells = [
            cell
            for cell, pending in self.pending_obstacle_memory.items()
            if now - pending["last_seen"] > self.obstacle_confirm_gap_sec
        ]
        for cell in stale_pending_cells:
            del self.pending_obstacle_memory[cell]

        stale_clear_cells = [
            cell
            for cell, pending in self.pending_clear_memory.items()
            if now - pending["last_seen"] > self.obstacle_clear_gap_sec
        ]
        for cell in stale_clear_cells:
            del self.pending_clear_memory[cell]

        if self.obstacle_memory_decay_sec <= 0.0:
            return

        cutoff = now - self.obstacle_memory_decay_sec
        stale_cells = [
            cell
            for cell, stamp in self.obstacle_memory.items()
            if stamp < cutoff
        ]
        for cell in stale_cells:
            del self.obstacle_memory[cell]

    def publish_semantic_obstacle_grid(self):
        if self.latest_map_info is None:
            self.get_logger().warn(
                "Waiting for /map before publishing /semantic_obstacle_grid",
                throttle_duration_sec=2.0,
            )
            return

        if self.enable_obstacle_memory:
            self.prune_obstacle_memory()

        width = int(self.latest_map_info.width)
        height = int(self.latest_map_info.height)
        # Use 0 as the background so Foxglove/Nav2-style viewers do not render
        # unknown cells as high cost. Only remembered bridge/wall cells are 100.
        obstacle_grid = np.zeros((height, width), dtype=np.int8)
        if self.enable_obstacle_memory:
            for key in self.obstacle_memory:
                point = self.memory_key_to_point(key)
                cell = self.map_point_to_static_map_cell(point[0], point[1])
                if cell is None:
                    continue
                col, row = cell
                if 0 <= col < width and 0 <= row < height:
                    obstacle_grid[row, col] = 100

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.latest_map_frame or self.target_frame
        msg.info = copy.deepcopy(self.latest_map_info)
        msg.data = obstacle_grid.reshape(-1).astype(np.int8).tolist()
        self.obstacle_grid_pub.publish(msg)

        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_obstacle_memory_log_time >= 2.0:
            self.last_obstacle_memory_log_time = now
            published_count = int(np.count_nonzero(obstacle_grid == 100))
            self.get_logger().info(
                "Semantic obstacle memory: "
                f"observed_obstacle_cells={self.last_observed_obstacle_cell_count}, "
                f"observed_clear_cells={self.last_observed_clear_cell_count}, "
                f"pending_obstacle_cells={len(self.pending_obstacle_memory)}, "
                f"pending_clear_cells={len(self.pending_clear_memory)}, "
                f"remembered_obstacle_cells={len(self.obstacle_memory)}, "
                f"published_obstacle_cells={published_count}, "
                f"memory_resolution={self.semantic_memory_resolution:.3f}, "
                f"map_size={width}x{height}, "
                f"map_resolution={self.latest_map_info.resolution:.3f}"
            )

    def publish_grid(self, grid, origin_x, origin_y):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.target_frame
        msg.info.resolution = self.resolution
        msg.info.width = self.grid_width
        msg.info.height = self.grid_height
        msg.info.origin.position.x = origin_x
        msg.info.origin.position.y = origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.reshape(-1).astype(np.int8).tolist()
        self.grid_pub.publish(msg)

    def quaternion_to_yaw(self, quat: Quaternion):
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def quaternion_to_matrix(self, quat: Quaternion):
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm == 0.0:
            return np.eye(3)
        x /= norm
        y /= norm
        z /= norm
        w /= norm
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )


def main(args=None):
    rclpy.init(args=args)
    node = SemanticCostmapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
