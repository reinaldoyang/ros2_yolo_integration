import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import json
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
import torch

class YoloSegmentationNode(Node):
    def __init__(self):
        super().__init__("yolo_segmentation_node")

        # 初始化 cv_bridge
        self.bridge = CvBridge()

        # Load only YOLO segmentation model
        seg_model_path = self.resolve_model_path("segmentation.pt")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device : ", device)

        self.seg_model = YOLO(seg_model_path)
        self.seg_model.to(device)

        # 訂閱影像 Topic
        self.image_sub = self.create_subscription(
            CompressedImage, "/camera/image/compressed", self.image_callback, 1
        )

        self.seg_image_pub = self.create_publisher(
            CompressedImage, "/yolo/segmentation/compressed", 10
        )
        self.seg_status_pub = self.create_publisher(
            String, "/yolo/segmentation/status", 10
        )

        # 設定 YOLO 可信度閾值
        self.conf_threshold = 0.5  # 可以修改這個值來調整可信度

    def resolve_model_path(self, filename):
        """Find a YOLO model in the ROS install share or local source tree."""
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
            f"Could not find YOLO model '{filename}'. Checked:\n  - {checked}\n"
            "Rebuild the package after adding models/*.pt, or place the model in "
            f"the installed share directory for {package_name}."
        )

    def image_callback(self, msg):
        """Receive image and run YOLO segmentation only."""
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        try:
            seg_results = self.seg_model(
                cv_image,
                conf=self.conf_threshold,
                verbose=False
            )
        except Exception as e:
            self.get_logger().error(f"Error during YOLO segmentation: {e}")
            return

        seg_image = self.draw_masks(cv_image, seg_results)
        self.publish_segmentation_status(seg_results, cv_image.shape)
        self.publish_seg_image(seg_image)


    def draw_bounding_boxes(self, image, results):
        """在影像上繪製 YOLO 檢測到的 Bounding Box"""
        det_image = image.copy()
        image_center_x = image.shape[1] / 2.0
        bear_candidates = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                class_id = int(box.cls[0])
                class_name = self.det_model.names[class_id]

                # 計算 Bounding Box 正中心點
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # 優先使用無壓縮的深度圖
                depth_value = self.get_depth_at(cx, cy)
                depth_text = f"{depth_value:.2f}m" if depth_value else "N/A"

                # ------ 計算與影像中心的偏移量 ------
                delta_x = cx - image_center_x

                # 根據 class_id 產生隨機但固定的顏色 (B, G, R)
                rng = np.random.RandomState(class_id)
                color = tuple(int(c) for c in rng.randint(0, 256, 3))

                # 繪製框和標籤
                cv2.rectangle(det_image, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {conf:.2f} Depth: {depth_text}"

                cv2.putText(
                    det_image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                if class_name != self.target_class:
                    continue

                bear_candidates.append(
                    {
                        "cx": cx,
                        "cy": cy,
                        "distance": depth_value,
                        "delta_x": delta_x,
                        "confidence": conf,
                        "valid_depth": depth_value > 0 and depth_value != -1.0,
                    }
                )

        selected_bear = self.select_locked_bear_target(bear_candidates)
        if selected_bear is None:
            self.publish_target_info(0, 0.0, 0.0)
        else:
            self.publish_target_info(
                1,
                selected_bear["distance"],
                selected_bear["delta_x"],
            )
        return det_image


    def draw_masks(self, image, results):
        """在影像上繪製 YOLO 檢測到的 Mask"""
        height, width = image.shape[:2]
        mask_image = image.copy()  # 從原始影像複製一份來繪製 Mask

        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes
                for i, mask in enumerate(masks):
                    # Create a boolean mask and assign color
                    mask_resized = cv2.resize(mask, (width, height))
                    mask_bool = mask_resized > 0.5
                    
                    # 根據 class_id 產生隨機但固定的顏色 (B, G, R)
                    class_id = int(boxes.cls[i])
                    rng = np.random.RandomState(class_id)
                    color = tuple(int(c) for c in rng.randint(0, 256, 3))
                    
                    # Blend the mask for better visibility
                    mask_colored = np.zeros_like(mask_image)
                    mask_colored[mask_bool] = color
                    mask_image = cv2.addWeighted(mask_image, 1, mask_colored, 0.5, 0)

        return mask_image

    def publish_segmentation_status(self, results, image_shape):
        height, width = image_shape[:2]
        bridge_mask = np.zeros((height, width), dtype=bool)
        road_mask = np.zeros((height, width), dtype=bool)
        wall_mask = np.zeros((height, width), dtype=bool)

        for result in results:
            if result.masks is None or result.boxes is None:
                continue

            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes
            for i, mask in enumerate(masks):
                class_id = int(boxes.cls[i])
                class_name = self.get_class_name(class_id)
                mask_resized = cv2.resize(mask, (width, height)) > 0.5

                if "wall" in class_name:
                    wall_mask |= mask_resized
                elif "bridge" in class_name or "whole_bridge" in class_name:
                    bridge_mask |= mask_resized
                elif "road" in class_name:
                    road_mask |= mask_resized

        bridge_pixels_v, bridge_pixels_u = np.where(bridge_mask)
        bridge_area_ratio = float(np.count_nonzero(bridge_mask)) / float(width * height)
        bridge_center_x = (
            float(np.mean(bridge_pixels_u)) if bridge_pixels_u.size > 0 else -1.0
        )
        lower_start = height // 2
        lower_pixel_count = float(width * (height - lower_start))
        road_lower_ratio = (
            float(np.count_nonzero(road_mask[lower_start:, :])) / lower_pixel_count
        )
        bridge_lower_ratio = (
            float(np.count_nonzero(bridge_mask[lower_start:, :])) / lower_pixel_count
        )

        status = {
            "bridge_detected": bridge_pixels_u.size > 0,
            "bridge_center_x": bridge_center_x,
            "bridge_area_ratio": bridge_area_ratio,
            "bridge_lower_ratio": bridge_lower_ratio,
            "road_area_ratio": float(np.count_nonzero(road_mask)) / float(width * height),
            "road_lower_ratio": road_lower_ratio,
            "wall_area_ratio": float(np.count_nonzero(wall_mask)) / float(width * height),
            "image_width": int(width),
            "image_height": int(height),
        }

        msg = String()
        msg.data = json.dumps(status)
        self.seg_status_pub.publish(msg)

    def get_class_name(self, class_id):
        names = self.seg_model.names
        if isinstance(names, dict):
            return str(names.get(class_id, class_id)).lower()
        if 0 <= class_id < len(names):
            return str(names[class_id]).lower()
        return str(class_id)

    def publish_seg_image(self, image):
        """將 Segmentation 影像轉換並發佈到 ROS"""
        try:
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(image)
            self.seg_image_pub.publish(compressed_msg)
        except Exception as e:
            self.get_logger().error(f"Could not publish segmentation image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = YoloSegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
