"""人脸检测模块 - 基于 InsightFace RetinaFace"""

from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceDetector:
    def __init__(self, det_size=(640, 640), gpu_id=0):
        """
        初始化人脸检测器
        :param det_size: 检测输入尺寸
        :param gpu_id: GPU 设备 ID
        """
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=gpu_id, det_size=det_size)

    def detect(self, image: np.ndarray) -> list:
        """
        检测图片中的所有人脸
        :param image: BGR 格式的图片 (OpenCV)
        :return: 人脸列表
        """
        faces = self.app.get(image)
        return faces

    def detect_with_info(self, image: np.ndarray) -> list[dict]:
        """
        检测并返回结构化信息
        """
        faces = self.detect(image)
        results = []
        for face in faces:
            results.append({
                "bbox": face.bbox.astype(int).tolist(),
                "embedding": face.normed_embedding,
                "score": float(face.det_score),
                "age": int(face.age) if hasattr(face, "age") else None,
                "gender": "M" if face.gender == 1 else "F" if hasattr(face, "gender") else None,
            })
        return results

    def extract_faces(
        self,
        image: np.ndarray,
        margin: float = 0.3,
        min_size: int = 40,
    ) -> list[dict]:
        """
        检测并裁剪人脸区域
        :param image: BGR 图片
        :param margin: 人脸周围留白比例
        :param min_size: 最小人脸尺寸（像素），过小的跳过
        :return: [{"crop": np.ndarray, "bbox": list, "embedding": np.array, ...}, ...]
        """
        h, w = image.shape[:2]
        faces = self.detect_with_info(image)
        results = []

        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            fw, fh = x2 - x1, y2 - y1

            if fw < min_size or fh < min_size:
                continue

            # 加 margin
            mx = int(fw * margin)
            my = int(fh * margin)
            cx1 = max(0, x1 - mx)
            cy1 = max(0, y1 - my)
            cx2 = min(w, x2 + mx)
            cy2 = min(h, y2 + my)

            crop = image[cy1:cy2, cx1:cx2].copy()
            face["crop"] = crop
            face["crop_bbox"] = [cx1, cy1, cx2, cy2]
            results.append(face)

        return results

    def save_extracted_faces(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        margin: float = 0.3,
        min_size: int = 40,
    ) -> list[dict]:
        """
        从图片提取人脸并保存到输出目录
        :return: 保存的人脸信息列表
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        faces = self.extract_faces(image, margin=margin, min_size=min_size)
        stem = Path(image_path).stem
        saved = []

        for i, face in enumerate(faces):
            suffix = f"_{i}" if len(faces) > 1 else ""
            out_path = output_dir / f"{stem}{suffix}.jpg"
            cv2.imwrite(str(out_path), face["crop"])
            face["saved_path"] = str(out_path)
            del face["crop"]  # 释放内存
            saved.append(face)

        return saved
