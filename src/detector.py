"""人脸检测模块 - 基于 InsightFace RetinaFace"""

import numpy as np
import insightface
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
        :return: 人脸列表，每个包含 bbox, embedding, landmarks 等
        """
        faces = self.app.get(image)
        return faces

    def detect_with_info(self, image: np.ndarray) -> list[dict]:
        """
        检测并返回结构化信息
        :return: [{"bbox": [x1,y1,x2,y2], "embedding": np.array, "score": float}, ...]
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
