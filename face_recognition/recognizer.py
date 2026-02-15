"""人脸识别模块 - 特征比对与身份识别"""

import os
import pickle
import numpy as np
from pathlib import Path

# 默认特征库路径
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "models" / "face_db.pkl"


class FaceRecognizer:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH, threshold: float = 0.4):
        """
        初始化人脸识别器
        :param db_path: 特征数据库路径
        :param threshold: 识别阈值（余弦距离），越小越严格
        """
        self.db_path = Path(db_path)
        self.threshold = threshold
        self.face_db: dict[str, list[np.ndarray]] = {}  # {name: [embeddings]}
        self._load_db()

    def _load_db(self):
        """加载特征数据库"""
        if self.db_path.exists():
            with open(self.db_path, "rb") as f:
                self.face_db = pickle.load(f)
            print(f"[识别器] 加载特征库: {', '.join(f'{k}({len(v)}张)' for k, v in self.face_db.items())}")
        else:
            print("[识别器] 特征库为空，请先注册人脸")

    def save_db(self):
        """保存特征数据库"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(self.face_db, f)
        print(f"[识别器] 特征库已保存: {self.db_path}")

    def register(self, name: str, embeddings: list[np.ndarray]):
        """
        注册新人脸
        :param name: 人名
        :param embeddings: 该人的多个特征向量
        """
        if name not in self.face_db:
            self.face_db[name] = []
        self.face_db[name].extend(embeddings)
        self.save_db()
        print(f"[识别器] 注册 '{name}': 新增 {len(embeddings)} 张，总计 {len(self.face_db[name])} 张")

    def remove(self, name: str) -> bool:
        """删除已注册的人脸"""
        if name in self.face_db:
            del self.face_db[name]
            self.save_db()
            print(f"[识别器] 已删除 '{name}'")
            return True
        return False

    def list_registered(self) -> dict[str, int]:
        """列出所有已注册的人"""
        return {name: len(embs) for name, embs in self.face_db.items()}

    def recognize(self, embedding: np.ndarray) -> tuple[str, float]:
        """
        识别单个人脸
        :param embedding: 512维特征向量
        :return: (name, confidence) 或 ("unknown", 0.0)
        """
        if not self.face_db:
            return "unknown", 0.0

        best_name = "unknown"
        best_score = 0.0

        for name, db_embeddings in self.face_db.items():
            db_matrix = np.array(db_embeddings)
            # 余弦相似度
            similarities = np.dot(db_matrix, embedding)
            max_sim = float(np.max(similarities))

            if max_sim > best_score:
                best_score = max_sim
                best_name = name

        if best_score < self.threshold:
            return "unknown", best_score

        return best_name, best_score

    def recognize_batch(self, embeddings: list[np.ndarray]) -> list[tuple[str, float]]:
        """批量识别"""
        return [self.recognize(emb) for emb in embeddings]
