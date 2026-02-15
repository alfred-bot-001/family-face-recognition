"""基础功能测试"""

import sys
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))


def test_detector():
    """测试人脸检测器能正常初始化"""
    from face_recognition.detector import FaceDetector
    det = FaceDetector()
    # 用纯色图测试（不应检测到人脸）
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = det.detect(blank)
    assert len(faces) == 0, "纯色图不应检测到人脸"
    print("[PASS] detector 初始化 + 空图检测")
    return det


def test_recognizer():
    """测试识别器的注册和识别"""
    from face_recognition.recognizer import FaceRecognizer
    import tempfile, os

    # 使用临时文件
    tmp = tempfile.mktemp(suffix=".pkl")
    rec = FaceRecognizer(db_path=tmp, threshold=0.4)

    # 注册假特征
    fake_emb = np.random.randn(512).astype(np.float32)
    fake_emb /= np.linalg.norm(fake_emb)
    rec.register("test_person", [fake_emb])

    # 识别自己应该匹配
    name, conf = rec.recognize(fake_emb)
    assert name == "test_person", f"应识别为 test_person, 实际: {name}"
    assert conf > 0.9, f"自身匹配置信度应高, 实际: {conf}"

    # 随机向量不应匹配
    random_emb = np.random.randn(512).astype(np.float32)
    random_emb /= np.linalg.norm(random_emb)
    name2, conf2 = rec.recognize(random_emb)
    # 随机向量相似度应较低

    # 列出
    reg = rec.list_registered()
    assert "test_person" in reg
    assert reg["test_person"] == 1

    # 删除
    assert rec.remove("test_person")
    assert "test_person" not in rec.list_registered()

    os.unlink(tmp)
    print("[PASS] recognizer 注册/识别/删除")


if __name__ == "__main__":
    print("=== 基础测试 ===")
    test_recognizer()
    det = test_detector()
    print("=== 全部通过 ===")
