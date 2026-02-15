"""识别准确率测试 - 用训练数据自测 + 交叉验证"""

import json
import time
from pathlib import Path

import cv2
import numpy as np

from face_recognition.detector import FaceDetector
from face_recognition.recognizer import FaceRecognizer

PROJECT_ROOT = Path(__file__).parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def test_self_recognition(detector: FaceDetector, recognizer: FaceRecognizer):
    """用原始照片测试识别率"""
    print("\n" + "=" * 60)
    print("  测试 1: 原始照片识别")
    print("=" * 60)

    data_dir = PROJECT_ROOT / "data" / "son"
    files = sorted([
        f for f in data_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTS and not f.name.startswith("._")
    ])

    correct = 0
    failed = 0
    no_face = 0
    total = len(files)
    confidences = []

    for f in files:
        image = cv2.imread(str(f))
        if image is None:
            continue

        faces = detector.detect_with_info(image)
        if not faces:
            no_face += 1
            print(f"  [无脸] {f.name}")
            continue

        # 取最大脸
        faces.sort(
            key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
            reverse=True,
        )
        name, conf = recognizer.recognize(faces[0]["embedding"])
        confidences.append(conf)

        if name == "son":
            correct += 1
        else:
            failed += 1
            print(f"  [错误] {f.name} → {name} ({conf:.4f})")

    tested = correct + failed
    accuracy = correct / tested * 100 if tested else 0
    avg_conf = np.mean(confidences) if confidences else 0
    min_conf = np.min(confidences) if confidences else 0

    print(f"\n  结果: {correct}/{tested} 正确 ({accuracy:.1f}%)")
    print(f"  无脸跳过: {no_face}")
    print(f"  平均置信度: {avg_conf:.4f}")
    print(f"  最低置信度: {min_conf:.4f}")

    return {"test": "self_recognition", "accuracy": accuracy, "correct": correct,
            "tested": tested, "avg_conf": float(avg_conf), "min_conf": float(min_conf)}


def test_extracted_faces(detector: FaceDetector, recognizer: FaceRecognizer):
    """用提取的人脸图直接测试（无需再检测）"""
    print("\n" + "=" * 60)
    print("  测试 2: 提取人脸直接识别")
    print("=" * 60)

    face_dir = PROJECT_ROOT / "data" / "son_faces"
    files = sorted([
        f for f in face_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTS and not f.name.startswith("_")
    ])

    correct = 0
    failed = 0
    no_face = 0
    confidences = []

    for f in files:
        image = cv2.imread(str(f))
        if image is None:
            continue

        faces = detector.detect_with_info(image)
        if not faces:
            no_face += 1
            continue

        name, conf = recognizer.recognize(faces[0]["embedding"])
        confidences.append(conf)

        if name == "son":
            correct += 1
        else:
            failed += 1
            print(f"  [错误] {f.name} → {name} ({conf:.4f})")

    tested = correct + failed
    accuracy = correct / tested * 100 if tested else 0
    avg_conf = np.mean(confidences) if confidences else 0

    print(f"\n  结果: {correct}/{tested} 正确 ({accuracy:.1f}%)")
    print(f"  无脸跳过: {no_face}")
    print(f"  平均置信度: {avg_conf:.4f}")

    return {"test": "extracted_faces", "accuracy": accuracy, "correct": correct,
            "tested": tested, "avg_conf": float(avg_conf)}


def test_unknown_rejection(detector: FaceDetector, recognizer: FaceRecognizer):
    """用随机噪声/合成人脸测试是否正确拒绝未知人"""
    print("\n" + "=" * 60)
    print("  测试 3: 未知人脸拒绝")
    print("=" * 60)

    rejected = 0
    total = 100

    for i in range(total):
        # 随机 512 维特征
        fake = np.random.randn(512).astype(np.float32)
        fake /= np.linalg.norm(fake)
        name, conf = recognizer.recognize(fake)
        if name == "unknown":
            rejected += 1

    reject_rate = rejected / total * 100
    print(f"  随机特征拒绝率: {rejected}/{total} ({reject_rate:.1f}%)")

    return {"test": "unknown_rejection", "reject_rate": reject_rate,
            "rejected": rejected, "total": total}


def test_speed(detector: FaceDetector, recognizer: FaceRecognizer):
    """测试识别速度"""
    print("\n" + "=" * 60)
    print("  测试 4: 识别速度")
    print("=" * 60)

    # 找一张测试图
    face_dir = PROJECT_ROOT / "data" / "son_faces"
    test_file = sorted(face_dir.glob("*.jpg"))[0]
    image = cv2.imread(str(test_file))

    # 预热
    for _ in range(3):
        faces = detector.detect_with_info(image)
        if faces:
            recognizer.recognize(faces[0]["embedding"])

    # 检测速度
    n = 20
    start = time.time()
    for _ in range(n):
        detector.detect_with_info(image)
    det_time = (time.time() - start) / n * 1000

    # 识别速度（纯比对）
    embedding = faces[0]["embedding"]
    start = time.time()
    for _ in range(1000):
        recognizer.recognize(embedding)
    rec_time = (time.time() - start) / 1000 * 1000

    # 端到端
    start = time.time()
    for _ in range(n):
        faces = detector.detect_with_info(image)
        if faces:
            recognizer.recognize(faces[0]["embedding"])
    e2e_time = (time.time() - start) / n * 1000

    print(f"  人脸检测: {det_time:.1f} ms/张")
    print(f"  特征比对: {rec_time:.3f} ms/次")
    print(f"  端到端:   {e2e_time:.1f} ms/张")
    print(f"  理论 FPS: {1000/e2e_time:.1f}")

    return {"test": "speed", "detection_ms": round(det_time, 1),
            "matching_ms": round(rec_time, 3), "e2e_ms": round(e2e_time, 1),
            "fps": round(1000 / e2e_time, 1)}


def main():
    print("加载模型...")
    detector = FaceDetector()
    recognizer = FaceRecognizer()

    print(f"已注册: {recognizer.list_registered()}")

    results = []
    results.append(test_self_recognition(detector, recognizer))
    results.append(test_extracted_faces(detector, recognizer))
    results.append(test_unknown_rejection(detector, recognizer))
    results.append(test_speed(detector, recognizer))

    # 汇总
    print("\n" + "=" * 60)
    print("  总结")
    print("=" * 60)
    for r in results:
        name = r["test"]
        if "accuracy" in r:
            print(f"  {name}: {r['accuracy']:.1f}% 准确率")
        elif "reject_rate" in r:
            print(f"  {name}: {r['reject_rate']:.1f}% 拒绝率")
        elif "fps" in r:
            print(f"  {name}: {r['e2e_ms']}ms/张, {r['fps']} FPS")

    # 保存
    report_path = PROJECT_ROOT / "models" / "test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  报告已保存: {report_path}")


if __name__ == "__main__":
    main()
