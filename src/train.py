"""人脸注册/训练脚本 - 从照片目录注册人脸特征"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from .detector import FaceDetector
from .recognizer import FaceRecognizer


def register_from_directory(
    name: str,
    data_dir: str,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
) -> int:
    """
    从目录中的照片注册人脸
    :param name: 人名
    :param data_dir: 包含照片的目录
    :param detector: 人脸检测器
    :param recognizer: 人脸识别器
    :return: 成功注册的人脸数量
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"[错误] 目录不存在: {data_dir}")
        return 0

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [f for f in data_path.iterdir() if f.suffix.lower() in image_exts]

    if not image_files:
        print(f"[错误] 目录中没有图片: {data_dir}")
        return 0

    print(f"[训练] 正在处理 '{name}' 的 {len(image_files)} 张照片...")
    embeddings = []

    for img_file in image_files:
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"  [跳过] 无法读取: {img_file.name}")
            continue

        faces = detector.detect_with_info(image)

        if len(faces) == 0:
            print(f"  [跳过] 未检测到人脸: {img_file.name}")
            continue
        elif len(faces) > 1:
            # 多张人脸时取最大的（面积最大）
            faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)
            print(f"  [注意] {img_file.name} 检测到 {len(faces)} 张人脸，取最大的")

        embeddings.append(faces[0]["embedding"])
        print(f"  [OK] {img_file.name} (置信度: {faces[0]['score']:.3f})")

    if embeddings:
        recognizer.register(name, embeddings)
        print(f"[训练] 完成! '{name}' 注册了 {len(embeddings)} 张人脸")
    else:
        print(f"[训练] 失败! 没有有效的人脸特征")

    return len(embeddings)


def register_all(data_root: str, detector: FaceDetector, recognizer: FaceRecognizer):
    """
    从 data/ 目录下的所有子目录批量注册
    data/
      max/
      wife/
      son/
    """
    data_path = Path(data_root)
    total = 0
    for person_dir in sorted(data_path.iterdir()):
        if person_dir.is_dir() and not person_dir.name.startswith("."):
            count = register_from_directory(
                name=person_dir.name,
                data_dir=str(person_dir),
                detector=detector,
                recognizer=recognizer,
            )
            total += count
    print(f"\n[训练] 全部完成，共注册 {total} 张人脸")
    print(f"[训练] 已注册: {recognizer.list_registered()}")


def main():
    parser = argparse.ArgumentParser(description="人脸注册工具")
    parser.add_argument("--name", type=str, help="要注册的人名（单人模式）")
    parser.add_argument("--data", type=str, help="照片目录（单人模式）")
    parser.add_argument("--data-root", type=str, default="data", help="批量注册的根目录")
    parser.add_argument("--batch", action="store_true", help="批量注册 data-root 下所有子目录")
    parser.add_argument("--list", action="store_true", help="列出已注册的人脸")
    parser.add_argument("--remove", type=str, help="删除指定人名的注册信息")
    args = parser.parse_args()

    detector = FaceDetector()
    recognizer = FaceRecognizer()

    if args.list:
        registered = recognizer.list_registered()
        if registered:
            print("已注册人脸:")
            for name, count in registered.items():
                print(f"  {name}: {count} 张")
        else:
            print("未注册任何人脸")
        return

    if args.remove:
        if recognizer.remove(args.remove):
            print(f"已删除 '{args.remove}'")
        else:
            print(f"未找到 '{args.remove}'")
        return

    if args.batch:
        register_all(args.data_root, detector, recognizer)
    elif args.name and args.data:
        register_from_directory(args.name, args.data, detector, recognizer)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
