"""人脸注册/训练脚本 - 从照片目录提取人脸并注册特征"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .detector import FaceDetector
from .recognizer import FaceRecognizer

PROJECT_ROOT = Path(__file__).parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def register_from_directory(
    name: str,
    data_dir: str | Path,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    extract_dir: str | Path | None = None,
    select: str = "largest",
) -> dict:
    """
    从目录中的照片注册人脸

    :param name: 人名
    :param data_dir: 包含照片的目录
    :param detector: 人脸检测器
    :param recognizer: 人脸识别器
    :param extract_dir: 提取的人脸保存目录（可选）
    :param select: 多人脸时的选择策略 "largest" | "all"
    :return: 训练报告 dict
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"[错误] 目录不存在: {data_dir}")
        return {"name": name, "status": "error", "message": "目录不存在"}

    # 过滤掉 macOS 资源文件
    image_files = sorted([
        f for f in data_path.iterdir()
        if f.suffix.lower() in IMAGE_EXTS and not f.name.startswith("._")
    ])

    if not image_files:
        print(f"[错误] 目录中没有图片: {data_dir}")
        return {"name": name, "status": "error", "message": "无图片"}

    # 提取目录
    if extract_dir is None:
        extract_dir = PROJECT_ROOT / "data" / f"{name}_faces"
    extract_path = Path(extract_dir)
    extract_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  注册人脸: {name}")
    print(f"  照片数量: {len(image_files)}")
    print(f"  人脸提取目录: {extract_path}")
    print(f"{'='*60}")

    embeddings = []
    report = {
        "name": name,
        "total_images": len(image_files),
        "faces_found": 0,
        "faces_registered": 0,
        "skipped": [],
        "multi_face": [],
        "details": [],
    }

    for img_file in image_files:
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"  [跳过] 无法读取: {img_file.name}")
            report["skipped"].append({"file": img_file.name, "reason": "无法读取"})
            continue

        faces = detector.extract_faces(image, margin=0.3, min_size=40)

        if len(faces) == 0:
            print(f"  [跳过] 未检测到人脸: {img_file.name}")
            report["skipped"].append({"file": img_file.name, "reason": "未检测到人脸"})
            continue

        report["faces_found"] += len(faces)

        if len(faces) > 1:
            report["multi_face"].append({"file": img_file.name, "count": len(faces)})
            if select == "largest":
                # 取最大人脸
                faces.sort(
                    key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
                    reverse=True,
                )
                faces = [faces[0]]
                print(f"  [多脸] {img_file.name}: {report['multi_face'][-1]['count']} 张，取最大")

        for i, face in enumerate(faces):
            # 保存裁剪的人脸
            suffix = f"_{i}" if len(faces) > 1 else ""
            out_name = f"{img_file.stem}{suffix}.jpg"
            out_path = extract_path / out_name
            cv2.imwrite(str(out_path), face["crop"])

            embeddings.append(face["embedding"])
            detail = {
                "file": img_file.name,
                "det_score": round(face["score"], 4),
                "age": face.get("age"),
                "gender": face.get("gender"),
                "face_saved": out_name,
            }
            report["details"].append(detail)
            print(f"  [OK] {img_file.name} → {out_name} "
                  f"(置信度: {face['score']:.3f}, 年龄: {face.get('age')}, 性别: {face.get('gender')})")

    if embeddings:
        recognizer.register(name, embeddings)
        report["faces_registered"] = len(embeddings)
        report["status"] = "success"
        print(f"\n  ✅ '{name}' 注册完成: {len(embeddings)}/{len(image_files)} 张")
    else:
        report["status"] = "failed"
        print(f"\n  ❌ '{name}' 注册失败: 没有有效人脸")

    return report


def register_all(
    data_root: str | Path,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
) -> list[dict]:
    """批量注册 data/ 下所有子目录"""
    data_path = Path(data_root)
    reports = []

    for person_dir in sorted(data_path.iterdir()):
        if person_dir.is_dir() and not person_dir.name.startswith("."):
            # 跳过 *_faces 提取目录
            if person_dir.name.endswith("_faces"):
                continue
            report = register_from_directory(
                name=person_dir.name,
                data_dir=person_dir,
                detector=detector,
                recognizer=recognizer,
            )
            reports.append(report)

    # 汇总报告
    print(f"\n{'='*60}")
    print("  训练汇总")
    print(f"{'='*60}")
    total_reg = 0
    for r in reports:
        status = "✅" if r.get("status") == "success" else "❌"
        reg = r.get("faces_registered", 0)
        total_reg += reg
        print(f"  {status} {r['name']}: {reg}/{r.get('total_images', 0)} 张注册成功")
        if r.get("skipped"):
            print(f"     跳过 {len(r['skipped'])} 张: {[s['reason'] for s in r['skipped']]}")
    print(f"\n  总计注册: {total_reg} 张人脸")
    print(f"  特征库: {recognizer.list_registered()}")

    # 保存报告
    report_path = PROJECT_ROOT / "models" / "train_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        # 移除 numpy 数组以便序列化
        clean_reports = []
        for r in reports:
            cr = {k: v for k, v in r.items() if k != "details"}
            cr["details"] = [
                {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}
                for d in r.get("details", [])
            ]
            clean_reports.append(cr)
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "reports": clean_reports,
        }, f, ensure_ascii=False, indent=2)
    print(f"  报告已保存: {report_path}")

    return reports


def main():
    parser = argparse.ArgumentParser(description="人脸注册工具")
    parser.add_argument("--name", type=str, help="要注册的人名（单人模式）")
    parser.add_argument("--data", type=str, help="照片目录（单人模式）")
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data"),
                        help="批量注册的根目录")
    parser.add_argument("--batch", action="store_true", help="批量注册 data-root 下所有子目录")
    parser.add_argument("--list", action="store_true", help="列出已注册的人脸")
    parser.add_argument("--remove", type=str, help="删除指定人名的注册信息")
    parser.add_argument("--select", choices=["largest", "all"], default="largest",
                        help="多人脸选择策略")
    args = parser.parse_args()

    if args.list:
        recognizer = FaceRecognizer()
        registered = recognizer.list_registered()
        if registered:
            print("已注册人脸:")
            for name, count in registered.items():
                print(f"  {name}: {count} 张")
        else:
            print("未注册任何人脸")
        return

    if args.remove:
        recognizer = FaceRecognizer()
        if recognizer.remove(args.remove):
            print(f"已删除 '{args.remove}'")
        else:
            print(f"未找到 '{args.remove}'")
        return

    detector = FaceDetector()
    recognizer = FaceRecognizer()

    if args.batch:
        register_all(args.data_root, detector, recognizer)
    elif args.name and args.data:
        register_from_directory(args.name, args.data, detector, recognizer)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
