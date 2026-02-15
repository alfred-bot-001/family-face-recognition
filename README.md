# Family Face Recognition

本地部署的家庭人脸识别系统，基于 InsightFace + ArcFace，支持照片和实时视频识别。

## 环境

- Python 3.12 + RTX 4090
- PyTorch 2.5 (CUDA 12.1)
- InsightFace 0.7.3 + ONNX Runtime GPU

## 功能

- 人脸注册（少量照片即可）
- 照片识别
- 实时视频流识别
- FastAPI REST API

## 快速开始

```bash
# 激活环境
source venv/bin/activate

# 注册人脸
python src/train.py --name "max" --data data/max/

# 启动 API
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## 项目结构

```
data/           # 训练照片（按人名分目录）
models/         # 预训练模型 + 特征库
src/
  detector.py   # 人脸检测
  recognizer.py # 人脸识别
  video.py      # 视频流处理
  train.py      # 注册新人脸
  api.py        # FastAPI 服务
tests/
```
