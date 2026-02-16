# Family Face Recognition

本地部署的家庭人脸识别系统，基于 InsightFace + ArcFace，支持照片和实时视频识别。

## 架构

```
┌─────────────────────┐         ┌──────────────────────────┐
│  树莓派 (老三)       │  HTTP   │  台式机 (老二)            │
│  摄像头抓帧          │ ──────→ │  GPU 人脸识别 API         │
│  Web 实时视频流      │ ←────── │  InsightFace + ArcFace   │
│  client/app.py      │  JSON   │  face_recognition/api.py │
└─────────────────────┘         └──────────────────────────┘
         ↓
  浏览器查看实时视频 + 识别结果
```

## 环境

- **服务端**: Python 3.12 + RTX 4090, PyTorch 2.5 (CUDA 12.1), InsightFace 0.7.3
- **客户端**: Python 3.12 + OpenCV (headless) + Flask + Requests

## 快速开始

### 服务端 (台式机)

```bash
source venv/bin/activate

# 注册人脸
python -m face_recognition.train --batch

# 启动 API
uvicorn face_recognition.api:app --host 0.0.0.0 --port 8000
```

### 客户端 (树莓派)

```bash
cd client
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python app.py --api http://192.168.0.69:8000
# 浏览器访问 http://<树莓派IP>:5000
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/registered` | 列出已注册人脸 |
| POST | `/recognize` | 识别照片中的人脸 |
| POST | `/recognize/annotated` | 识别并返回标注图 |
| POST | `/register?name=xxx` | 注册单张人脸 |
| DELETE | `/register/{name}` | 删除注册 |
| POST | `/recognize/video` | 识别视频文件 |

## 项目结构

```
face_recognition/       # 服务端核心代码
  api.py               # FastAPI 服务
  detector.py          # 人脸检测 (RetinaFace)
  recognizer.py        # 人脸识别 (ArcFace)
  train.py             # 人脸注册/训练
  video.py             # 视频流处理
  static/index.html    # 服务端测试页面
client/                # 树莓派客户端
  app.py               # 摄像头 + Web 服务
  static/index.html    # 实时监控页面
  requirements.txt     # 客户端依赖
data/                  # 训练照片（按人名分目录）
models/                # 特征库
```
