# 树莓派客户端 - 家庭人脸识别

摄像头抓帧 → 发送到服务端 API 识别 → 实时显示视频流和识别结果

## 架构

```
[树莓派 摄像头] → JPEG → [台式机 GPU API] → 识别结果 JSON
       ↓                                         ↓
  [MJPEG 视频流] ← 标注画框 ← ─ ─ ─ ─ ─ ─ ─ ─ ┘
       ↓
  [浏览器实时查看]
```

## 部署

```bash
# 1. 克隆代码
cd ~/workspace-agent
git clone https://github.com/alfred-bot-001/family-face-recognition.git
cd family-face-recognition/client

# 2. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行
python app.py --api http://192.168.0.69:8000
```

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--api` | `http://192.168.0.69:8000` | 服务端 API 地址 |
| `--camera` | `0` | 摄像头编号 |
| `--width` | `640` | 分辨率宽度 |
| `--height` | `480` | 分辨率高度 |
| `--fps` | `10` | 识别帧率上限 |
| `--port` | `5000` | Web 服务端口 |

## 访问

启动后打开浏览器访问: `http://<树莓派IP>:5000`

- 实时视频流 + 人脸框标注
- 侧边栏显示识别结果、FPS 状态
- 识别日志记录

## 注意

- 摄像头使用 V4L2 接口，确保 `/dev/video0` 存在
- 如果用 CSI 摄像头，可能需要 `libcamera` 配合
- 网络延迟会影响识别速度，建议在同一局域网使用
