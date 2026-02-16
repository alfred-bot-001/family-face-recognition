"""FastAPI 人脸识别服务"""

import io
import time
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .detector import FaceDetector
from .recognizer import FaceRecognizer
from .video import VideoRecognizer

STATIC_DIR = Path(__file__).parent / "static"

# 全局实例
detector: FaceDetector = None
recognizer: FaceRecognizer = None
video_recognizer: VideoRecognizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化模型"""
    global detector, recognizer, video_recognizer
    print("[API] 正在加载模型...")
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    video_recognizer = VideoRecognizer(detector, recognizer)
    print("[API] 模型加载完成")
    yield
    print("[API] 服务关闭")


app = FastAPI(
    title="家庭人脸识别 API",
    description="本地部署的人脸识别服务，支持照片和视频",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - 允许客户端跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"service": "family-face-recognition", "status": "running"}


@app.get("/registered")
async def list_registered():
    """列出已注册的人脸"""
    return {"registered": recognizer.list_registered()}


@app.post("/recognize")
async def recognize_image(file: UploadFile = File(...)):
    """
    识别上传照片中的人脸
    返回每张人脸的身份和置信度
    """
    start = time.time()

    # 读取图片
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="无法解析图片")

    # 检测人脸
    faces = detector.detect_with_info(image)

    if not faces:
        return JSONResponse(content={
            "faces": [],
            "message": "未检测到人脸",
            "time_ms": round((time.time() - start) * 1000),
        })

    # 识别
    results = []
    for face in faces:
        name, confidence = recognizer.recognize(face["embedding"])
        results.append({
            "name": name,
            "confidence": round(confidence, 4),
            "bbox": face["bbox"],
            "det_score": round(face["score"], 4),
            "age": face.get("age"),
            "gender": face.get("gender"),
        })

    elapsed = round((time.time() - start) * 1000)
    return {
        "faces": results,
        "count": len(results),
        "time_ms": elapsed,
    }


@app.post("/recognize/annotated")
async def recognize_and_annotate(file: UploadFile = File(...)):
    """
    识别并返回标注后的图片
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="无法解析图片")

    faces = detector.detect_with_info(image)
    draw_results = []
    for face in faces:
        name, confidence = recognizer.recognize(face["embedding"])
        draw_results.append({
            "bbox": face["bbox"],
            "name": name,
            "confidence": confidence,
            "score": face["score"],
        })

    annotated = VideoRecognizer.draw_results(image, draw_results)

    _, buffer = cv2.imencode(".jpg", annotated)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")


@app.post("/register")
async def register_face(name: str, file: UploadFile = File(...)):
    """
    上传单张照片注册人脸
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="无法解析图片")

    faces = detector.detect_with_info(image)

    if not faces:
        raise HTTPException(status_code=400, detail="照片中未检测到人脸")

    if len(faces) > 1:
        # 取最大人脸
        faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)

    recognizer.register(name, [faces[0]["embedding"]])
    return {
        "message": f"已注册 '{name}'",
        "total": len(recognizer.face_db.get(name, [])),
    }


@app.delete("/register/{name}")
async def remove_face(name: str):
    """删除已注册的人脸"""
    if recognizer.remove(name):
        return {"message": f"已删除 '{name}'"}
    raise HTTPException(status_code=404, detail=f"未找到 '{name}'")


@app.post("/recognize/video")
async def recognize_video(file: UploadFile = File(...)):
    """
    识别上传的视频文件
    返回识别到的人物汇总
    """
    start = time.time()

    # 保存临时文件
    tmp_path = Path("/tmp/face_rec_upload.mp4")
    contents = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(contents)

    # 处理视频
    all_results = video_recognizer.process_video_file(str(tmp_path))

    # 汇总
    seen = {}
    for fr in all_results:
        for face in fr["faces"]:
            name = face["name"]
            if name != "unknown":
                if name not in seen:
                    seen[name] = {"frames": 0, "max_confidence": 0}
                seen[name]["frames"] += 1
                seen[name]["max_confidence"] = max(seen[name]["max_confidence"], face["confidence"])

    # 清理
    tmp_path.unlink(missing_ok=True)

    elapsed = round((time.time() - start) * 1000)
    return {
        "people": {k: {**v, "max_confidence": round(v["max_confidence"], 4)} for k, v in seen.items()},
        "total_frames": len(all_results),
        "time_ms": elapsed,
    }
