"""
本地手势检测模块 - 基于 MediaPipe Hands
复用 ugv_rpi/cv_ctrl.py 的手势识别逻辑 + 挥手检测
在树莓派上直接运行，不需要 GPU
"""

import math
import time
from collections import deque

import cv2
import numpy as np

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False


# ============================================================
#  工具函数（来自 ugv_rpi/cv_ctrl.py）
# ============================================================

def calculate_angle(a1, a2, b1, b2) -> float:
    """计算两个向量之间的角度（度）"""
    va = (a2.x - a1.x, a2.y - a1.y)
    vb = (b2.x - b1.x, b2.y - b1.y)
    dot = va[0] * vb[0] + va[1] * vb[1]
    mag_a = math.sqrt(va[0] ** 2 + va[1] ** 2)
    mag_b = math.sqrt(vb[0] ** 2 + vb[1] ** 2)
    if mag_a * mag_b == 0:
        return 0
    cos_val = max(-1, min(1, dot / (mag_a * mag_b)))
    return math.degrees(math.acos(cos_val))


def calculate_distance(lm1, lm2) -> float:
    """计算两个关键点之间的距离"""
    return ((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2) ** 0.5


# ============================================================
#  手势检测器
# ============================================================

class GestureDetector:
    """
    手势检测（本地运行，与 ugv_rpi 手势逻辑一致）

    支持的手势：
    - wave: 挥手（张开手掌左右摆动）
    - open_palm: 张开手掌
    - fist: 握拳
    - peace: 比 V / 剪刀手
    """

    def __init__(self, log_func=None):
        self.log = log_func or print
        self.mode = "opencv"  # opencv | mediapipe
        self.available = True

        # MediaPipe 作为兜底
        self.mp_hands = None
        self.hands = None
        if MP_AVAILABLE:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
            )

        # 挥手检测参数
        self.wrist_history: deque = deque(maxlen=20)
        self.wave_cooldown = 3.0          # 检测冷却（秒）
        self.last_wave_time = 0
        self.min_direction_changes = 3    # 最少方向变化
        self.history_window = 1.5         # 时间窗口（秒）
        self.min_move = 0.03              # 最小移动

        self.log("INFO", "手势检测已初始化 (OpenCV + MediaPipe兜底)")

    def _finger_extended(self, lms, tip_id: int, pip_id: int) -> bool:
        """手指是否伸直"""
        return lms[tip_id].y < lms[pip_id].y

    def _get_finger_states(self, lms) -> dict:
        """
        获取五指状态（改进：角度 + 几何双判定，降低误判“握拳”）
        """
        HL = self.mp_hands.HandLandmark

        # 四指角度
        index_angle = calculate_angle(
            lms[HL.INDEX_FINGER_MCP], lms[HL.INDEX_FINGER_PIP],
            lms[HL.INDEX_FINGER_PIP], lms[HL.INDEX_FINGER_TIP])
        middle_angle = calculate_angle(
            lms[HL.MIDDLE_FINGER_MCP], lms[HL.MIDDLE_FINGER_PIP],
            lms[HL.MIDDLE_FINGER_PIP], lms[HL.MIDDLE_FINGER_TIP])
        ring_angle = calculate_angle(
            lms[HL.RING_FINGER_MCP], lms[HL.RING_FINGER_PIP],
            lms[HL.RING_FINGER_PIP], lms[HL.RING_FINGER_TIP])
        pinky_angle = calculate_angle(
            lms[HL.WRIST], lms[HL.PINKY_MCP],
            lms[HL.PINKY_MCP], lms[HL.PINKY_TIP])

        # 几何判定：tip 在 pip 上方（图像坐标 y 越小越高）
        y_margin = 0.015
        index_up = lms[HL.INDEX_FINGER_TIP].y < (lms[HL.INDEX_FINGER_PIP].y - y_margin)
        middle_up = lms[HL.MIDDLE_FINGER_TIP].y < (lms[HL.MIDDLE_FINGER_PIP].y - y_margin)
        ring_up = lms[HL.RING_FINGER_TIP].y < (lms[HL.RING_FINGER_PIP].y - y_margin)
        pinky_up = lms[HL.PINKY_TIP].y < (lms[HL.PINKY_PIP].y - y_margin)

        # 拇指：比较 thumb tip 相对 index_mcp 的外展距离
        thumb_tip = lms[HL.THUMB_TIP]
        thumb_ip = lms[HL.THUMB_IP]
        index_mcp = lms[HL.INDEX_FINGER_MCP]
        thumb_tip_dist = calculate_distance(thumb_tip, index_mcp)
        thumb_ip_dist = calculate_distance(thumb_ip, index_mcp)
        thumb_extended = thumb_tip_dist > (thumb_ip_dist * 1.08)

        return {
            "thumb": {"extended": thumb_extended},
            "index": {"angle": index_angle, "extended": (index_angle < 35) or index_up},
            "middle": {"angle": middle_angle, "extended": (middle_angle < 35) or middle_up},
            "ring": {"angle": ring_angle, "extended": (ring_angle < 35) or ring_up},
            "pinky": {"angle": pinky_angle, "extended": (pinky_angle < 55) or pinky_up},
        }

    def _classify_gesture(self, fingers: dict) -> str:
        """根据手指状态分类手势"""
        extended_count = sum(1 for f in fingers.values() if f["extended"])

        # 张开手掌（4指以上伸直）
        if extended_count >= 4:
            return "open_palm"

        # 比 V（食指+中指伸直，其他弯曲）
        if (fingers["index"]["extended"] and fingers["middle"]["extended"]
                and not fingers["ring"]["extended"] and not fingers["pinky"]["extended"]):
            return "peace"

        # 握拳（所有手指弯曲）
        if extended_count <= 1:
            return "fist"

        return "unknown"

    def _check_wave(self) -> bool:
        """分析手腕历史，检测挥手"""
        now = time.time()
        if now - self.last_wave_time < self.wave_cooldown:
            return False

        recent = [(t, x) for t, x in self.wrist_history if now - t < self.history_window]
        if len(recent) < 5:
            return False

        # 统计方向变化
        changes = 0
        prev_dx = 0
        for i in range(1, len(recent)):
            dx = recent[i][1] - recent[i - 1][1]
            if abs(dx) < self.min_move:
                continue
            if prev_dx != 0 and (dx > 0) != (prev_dx > 0):
                changes += 1
            prev_dx = dx

        if changes >= self.min_direction_changes:
            self.last_wave_time = now
            self.wrist_history.clear()
            return True
        return False

    def _detect_opencv(self, frame_bgr: np.ndarray) -> dict:
        """OpenCV 轮廓 + 凹陷点手势检测（主算法）"""
        h, w = frame_bgr.shape[:2]
        area_min = max(1200, int(h * w * 0.01))

        blur = cv2.GaussianBlur(frame_bgr, (7, 7), 0)
        ycrcb = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)
        # 常见肤色阈值（YCrCb）
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"hands_count": 0, "wave_detected": False, "gesture": "none", "hand_landmarks": []}

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < area_min:
            return {"hands_count": 0, "wave_detected": False, "gesture": "none", "hand_landmarks": []}

        hull = cv2.convexHull(cnt, returnPoints=False)
        if hull is None or len(hull) < 4:
            return {"hands_count": 1, "wave_detected": False, "gesture": "unknown", "hand_landmarks": []}

        defects = cv2.convexityDefects(cnt, hull)
        finger_gaps = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, depth = defects[i, 0]
                start = cnt[s][0]
                end = cnt[e][0]
                far = cnt[f][0]
                a = np.linalg.norm(start - end)
                b = np.linalg.norm(start - far)
                c = np.linalg.norm(end - far)
                if b * c == 0:
                    continue
                angle = math.degrees(math.acos(max(-1, min(1, (b*b + c*c - a*a) / (2*b*c)))))
                if angle < 85 and depth > 8000:
                    finger_gaps += 1

        # 结合凸包实心度（solidity）提升稳定性
        hull_pts = cv2.convexHull(cnt)
        hull_area = max(1.0, cv2.contourArea(hull_pts))
        solidity = area / hull_area

        # 张手：手指缝较多 或 实心度偏低
        if finger_gaps >= 2 or solidity < 0.82:
            gesture = "open_palm"
        # 握拳：手指缝少 且 实心度高
        elif finger_gaps <= 1 and solidity > 0.90:
            gesture = "fist"
        else:
            gesture = "unknown"

        M = cv2.moments(cnt)
        cx, cy = 0.5, 0.5
        if M["m00"] != 0:
            cx = (M["m10"] / M["m00"]) / w
            cy = (M["m01"] / M["m00"]) / h

        if gesture == "open_palm":
            self.wrist_history.append((time.time(), cx))
            wave = self._check_wave()
        else:
            wave = False

        return {
            "hands_count": 1,
            "wave_detected": wave,
            "gesture": gesture,
            "hand_landmarks": [{"wrist_x": round(cx, 3), "wrist_y": round(cy, 3), "gesture": gesture, "gaps": finger_gaps, "solidity": round(solidity, 3)}],
        }

    def detect(self, frame_bgr: np.ndarray) -> dict:
        """
        检测手势

        返回:
            {
                "hands_count": int,
                "wave_detected": bool,
                "gesture": str,          # open_palm / peace / fist / unknown / none
                "hand_landmarks": list,   # 简要信息
            }
        """
        # 1) 主算法：OpenCV
        cv_result = self._detect_opencv(frame_bgr)
        if cv_result["gesture"] != "none":
            return cv_result

        # 2) 兜底：MediaPipe
        result = {
            "hands_count": 0,
            "wave_detected": False,
            "gesture": "none",
            "hand_landmarks": [],
        }
        if not self.hands:
            return result

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_result = self.hands.process(frame_rgb)
        if not mp_result.multi_hand_landmarks:
            self.wrist_history.clear()
            return result

        result["hands_count"] = len(mp_result.multi_hand_landmarks)
        priority = {"open_palm": 4, "peace": 3, "unknown": 2, "fist": 1, "none": 0}
        best_gesture = "none"

        for hand_lms in mp_result.multi_hand_landmarks:
            lms = hand_lms.landmark
            HL = self.mp_hands.HandLandmark
            wrist = lms[HL.WRIST]
            fingers = self._get_finger_states(lms)
            gesture = self._classify_gesture(fingers)
            ext_cnt = sum(1 for f in fingers.values() if f.get("extended"))
            result["hand_landmarks"].append({
                "wrist_x": round(wrist.x, 3),
                "wrist_y": round(wrist.y, 3),
                "gesture": gesture,
                "extended_count": ext_cnt,
            })

            if priority.get(gesture, 0) > priority.get(best_gesture, 0):
                best_gesture = gesture
            if gesture == "open_palm":
                self.wrist_history.append((time.time(), wrist.x))
                if self._check_wave():
                    result["wave_detected"] = True

        result["gesture"] = best_gesture
        return result
