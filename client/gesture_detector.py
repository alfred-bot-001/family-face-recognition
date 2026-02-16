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
        self.available = MP_AVAILABLE

        if not MP_AVAILABLE:
            self.log("WARN", "mediapipe 未安装，手势检测不可用")
            return

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

        self.log("INFO", "手势检测已初始化 (MediaPipe Hands)")

    def _finger_extended(self, lms, tip_id: int, pip_id: int) -> bool:
        """手指是否伸直"""
        return lms[tip_id].y < lms[pip_id].y

    def _get_finger_states(self, lms) -> dict:
        """
        获取五指状态（复用 ugv_rpi calculate_angle 逻辑）
        返回每根手指的弯曲角度和是否伸直
        """
        HL = self.mp_hands.HandLandmark

        # 四指弯曲角度（与 ugv_rpi 一致）
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

        # 拇指（通过 x 轴距离判断）
        thumb_extended = abs(lms[HL.THUMB_TIP].x - lms[HL.WRIST].x) > 0.06

        return {
            "thumb": {"extended": thumb_extended},
            "index": {"angle": index_angle, "extended": index_angle < 20},
            "middle": {"angle": middle_angle, "extended": middle_angle < 20},
            "ring": {"angle": ring_angle, "extended": ring_angle < 20},
            "pinky": {"angle": pinky_angle, "extended": pinky_angle < 40},
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
        result = {
            "hands_count": 0,
            "wave_detected": False,
            "gesture": "none",
            "hand_landmarks": [],
        }

        if not self.available:
            return result

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_result = self.hands.process(frame_rgb)

        if not mp_result.multi_hand_landmarks:
            self.wrist_history.clear()
            return result

        result["hands_count"] = len(mp_result.multi_hand_landmarks)

        for hand_lms in mp_result.multi_hand_landmarks:
            lms = hand_lms.landmark
            HL = self.mp_hands.HandLandmark
            wrist = lms[HL.WRIST]

            # 手指状态
            fingers = self._get_finger_states(lms)
            gesture = self._classify_gesture(fingers)

            hand_info = {
                "wrist_x": round(wrist.x, 3),
                "wrist_y": round(wrist.y, 3),
                "gesture": gesture,
            }
            result["hand_landmarks"].append(hand_info)
            result["gesture"] = gesture  # 最后一只手的手势

            # 张开手掌时跟踪挥手
            if gesture == "open_palm":
                self.wrist_history.append((time.time(), wrist.x))
                if self._check_wave():
                    result["wave_detected"] = True

        return result
