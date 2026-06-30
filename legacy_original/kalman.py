import numpy as np

class KalmanFilter4D:
    """4D卡尔曼滤波器 (x, y, vx, vy) — 空间换时间优化版"""

    def __init__(self, q: float = 0.06, r: float = 0.12):
        self.q = q
        self.r = r
        # 预分配常量矩阵（空间换时间，避免每次重建）
        self._I4 = np.eye(4, dtype=np.float64)
        self._H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=np.float64)
        self._Ht = self._H.T.copy()                    # 预计算转置
        self._F_base = np.eye(4, dtype=np.float64)     # 复用F矩阵，只改 dt 位置
        self._R2 = np.eye(2, dtype=np.float64)
        self.reset()

    def apply_ego_motion(self, dx: float, dy: float, scaler: float = 2.7):
        if self.initialized:
            self.state[0] -= dx * scaler
            self.state[1] -= dy * scaler

    def predict(self, dt: float):
        if not self.initialized:
            return 0.0, 0.0
        # 复用 F_base，只更新 dt 相关位置（避免每次 np.array 分配）
        self._F_base[0, 2] = dt
        self._F_base[1, 3] = dt
        self.state = self._F_base @ self.state
        self.P = self._F_base @ self.P @ self._F_base.T + self.Q
        return float(self.state[0]), float(self.state[1])

    def update(self, mx: float, my: float):
        if not self.initialized:
            self.state = np.array([mx, my, 0.0, 0.0], dtype=np.float64)
            self.initialized = True
            return mx, my

        # S = H @ P @ Ht + R  (2x2)
        PHt = self.P @ self._Ht          # 4x2
        S = self._H @ PHt + self.R       # 2x2

        # 2x2 解析求逆（比 np.linalg.inv 快 5~10x）
        s00, s01 = S[0, 0], S[0, 1]
        s10, s11 = S[1, 0], S[1, 1]
        det = s00 * s11 - s01 * s10
        if abs(det) < 1e-15:
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)
        else:
            inv_det = 1.0 / det
            S_inv = np.array([[s11 * inv_det, -s01 * inv_det],
                              [-s10 * inv_det, s00 * inv_det]], dtype=np.float64)

        K = PHt @ S_inv                   # 4x2
        z = np.array([mx, my], dtype=np.float64)
        y = z - self._H @ self.state
        self.state = self.state + K @ y
        self.P = (self._I4 - K @ self._H) @ self.P

        if np.isnan(self.state).any() or np.isinf(self.state).any():
            self.reset()
            return mx, my
        return float(self.state[0]), float(self.state[1])

    def decay_velocity(self, decay_factor: float = 0.86):
        if self.initialized:
            self.state[2] *= decay_factor
            self.state[3] *= decay_factor

    def reset(self):
        self.state = np.zeros(4, dtype=np.float64)
        self.P = self._I4 * 10.0
        self.Q = self._I4 * self.q
        self.R = self._R2 * self.r
        self.initialized = False
