import time, math, threading
import serial
import serial.tools.list_ports
from log_utils import log


class LeonardoMouseDriver:
    HEADER = 0xAA
    CMD_MOVE = 0x01
    CMD_PRESS = 0x02
    CMD_RELEASE = 0x03
    CMD_CLICK = 0x04
    CMD_MOVE_PRESS = 0x05
    CMD_MOVE_RELEASE = 0x06
    CMD_HEARTBEAT = 0xFF

    def __init__(self, port="auto", baud=115200):
        self.ser = None
        self.initialized = False
        self.port = port
        self.baud = baud
        self._lock = threading.Lock()
        self._reconnect_lock = threading.Lock()
        self._reconnecting = False
        self._next_reconnect_at = 0.0
        self._closed = False
        # 预构建心跳包（空间换时间）
        self._heartbeat_pkt = self._build_packet(self.CMD_HEARTBEAT, 0, 0)
        self._connect()

    def _find_leonardo_port(self):
        ports = serial.tools.list_ports.comports()
        for p in ports:
            desc = (p.description or "").lower()
            hwid = (p.hwid or "").lower()
            if "2341:8036" in hwid or "2341:0036" in hwid:
                return p.device
            if "leonardo" in desc or "arduino" in desc:
                return p.device
        if ports:
            return ports[-1].device
        return None

    def _connect(self):
        try:
            if self._closed:
                return
            if self.port == "auto":
                detected = self._find_leonardo_port()
                if detected is None:
                    log("未检测到 Leonardo，请检查USB连接", "ERROR")
                    return
                self.port = detected
                log(f"自动检测到 Leonardo: {self.port}", "SUCCESS")
            self.ser = serial.Serial(
                port=self.port, baudrate=self.baud,
                timeout=0.05, write_timeout=0.05)
            time.sleep(2.0)
            if self._closed:
                try:
                    self.ser.close()
                except Exception:
                    pass
                return
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            if self._heartbeat():
                self.initialized = True
                log(f"Leonardo HID 驱动就绪 @ {self.port} ({self.baud}bps)", "SUCCESS")
            else:
                log("Leonardo 握手失败，请检查固件", "ERROR")
        except Exception as e:
            log(f"Leonardo 连接失败: {e}", "ERROR")

    def _heartbeat(self) -> bool:
        try:
            self.ser.write(self._heartbeat_pkt)
            resp = self.ser.read(1)
            return len(resp) > 0 and resp[0] == 0xBB
        except Exception:
            return False

    def _build_packet(self, cmd: int, dx: int, dy: int) -> bytes:
        dx_u8 = dx & 0xFF
        dy_u8 = dy & 0xFF
        checksum = (cmd + dx_u8 + dy_u8) & 0xFF
        return bytes([self.HEADER, cmd, dx_u8, dy_u8, checksum])

    def _send_cmd(self, cmd: int, dx: int = 0, dy: int = 0) -> bool:
        if self._closed:
            return False
        if not self.initialized:
            self._start_reconnect()
            return False
        try:
            with self._lock:
                pkt = self._build_packet(cmd, dx, dy)
                self.ser.write(pkt)
                if self.ser.in_waiting > 16:
                    self.ser.reset_input_buffer()
                return True
        except serial.SerialException:
            self.initialized = False
            log("Leonardo 串口断开，尝试重连...", "WARN")
            self._start_reconnect()
            return False
        except Exception:
            return False

    def _start_reconnect(self):
        with self._reconnect_lock:
            now = time.perf_counter()
            if self._closed or self._reconnecting or now < self._next_reconnect_at:
                return
            self._next_reconnect_at = now + 1.0
            self._reconnecting = True
        threading.Thread(target=self._reconnect_async, daemon=True).start()

    def _reconnect_async(self):
        try:
            self._reconnect()
        finally:
            with self._reconnect_lock:
                self._reconnecting = False

    def _reconnect(self):
        self.initialized = False
        with self._lock:
            try:
                if self.ser:
                    self.ser.close()
            except Exception:
                pass
        time.sleep(1.0)
        self._connect()

    def move(self, dx, dy) -> bool:
        if math.isnan(dx) or math.isnan(dy):
            return False
        dx = max(-127, min(127, int(dx)))
        dy = max(-127, min(127, int(dy)))
        if dx == 0 and dy == 0:
            return False
        return self._send_cmd(self.CMD_MOVE, dx, dy)

    def press_left_click(self) -> bool:
        return self._send_cmd(self.CMD_PRESS)

    def release_left_click(self) -> bool:
        return self._send_cmd(self.CMD_RELEASE)

    def close(self):
        try:
            if self.initialized:
                self._send_cmd(self.CMD_RELEASE)
            self._closed = True
            with self._lock:
                if self.ser:
                    self.ser.close()
                self.initialized = False
        except Exception:
            self._closed = True
            pass
