from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import json
import os
import platform
import socket
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from src.app_paths import user_data_dir
except Exception:
    def user_data_dir() -> Path:  # type: ignore
        return Path.home() / ".visionforge"

PRODUCT_ID = "VISIONFORGE"
ACCEPTED_PRODUCT_IDS = {"VISIONFORGE", "V17_8_RUNTIME_GUI"}
LICENSE_FILE_NAME = "license.key"

# Offline RSA public key. Keep the matching private key only on the owner's machine.
# Replace by running: python tools/license_keygen.py --init-keys
PUBLIC_KEY_N = int("22459416822622734061776761746190779946552826640552470861054923585952698577920808175201881289778403840688615550099227602265442782731250435571435224270679644782764424935377317705135447640366383464500189111817754253349671306405400772182381586502961175837600190241513157475002389653945347574784525547375326189053103921113588688521698418217617651572515195178284567216738736008743196217191703795693930648258498337000354961508057687068774733708994972245386157472979673904938363709996113288707713260384572972122063970816984952681322414726951238266328664569715780745489051783104265180575318956531366563729046018356568917008113")
PUBLIC_KEY_E = 65537
RSA_BYTES = 256
DIGESTINFO_SHA256_PREFIX = bytes.fromhex("3031300d060960864801650304020105000420")
MACHINE_HASH_SALT = "V17_8_RUNTIME_GUI_MACHINE_V1"


def _b64u_decode(s: str) -> bytes:
    s = s.strip().replace("-", "+").replace("_", "/")
    s += "=" * (-len(s) % 4)
    return base64.b64decode(s.encode("ascii"))


def _b64u_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def canonical_payload(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def machine_code() -> str:
    raw_parts = []
    try:
        raw_parts.append(f"host={socket.gethostname()}")
    except Exception:
        pass
    try:
        raw_parts.append(f"node={uuid.getnode():012x}")
    except Exception:
        pass
    try:
        raw_parts.append(f"platform={platform.platform()}")
    except Exception:
        pass
    if os.name == "nt":
        cmds = [
            ["wmic", "csproduct", "get", "UUID"],
            ["wmic", "bios", "get", "serialnumber"],
            ["wmic", "baseboard", "get", "serialnumber"],
        ]
        for cmd in cmds:
            try:
                cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="ignore", timeout=2)
                vals = [x.strip() for x in cp.stdout.splitlines() if x.strip() and "UUID" not in x.upper() and "SERIAL" not in x.upper()]
                if vals:
                    raw_parts.append(" ".join(vals[:2]))
            except Exception:
                pass
    raw = "|".join(raw_parts) or "unknown-machine"
    return hashlib.sha256((MACHINE_HASH_SALT + "|" + raw).encode("utf-8", errors="ignore")).hexdigest().upper()[:32]


def license_path() -> Path:
    return user_data_dir() / LICENSE_FILE_NAME


@dataclass
class LicenseStatus:
    valid: bool
    reason: str
    plan: str = "未授权"
    license_id: str = ""
    expires_at: str = ""
    days_left: Optional[int] = None
    payload: Optional[Dict[str, Any]] = None
    path: Path = license_path()

    def summary(self) -> str:
        if not self.valid:
            return f"未授权：{self.reason}"
        plan_map = {"day": "一天", "week": "一周", "month": "一个月", "permanent": "永久", "永久": "永久"}
        plan_cn = plan_map.get(str(self.plan).lower(), str(self.plan))
        if self.days_left is None:
            return f"已授权：{plan_cn}"
        return f"已授权：{plan_cn} / 剩余 {self.days_left} 天"


def verify_signature(payload: Dict[str, Any], sig: bytes) -> bool:
    if len(sig) != RSA_BYTES:
        return False
    m = pow(int.from_bytes(sig, "big"), PUBLIC_KEY_E, PUBLIC_KEY_N).to_bytes(RSA_BYTES, "big")
    # PKCS#1 v1.5: 00 01 FF..FF 00 DigestInfo(SHA256(payload))
    expected_tail = DIGESTINFO_SHA256_PREFIX + _sha256(canonical_payload(payload))
    if not (m.startswith(b"\x00\x01") and expected_tail == m[-len(expected_tail):]):
        return False
    sep = m.find(b"\x00", 2)
    return sep >= 10 and all(x == 0xFF for x in m[2:sep])


def parse_license_key(key_text: str) -> Tuple[Dict[str, Any], bytes]:
    s = "".join(key_text.strip().split())
    for prefix in ("VFG-", "V28-", "V27-"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if "." not in s:
        raise ValueError("卡密格式错误：缺少签名分隔符")
    p_b64, sig_b64 = s.split(".", 1)
    payload = json.loads(_b64u_decode(p_b64).decode("utf-8"))
    sig = _b64u_decode(sig_b64)
    if not isinstance(payload, dict):
        raise ValueError("卡密格式错误")
    return payload, sig


def validate_license_text(key_text: str, *, now: Optional[_dt.datetime] = None) -> LicenseStatus:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    try:
        payload, sig = parse_license_key(key_text)
    except Exception as e:
        return LicenseStatus(False, f"无法解析卡密：{e}")
    if payload.get("product") not in ACCEPTED_PRODUCT_IDS:
        return LicenseStatus(False, "产品不匹配", payload=payload)
    if not verify_signature(payload, sig):
        return LicenseStatus(False, "签名校验失败", payload=payload)
    hwid_hash = str(payload.get("hwid_hash") or "").upper().strip()
    if hwid_hash and hwid_hash != machine_code():
        return LicenseStatus(False, "机器码不匹配", payload=payload)
    expires_at = str(payload.get("expires_at") or "permanent")
    days_left: Optional[int] = None
    if expires_at.lower() not in {"permanent", "永久", "never"}:
        try:
            exp = _dt.datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=_dt.timezone.utc)
        except Exception:
            return LicenseStatus(False, "到期时间格式错误", payload=payload)
        if now > exp:
            return LicenseStatus(False, "卡密已过期", payload=payload, expires_at=expires_at)
        days_left = max(0, int((exp - now).total_seconds() // 86400) + 1)
    return LicenseStatus(True, "OK", plan=str(payload.get("plan") or "unknown"), license_id=str(payload.get("license_id") or ""), expires_at=expires_at, days_left=days_left, payload=payload)


def load_license() -> LicenseStatus:
    p = license_path()
    if not p.exists():
        return LicenseStatus(False, f"未找到卡密文件：{p}", path=p)
    try:
        return validate_license_text(p.read_text(encoding="utf-8"))
    except Exception as e:
        return LicenseStatus(False, f"读取卡密失败：{e}", path=p)


def save_license(key_text: str) -> LicenseStatus:
    status = validate_license_text(key_text)
    if not status.valid:
        return status
    p = license_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(key_text.strip() + "\n", encoding="utf-8")
    status.path = p
    return status
