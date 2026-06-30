from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import secrets
from pathlib import Path
from typing import Any, Dict

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_KEY_PATH = ROOT / "owner_secrets" / "license_private_key.pem"
PUBLIC_MODULE_PATH = ROOT / "src" / "offline_license.py"
PRODUCT_ID = "VISIONFORGE"


def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def canonical(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load_private_key(path: Path = PRIVATE_KEY_PATH):
    return serialization.load_pem_private_key(path.read_bytes(), password=None)


def init_keys(force: bool = False) -> None:
    PRIVATE_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PRIVATE_KEY_PATH.exists() and not force:
        raise SystemExit(f"私钥已存在：{PRIVATE_KEY_PATH}\n如需重置，添加 --force。注意：重置后旧卡密全部失效。")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    PRIVATE_KEY_PATH.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    pub = key.public_key().public_numbers()
    text = PUBLIC_MODULE_PATH.read_text(encoding="utf-8")
    import re
    text = re.sub(r"PUBLIC_KEY_N = int\(\"\d+\"\)", f'PUBLIC_KEY_N = int("{pub.n}")', text)
    text = re.sub(r"PUBLIC_KEY_E = \d+", f"PUBLIC_KEY_E = {pub.e}", text)
    PUBLIC_MODULE_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(f"[OK] 新私钥：{PRIVATE_KEY_PATH}")
    print(f"[OK] 已写入公钥：{PUBLIC_MODULE_PATH}")
    print("[WARN] 私钥只允许保存在你自己的机器，不要发给用户，不要打包进 EXE。")


def make_payload(plan: str, days: int | None, hwid_hash: str | None, note: str = "") -> Dict[str, Any]:
    now = dt.datetime.now(dt.timezone.utc)
    if days is None:
        expires = "permanent"
    else:
        expires = (now + dt.timedelta(days=int(days))).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "v": 1,
        "product": PRODUCT_ID,
        "license_id": secrets.token_hex(8).upper(),
        "plan": plan,
        "issued_at": now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "expires_at": expires,
        "hwid_hash": (hwid_hash or "").strip().upper(),
        "note": note,
    }


def sign_payload(payload: Dict[str, Any]) -> str:
    key = load_private_key()
    sig = key.sign(canonical(payload), padding.PKCS1v15(), hashes.SHA256())
    return "VFG-" + b64u(canonical(payload)) + "." + b64u(sig)


def main() -> int:
    ap = argparse.ArgumentParser(description="VisionForge 离线卡密生成工具。不要把 owner_secrets 私钥发给用户。")
    ap.add_argument("--init-keys", action="store_true", help="生成新的 RSA 私钥并写入程序公钥。会使旧卡密失效。")
    ap.add_argument("--force", action="store_true", help="允许覆盖已有私钥。")
    ap.add_argument("--plan", choices=["day", "week", "month", "permanent"], default="day", help="卡密类型")
    ap.add_argument("--hwid", default="", help="可选：绑定用户 GUI 授权页显示的机器码。留空则不绑定机器。")
    ap.add_argument("--note", default="", help="可选备注")
    ap.add_argument("--out", default="", help="输出到文件；留空打印到控制台")
    args = ap.parse_args()
    if args.init_keys:
        init_keys(force=args.force)
        return 0
    plan_days = {"day": 1, "week": 7, "month": 31, "permanent": None}
    payload = make_payload(args.plan, plan_days[args.plan], args.hwid, args.note)
    key_text = sign_payload(payload)
    if args.out:
        Path(args.out).write_text(key_text + "\n", encoding="utf-8")
        print(f"[OK] 已写入：{args.out}")
    else:
        print(key_text)
    print("\n载荷：")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
