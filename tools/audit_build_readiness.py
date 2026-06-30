from __future__ import annotations

import ast
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "app_gui.py",
    "main.py",
    "src/app_paths.py",
    "src/offline_license.py",
    "tools/env_diagnostics.py",
    "tools/config_tuner_gui.py",
    "tools/license_keygen.py",
]


def compile_check() -> None:
    for rel in FILES:
        p = ROOT / rel
        if p.exists():
            py_compile.compile(str(p), doraise=True)


def class_method_names(tree: ast.AST, class_name: str) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for body in node.body:
                if isinstance(body, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.add(body.name)
    return out


def assigned_self_attrs(tree: ast.AST, class_name: str) -> set[str]:
    out: set[str] = set()
    target_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            target_class = node
            break
    if target_class is None:
        return out
    for node in ast.walk(target_class):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                out.add(t.attr)
    return out


def collision_check() -> None:
    tree = ast.parse((ROOT / "app_gui.py").read_text(encoding="utf-8"))
    methods = class_method_names(tree, "VisionForgeApp")
    attrs = assigned_self_attrs(tree, "VisionForgeApp")
    collisions = sorted(methods & attrs)
    if collisions:
        raise SystemExit(f"DesktopApp self 属性/方法命名冲突: {collisions}")


def required_assets_check() -> None:
    required = [
        ROOT / "config.default_v17_8_30.yaml",
        ROOT / "config.yaml",
        ROOT / "vendor_models" / "valorant_320_v11n.onnx",
        ROOT / "assets" / "app_icon.ico",
        ROOT / "assets" / "app_icon.png",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("缺少构建资源:\n" + "\n".join(missing))


def main() -> int:
    compile_check()
    collision_check()
    required_assets_check()
    print("VISIONFORGE_AUDIT_OK compile=OK self_collision=OK assets=OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
