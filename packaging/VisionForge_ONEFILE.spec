# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

block_cipher = None
ROOT = Path.cwd()

datas = [
    (str(ROOT / 'config.default_v17_8_32.yaml'), '.'),
    (str(ROOT / 'config.yaml'), '.'),
    (str(ROOT / 'vendor_models' / 'valorant_320_v11n.onnx'), 'vendor_models'),
    (str(ROOT / 'assets' / 'app_icon.ico'), 'assets'),
    (str(ROOT / 'assets' / 'app_icon.png'), 'assets'),
]
for folder in ['docs', 'runtime_dlls']:
    p = ROOT / folder
    if p.exists():
        for fp in p.rglob('*'):
            if fp.is_file():
                datas.append((str(fp), str(Path(folder) / fp.relative_to(p).parent)))
try:
    datas += collect_data_files('customtkinter')
except Exception:
    pass

binaries = []
for pkg in ['onnxruntime', 'cv2', 'numpy']:
    try:
        binaries += collect_dynamic_libs(pkg)
    except Exception:
        pass

hiddenimports = ['src.offline_license', 'src.app_paths', 'tools.env_diagnostics', 'tools.config_tuner_gui']
for pkg in ['onnxruntime', 'cv2', 'numpy', 'yaml', 'serial', 'keyboard', 'dxcam', 'mss', 'psutil', 'requests', 'customtkinter']:
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        hiddenimports.append(pkg)

a = Analysis(
    ['app_gui.py'],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'scipy', 'pandas', 'torch', 'tensorflow'],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='VisionForge',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / 'assets' / 'app_icon.ico'),
)
