#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn -r requirements-cn.txt
python - <<'PYCODE'
import onnxruntime as ort
print('onnxruntime:', ort.__version__)
print('providers:', ort.get_available_providers())
PYCODE
