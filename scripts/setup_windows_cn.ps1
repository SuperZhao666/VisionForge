$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

function Step($Text) {
    Write-Host ""
    Write-Host $Text -ForegroundColor Cyan
}

function RunCmd($File, $Arguments) {
    Write-Host "> $File $Arguments" -ForegroundColor DarkGray
    $p = Start-Process -FilePath $File -ArgumentList $Arguments -NoNewWindow -Wait -PassThru
    if ($p.ExitCode -ne 0) {
        throw "Command failed with exit code $($p.ExitCode): $File $Arguments"
    }
}

Step "[1/8] Project root"
Write-Host $ProjectRoot

if (-not (Test-Path "requirements-cn.txt")) {
    throw "requirements-cn.txt not found. Run this script from the project root or use scripts\\setup_windows_cn.bat."
}

Step "[2/8] Check Python"
python --version
python -c "import sys; print('python executable:', sys.executable); assert sys.version_info >= (3,10), 'Python 3.10+ is required'"

Step "[3/8] Configure pip mirror"
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config set global.timeout 120
python -m pip config list

Step "[4/8] Install Python dependencies"
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn -r requirements-cn.txt

Step "[5/8] Create runtime directories"
New-Item -ItemType Directory -Force -Path vendor_models, outputs, samples, logs | Out-Null

Step "[6/8] Check NVIDIA driver"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi
} else {
    Write-Host "[WARN] nvidia-smi not found. Install NVIDIA driver first if this PC has an NVIDIA GPU." -ForegroundColor Yellow
}

Step "[7/8] Check Python imports"
python -c "import cv2, numpy, yaml, serial, keyboard, dxcam, mss, psutil; print('basic imports: OK')"
python -c "import onnxruntime as ort; print('onnxruntime:', ort.__version__); print('available providers:', ort.get_available_providers())"

Step "[8/8] Next steps"
Write-Host "Put the ONNX model here: vendor_models\\valorant_320_v11n.onnx" -ForegroundColor Green
Write-Host "Then run: scripts\\diagnose_windows.bat" -ForegroundColor Green
Write-Host "If CUDA provider fails, see docs\\DRIVER_CUDA_DOWNLOAD_LINKS.md or run scripts\\open_driver_cuda_links.bat" -ForegroundColor Yellow
