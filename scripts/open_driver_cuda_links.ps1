$ErrorActionPreference = "Stop"
$links = @(
    "https://www.nvidia.com/en-us/software/nvidia-app/",
    "https://www.nvidia.com/en-us/drivers/",
    "https://developer.nvidia.com/cuda-12-8-0-download-archive",
    "https://developer.nvidia.com/cudnn-downloads",
    "https://docs.nvidia.com/deeplearning/cudnn/installation/latest/windows.html",
    "https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170"
)
foreach ($link in $links) {
    Write-Host "Opening: $link"
    Start-Process $link
}
