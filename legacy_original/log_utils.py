import datetime

def log(msg: str, level: str = "INFO"):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    colors = {"INFO": "\033[97m", "ERROR": "\033[91m", "SUCCESS": "\033[92m", "WARN": "\033[93m"}
    reset = "\033[0m"
    print(f"{colors.get(level, '')}[{timestamp}] [{level}] {msg}{reset}")