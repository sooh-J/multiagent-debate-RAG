import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


class Tee:
    """stdout을 터미널과 파일에 동시에 출력. 로그 파일명에 실행 시각 포함."""

    def __init__(self, prefix: str = "run"):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.filepath = LOG_DIR / f"{prefix}_{timestamp}.log"
        self.terminal = sys.stdout
        self.file = open(self.filepath, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.terminal
