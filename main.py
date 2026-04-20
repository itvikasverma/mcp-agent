import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd):
    return subprocess.Popen(cmd, cwd=ROOT, shell=False)


def main():
    procs = []
    try:
        # 1) API
        procs.append(run([sys.executable, "-m", "uvicorn", "app.app:app", "--host", "127.0.0.1", "--port", "9000", "--reload"]))
        time.sleep(1)

        # 2) MCP Server (optional standalone; web also launches it internally)
        procs.append(run([sys.executable, "-m", "services.mcp_server"]))
        time.sleep(1)

        # 3) Web UI
        procs.append(run([sys.executable, "-m", "uvicorn", "web.web:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]))
        time.sleep(1)

        # 4) Crawler (one-shot)
        run([sys.executable, "-m", "services.crawler"])

        print("API running: http://127.0.0.1:9000")
        print("Web UI running: http://127.0.0.1:8000")
        print("Crawler started once.")
        print("Press Ctrl+C to stop.")

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
