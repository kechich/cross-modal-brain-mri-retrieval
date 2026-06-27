#!/usr/bin/env python3
"""Upload a local file to the AMD JupyterLab box via the Jupyter Contents API.

Usage:
    python tools/jupyter_put.py <local_path> [remote_name]

- .ipynb files are uploaded as notebooks, everything else as text files.
- Reads JUPYTER_HOST / JUPYTER_TOKEN from .env (or the environment).
- Files land in the Jupyter root (/shared-docker) and show up in the file browser.
"""
import json, os, sys, urllib.request, pathlib


def load_env():
    env = dict(os.environ)
    dotenv = pathlib.Path(__file__).resolve().parent.parent / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip())
    return env


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: jupyter_put.py <local_path> [remote_name]")
    local = pathlib.Path(sys.argv[1])
    remote = sys.argv[2] if len(sys.argv) > 2 else local.name

    env = load_env()
    host = env["JUPYTER_HOST"].rstrip("/")
    tok = env["JUPYTER_TOKEN"]

    if remote.endswith(".ipynb"):
        payload = {"type": "notebook", "format": "json",
                   "content": json.loads(local.read_text(encoding="utf-8"))}
    else:
        payload = {"type": "file", "format": "text",
                   "content": local.read_text(encoding="utf-8")}

    req = urllib.request.Request(f"{host}/api/contents/{remote}",
                                 data=json.dumps(payload).encode(), method="PUT")
    req.add_header("Authorization", f"token {tok}")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as r:
        print(f"{remote} -> HTTP {r.status}")


if __name__ == "__main__":
    main()
