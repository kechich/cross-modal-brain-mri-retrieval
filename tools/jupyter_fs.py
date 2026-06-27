#!/usr/bin/env python3
"""Manage files on the AMD JupyterLab box via the Jupyter Contents API (HTTP, no SSH).

Usage:
    python tools/jupyter_fs.py ls   [remote_dir]          # list a directory (default root)
    python tools/jupyter_fs.py mkdir <remote_dir>         # create a directory
    python tools/jupyter_fs.py mv   <old_path> <new_path> # move/rename server-side

Reads JUPYTER_HOST / JUPYTER_TOKEN from .env (same as jupyter_put.py).
Paths are relative to the Jupyter root (/shared-docker). Symlinks can't be made over
this API — do those in a notebook cell with os.symlink (see run_eval.ipynb).
"""
import json, os, sys, urllib.request, urllib.error, pathlib


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


ENV = load_env()
HOST = ENV["JUPYTER_HOST"].rstrip("/")
TOK = ENV["JUPYTER_TOKEN"]


def call(method, path, body=None):
    req = urllib.request.Request(f"{HOST}/api/contents/{path}",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 method=method)
    req.add_header("Authorization", f"token {TOK}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
            return r.status, (json.loads(data) if data else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    if cmd == "ls":
        path = sys.argv[2] if len(sys.argv) > 2 else ""
        status, body = call("GET", path)
        if status >= 400:
            sys.exit(f"ls {path!r} -> HTTP {status}: {body.get('error')}")
        for item in sorted(body.get("content", []), key=lambda c: (c["type"] != "directory", c["name"])):
            tag = "d" if item["type"] == "directory" else "-"
            print(f"  {tag} {item['name']}")
    elif cmd == "mkdir":
        path = sys.argv[2]
        status, body = call("PUT", path, {"type": "directory"})
        print(f"mkdir {path} -> HTTP {status}{'' if status < 400 else ': ' + str(body.get('error'))}")
    elif cmd == "mv":
        old, new = sys.argv[2], sys.argv[3]
        status, body = call("PATCH", old, {"path": new})
        print(f"mv {old} -> {new} : HTTP {status}{'' if status < 400 else ': ' + str(body.get('error'))}")
    else:
        sys.exit(f"unknown command {cmd!r}\n{__doc__}")


if __name__ == "__main__":
    main()
