#!/usr/bin/env python3
"""
Jupyter API client for uploading/downloading files and managing the AMD MI300X box.
Uses the Jupyter REST API (no SSH needed).

Usage:
  python tools/jupyter_api.py upload <local_file> [<jupyter_path>]
  python tools/jupyter_api.py download <jupyter_file> [<local_path>]
  python tools/jupyter_api.py ls [<jupyter_path>]
"""

import os
import sys
import json
import base64
import requests
from pathlib import Path
from urllib.parse import urljoin

# Jupyter instance config
JUPYTER_HOST = os.getenv("JUPYTER_HOST", "165.245.141.178")
JUPYTER_TOKEN = os.getenv("JUPYTER_TOKEN", "OiHQAaQg0bTJHa6MmNb46XwR/gyNatO+POHw3HIII8do62B2R")
JUPYTER_ROOT = "/shared-docker"  # Files land here in Jupyter

JUPYTER_URL = f"http://{JUPYTER_HOST}"
API_URL = urljoin(JUPYTER_URL, "/api/contents")

session = requests.Session()
session.headers.update({"Authorization": f"token {JUPYTER_TOKEN}"})


def upload(local_file: str, jupyter_path: str = None) -> bool:
    """Upload a local file to Jupyter."""
    local_path = Path(local_file)
    if not local_path.exists():
        print(f"❌ Local file not found: {local_file}")
        return False

    if jupyter_path is None:
        jupyter_path = local_path.name

    # Make it relative to JUPYTER_ROOT
    if not jupyter_path.startswith("/"):
        jupyter_path = f"{JUPYTER_ROOT}/{jupyter_path}"

    # Read file and encode
    with open(local_path, "rb") as f:
        content = f.read()

    # For text files, encode as utf-8 string; for binary, base64
    try:
        content_str = content.decode("utf-8")
        file_format = "text"
    except UnicodeDecodeError:
        content_str = base64.b64encode(content).decode("utf-8")
        file_format = "base64"

    payload = {
        "type": "file",
        "name": Path(jupyter_path).name,
        "format": file_format,
        "content": content_str,
    }

    # Create parent dirs if needed
    parent = str(Path(jupyter_path).parent)
    _ensure_dir(parent)

    # Upload
    url = urljoin(API_URL, jupyter_path.lstrip("/"))
    resp = session.put(url, json=payload)

    if resp.status_code in (200, 201):
        print(f"✅ Uploaded {local_path.name} → {jupyter_path}")
        return True
    else:
        print(f"❌ Upload failed ({resp.status_code}): {resp.text}")
        return False


def download(jupyter_file: str, local_path: str = None) -> bool:
    """Download a file from Jupyter."""
    if not jupyter_file.startswith("/"):
        jupyter_file = f"{JUPYTER_ROOT}/{jupyter_file}"

    if local_path is None:
        local_path = Path(jupyter_file).name

    url = urljoin(API_URL, jupyter_file.lstrip("/"))
    resp = session.get(url)

    if resp.status_code != 200:
        print(f"❌ Download failed ({resp.status_code}): {resp.text}")
        return False

    data = resp.json()
    content = data.get("content", "")

    # Decode if base64
    if data.get("format") == "base64":
        content = base64.b64decode(content)
        mode = "wb"
    else:
        mode = "w"

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, mode) as f:
        f.write(content)

    print(f"✅ Downloaded {jupyter_file} → {local_path}")
    return True


def ls(jupyter_path: str = None) -> bool:
    """List files in Jupyter."""
    if jupyter_path is None:
        jupyter_path = JUPYTER_ROOT

    if not jupyter_path.startswith("/"):
        jupyter_path = f"{JUPYTER_ROOT}/{jupyter_path}"

    url = urljoin(API_URL, jupyter_path.lstrip("/"))
    resp = session.get(url)

    if resp.status_code != 200:
        print(f"❌ List failed ({resp.status_code}): {resp.text}")
        return False

    data = resp.json()
    if data.get("type") != "directory":
        print(f"❌ {jupyter_path} is not a directory")
        return False

    print(f"\n📁 {jupyter_path}:\n")
    for entry in sorted(data.get("content", []), key=lambda x: x["name"]):
        typ = "📄" if entry["type"] == "file" else "📁"
        size = f" ({entry['size']} B)" if entry["type"] == "file" else ""
        print(f"  {typ} {entry['name']}{size}")
    return True


def _ensure_dir(path: str) -> None:
    """Create a directory in Jupyter if it doesn't exist."""
    if not path.startswith("/"):
        path = f"{JUPYTER_ROOT}/{path}"

    url = urljoin(API_URL, path.lstrip("/"))
    resp = session.get(url)
    if resp.status_code == 200:
        return  # Already exists

    # Create
    payload = {"type": "directory", "name": Path(path).name}
    parent = str(Path(path).parent)
    _ensure_dir(parent)  # Recursive parent creation
    resp = session.put(url, json=payload)
    if resp.status_code not in (200, 201):
        print(f"⚠️ Failed to create {path}: {resp.text}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "upload":
        if len(sys.argv) < 3:
            print("Usage: python jupyter_api.py upload <local_file> [<jupyter_path>]")
            sys.exit(1)
        local_file = sys.argv[2]
        jupyter_path = sys.argv[3] if len(sys.argv) > 3 else None
        upload(local_file, jupyter_path)

    elif cmd == "download":
        if len(sys.argv) < 3:
            print("Usage: python jupyter_api.py download <jupyter_file> [<local_path>]")
            sys.exit(1)
        jupyter_file = sys.argv[2]
        local_path = sys.argv[3] if len(sys.argv) > 3 else None
        download(jupyter_file, local_path)

    elif cmd == "ls":
        jupyter_path = sys.argv[2] if len(sys.argv) > 2 else None
        ls(jupyter_path)

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
