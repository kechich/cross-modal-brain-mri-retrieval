# Claude ↔ AMD Jupyter Workflow

Fast iteration loop: Claude codes locally (captured by `entire`), uploads to the AMD MI300X box via REST API, you run in JupyterLab, Claude reads results back.

## The Interaction Loop

1. **You:** Prompt Claude in terminal (e.g., "run the baseline")
2. **Claude:** 
   - Writes/edits code locally (captured by `entire` for grading)
   - Uploads to Jupyter via `tools/jupyter_put.py`
   - Provides instructions for running in JupyterLab
3. **You:** Execute the notebook/script in JupyterLab (Run All)
4. **Claude:** Downloads results and reads them back
5. **Repeat:** Loop back to step 1

## Jupyter Box Details

| | |
|---|---|
| **URL** | `http://165.245.141.178/lab` |
| **File root** | `/shared-docker` (uploads land here; visible in file browser) |
| **Data** | `/workspace/data/ehl` (1454 `.nii` files + CSVs) |
| **Output** | `/workspace/out` (training checkpoints, submissions, logs) |
| **Env** | ROCm torch, Python 3.x, docker container |

## File Upload/Download

**Upload a file to Jupyter:**
```bash
python tools/jupyter_put.py <local_file> [<remote_path>]

# Examples:
python tools/jupyter_put.py kaggle_baseline.py                # lands in /shared-docker/
python tools/jupyter_put.py run_training.ipynb amine/training.ipynb  # lands in /shared-docker/amine/
```

**Manage files on server (no SSH needed):**
```bash
python tools/jupyter_fs.py ls [<dir>]       # list directory
python tools/jupyter_fs.py mkdir <dir>      # create directory
python tools/jupyter_fs.py mv <old> <new>   # move/rename
```

## Single-Source-of-Truth Scripts

All training/eval scripts auto-detect their environment:

- **AMD / Jupyter:** Script detects `/workspace/data/ehl` → uses it; outputs to `/workspace/out`
- **Kaggle:** Script detects Kaggle paths → uses `/kaggle/input` + `/kaggle/working`

Same code, different paths. No divergence.

## Example Workflow

**Running the baseline:**
1. You: *"Run the baseline on the AMD box"*
2. Claude:
   ```bash
   python tools/jupyter_put.py kaggle_baseline.py
   ```
   Then: *"Uploaded. In JupyterLab: open `kaggle_baseline.py` → Run All"*
3. You: Open in JupyterLab, Run All
4. Claude: Reads results from `/workspace/out/`, shows metrics
5. Iterate: edit code locally → upload → run → analyze

## Notebooks vs Scripts

- **`.py` scripts:** Upload → run in JupyterLab terminal or notebook cell (`!python file.py`)
- **`.ipynb` notebooks:** Upload → open in JupyterLab → Run All

Claude handles both; you pick based on what's easier to run/monitor.
