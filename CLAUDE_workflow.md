# Claude ↔ AMD Jupyter Workflow

Fast iteration loop: Claude codes locally (captured by `entire`), uploads to the AMD MI300X box via REST API, you run in JupyterLab, Claude reads results back.

## The Interaction Loop

1. **You:** Prompt Claude in terminal (e.g., "run the baseline")
2. **Claude:** 
   - Writes/edits code locally (captured by `entire` for grading)
   - Uploads to Jupyter via `tools/jupyter_api.py`
   - Provides instructions for running in JupyterLab
3. **You:** Execute the notebook/script in JupyterLab (Run All)
4. **Claude:** Downloads results and reads them back into our conversation
5. **Repeat:** Loop back to step 1

## Jupyter Box Details

| | |
|---|---|
| **URL** | `http://165.245.141.178/lab` (Jupyter token in credentials) |
| **File root** | `/shared-docker` (Claude uploads here; visible in file browser) |
| **Data** | `/workspace/data/ehl` (1454 `.nii` files + CSVs) |
| **Output** | `/workspace/out` (training checkpoints, submissions, logs) |
| **Environment** | ROCm torch (AMD GPU support), Python 3.x, docker container |

## File Upload/Download

**Upload (Claude does this):**
```bash
python tools/jupyter_api.py upload <local_file> [<jupyter_path>]
# Examples:
python tools/jupyter_api.py upload kaggle_baseline.py              # → /shared-docker/kaggle_baseline.py
python tools/jupyter_api.py upload run_training.ipynb notebooks/  # → /shared-docker/notebooks/run_training.ipynb
```

**Download (Claude does this):**
```bash
python tools/jupyter_api.py download <jupyter_file> [<local_path>]
# Example:
python tools/jupyter_api.py download /workspace/out/submission.csv  # → ./submission.csv
```

**List files (Claude does this):**
```bash
python tools/jupyter_api.py ls [<jupyter_path>]
# Example:
python tools/jupyter_api.py ls  # List /shared-docker
python tools/jupyter_api.py ls /workspace/out  # List outputs
```

## Single-Source-of-Truth Scripts

All training/eval scripts auto-detect their environment and pick paths accordingly:

- **AMD / Jupyter run:** Script detects `/workspace/data/ehl` exists → uses it; outputs to `/workspace/out`
- **Kaggle run:** Script detects Kaggle paths → uses `/kaggle/input` + `/kaggle/working`

This means **no divergence** — same code, different environment detection.

### Example: `kaggle_baseline.py`
- Self-contained, one file
- On Jupyter: uses `/workspace/data/ehl`
- On Kaggle: uses `/kaggle/input`, auto-pip-installs MONAI
- Writes `submission.csv` to current working directory

## Workflow in Practice

**Session start:**
1. You: *"Let's run the baseline and see where we are."*
2. Claude: 
   - Checks `kaggle_baseline.py`
   - Uploads it to Jupyter
   - Says: *"Uploaded to `/shared-docker/kaggle_baseline.py`. In JupyterLab: File → Open → kaggle_baseline.py → Run All"*
3. You: Open notebook in JupyterLab, click Run All, wait for results
4. Claude: Downloads `/workspace/out/submission.csv` + logs, reads them, shows metrics
5. You: *"Now add deformation augmentation to the training loop"*
6. Claude: Edits `kaggle_baseline.py`, uploads new version, you run again

## Notebooks vs Scripts

- **Scripts** (`.py`): Upload, then run in JupyterLab's terminal or as a notebook cell (`!python <script>`)
- **Notebooks** (`.ipynb`): Upload, then open in JupyterLab and Run All

Claude can generate either; you pick based on what's easier to run/monitor.
