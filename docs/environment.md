# Environment

The existing `thesis/` virtual environment is ignored by Git. Activate it with `source thesis/bin/activate`. Validation used Python 3.10, PyTorch 2.6.0, PyG 2.6.1, NumPy 1.26.4, and pandas 2.2.3.

For another machine, create an environment and install compatible PyTorch, PyG, and sampling extensions for its CPU/CUDA platform:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[report,sampling]"
python -m fedptha --help
```

The `sampling` extra provides `torch-scatter` and `torch-sparse`; another compatible PyG sampling backend also works. Platform-specific binaries may be needed. The `report` extra adds Matplotlib for headless figure generation.

`pyproject.toml` describes the installable package, dependencies, and `fedptha` command. Editable installation lets source edits take effect immediately. It does not freeze GPU drivers or CUDA builds. Installation was checked in the existing environment; a freshly downloaded environment and cluster/GPU execution were not tested.

SLURM uses an Apptainer image whose default name is `pytorch2.2.2-cuda11.8-ubuntu22.04-fraudGT_v2.sif`. Set `FEDPTHA_CONTAINER` to your actual image path. Its filename does not establish which packages are installed inside it.
