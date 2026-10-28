# Environment setup

Use Python **3.10–3.12**. The reference environment uses PyTorch **2.6.0**, PyG **2.6.1**, NumPy **1.26.4**, and pandas **2.2.3**. Run the following commands from the repository root after creating and activating a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## Linux CPU

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install pyg_lib torch_scatter torch_sparse --only-binary=:all: \
  -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
python -m pip install -e ".[report]"
```

## macOS CPU

```bash
python -m pip install torch==2.6.0
python -m pip install pyg_lib torch_scatter torch_sparse --only-binary=:all: \
  -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
python -m pip install -e ".[report]"
```

These examples use the official [PyTorch 2.6.0 packages](https://pytorch.org/get-started/previous-versions/#v260) and [PyG CPU wheel index](https://data.pyg.org/whl/torch-2.6.0+cpu.html). Wheel availability depends on Python version, operating system, and architecture; see the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html) if no matching binary is available.

## Linux with CUDA 12.4

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install pyg_lib torch_scatter torch_sparse --only-binary=:all: \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
python -m pip install -e ".[report]"
```

Choose a CUDA build compatible with your NVIDIA driver using the [PyTorch version guide](https://pytorch.org/get-started/previous-versions/#v260), and use the same build in the [PyG extension URL](https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html). Pass `--device cuda` to training to select the GPU.

## Check the installation

```bash
python -c "import torch, torch_geometric, torch_sparse; print(torch.__version__, torch_geometric.__version__)"
python -m fedptha --help
python -m fedptha train --method fedptha --benchmark five_client --dry-run
```

The editable installation picks up changes under `src/` immediately. The `report` extra installs Matplotlib for figures. `requirements.txt` is an alternative to the final package-install command after the PyTorch and sampling binaries are installed.
