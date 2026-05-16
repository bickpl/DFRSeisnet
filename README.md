# DFRSeisnet

DFRSeisnet is a seismic denoising framework based on a dual-stage restoration strategy:
1. Estimate a global prior on a downsampled input.
2. Use that prior to guide patch-wise local restoration on the full-resolution data.

This repository includes:
- A pretrained checkpoint and a runnable denoising demo.
- Training code built with PyTorch Lightning.

## Repository Structure

- `models/RUNet.py`: core restoration backbone (RUNet).
- `train_code/net.py`: full training model (global prior + local refinement + uncertainty weighting).
- `train_code/data_modul.py`: dataset and dataloaders.
- `train_code/main.py`: training entry script.
- `test/test_demo.py`: inference/demo script for full-size denoising.
- `example_data/example_noisy.npy`: sample noisy input.
- `example_data/example_denoised.npy`: output written by demo script.
- `pt/example_pt.ckpt`: example pretrained checkpoint.

## Environment Setup

1. Create and activate a Python environment (recommended: Python 3.10+).
2. Install dependencies:

```bash
pip install -r requirementlist.txt
```

3. Install a CUDA-enabled PyTorch build if you plan to use GPU. See the official PyTorch install page for your CUDA version.

## Quick Start: Denoise a Seismic Section

Run the demo script from repository root:

```bash
python test/test_demo.py
```

What the script does:
- Loads `pt/example_pt.ckpt`.
- Reads `example_data/example_noisy.npy` (expected shape: `(H, W)`).
- Performs global-prior + patch-wise denoising.
- Saves output to `example_data/example_denoised.npy`.

Console output includes device and tensor shapes.

## Inference Input/Output Format

- Input: `float32` NumPy array saved in `.npy`, shape `(H, W)`.
- Output: `float32` NumPy array `.npy`, shape `(H, W)`.

If you want to denoise your own file, update paths in `test/test_demo.py`:
- `ckpt_path`
- `noisy_path`
- `out_path`

You can also tune tiling behavior in `denoise_full_image(...)`:
- `downsample_size`
- `patch_size`
- `stride`

## Training

Training entry:

```bash
python train_code/main.py
```

Before running, edit dataset paths in `train_code/main.py`:
- `train_dir`
- `val_dir`

Expected dataset layout:

```text
<train_dir>/
  noisy/*.npy
  clean/*.npy

<val_dir>/
  noisy/*.npy
  clean/*.npy
```

Notes:
- Each noisy sample should have a matching clean sample with the same filename under the `clean` folder.
- Checkpoints and logs are saved under `./DRUNet` by default.

## Reproducibility Notes

- Inference script runs on GPU if available, otherwise CPU.
- Mixed precision is enabled in training (`precision="16-mixed"`).
- Patch-based inference uses zero-padding at image borders and averages overlaps.

## License

This project is released under the MIT License. See `LICENSE`.
