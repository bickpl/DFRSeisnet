import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.RUNet import RUNet


def load_trained_model(ckpt_path: str, device: torch.device) -> RUNet:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    resto_weights = {
        k.replace("restoration_net.", ""): v
        for k, v in state_dict.items()
        if k.startswith("restoration_net.")
    }

    model = RUNet(
        inp_channels=2,
        out_channels=1,
        dim=64,
        patch_size=4,
        window_size=8,
        use_checkpoint=False,
    ).to(device)
    model.load_state_dict(resto_weights, strict=True)
    model.eval()
    return model


@torch.no_grad()
def denoise_full_image(
    model: RUNet,
    noisy_np: np.ndarray,
    device: torch.device,
    downsample_size=(256, 256),
    patch_size=(256, 256),
    stride=(256, 256),
) -> np.ndarray:
    if noisy_np.ndim != 2:
        raise ValueError(f"Expected noisy array shape (H, W), got {noisy_np.shape}")

    noisy = torch.from_numpy(noisy_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    h, w = noisy.shape[-2:]

    noisy_small = F.interpolate(noisy, size=downsample_size, mode="bilinear", align_corners=False)
    noisy_small = noisy_small.repeat(1, 2, 1, 1)
    prior_small, _ = model(noisy_small)
    prior_full = F.interpolate(prior_small, size=(h, w), mode="bilinear", align_corners=False)

    ph, pw = patch_size
    sh, sw = stride

    out = torch.zeros_like(noisy)
    weight = torch.zeros_like(noisy)

    for top in range(0, h, sh):
        for left in range(0, w, sw):
            bottom = min(top + ph, h)
            right = min(left + pw, w)

            noisy_patch = noisy[:, :, top:bottom, left:right]
            prior_patch = prior_full[:, :, top:bottom, left:right]

            if noisy_patch.shape[-2:] != (ph, pw):
                padded_noisy = torch.zeros((1, 1, ph, pw), dtype=noisy.dtype, device=device)
                padded_prior = torch.zeros((1, 1, ph, pw), dtype=noisy.dtype, device=device)
                padded_noisy[:, :, : noisy_patch.shape[-2], : noisy_patch.shape[-1]] = noisy_patch
                padded_prior[:, :, : prior_patch.shape[-2], : prior_patch.shape[-1]] = prior_patch
                noisy_patch = padded_noisy
                prior_patch = padded_prior

            local_input = torch.cat([noisy_patch, prior_patch], dim=1)
            denoised_patch, _ = model(local_input)

            valid_h = bottom - top
            valid_w = right - left
            out[:, :, top:bottom, left:right] += denoised_patch[:, :, :valid_h, :valid_w]
            weight[:, :, top:bottom, left:right] += 1.0

    out = out / torch.clamp(weight, min=1.0)
    return out.squeeze(0).squeeze(0).cpu().numpy()


def main():
    ckpt_path = ROOT / "pt" / "example_pt.ckpt"
    noisy_path = ROOT / "example_data" / "example_noisy.npy"
    out_path = ROOT / "example_data" / "example_denoised.npy"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(str(ckpt_path), device)

    noisy_np = np.load(str(noisy_path))
    denoised_np = denoise_full_image(model, noisy_np, device=device)
    np.save(str(out_path), denoised_np.astype(np.float32))

    print(f"Device: {device}")
    print(f"Noisy input: {noisy_np.shape}, dtype={noisy_np.dtype}")
    print(f"Denoised output saved to: {out_path}")
    print(f"Denoised output: shape={denoised_np.shape}, dtype={denoised_np.dtype}")


if __name__ == "__main__":
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    main()
