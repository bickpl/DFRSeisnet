import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from models.RUNet import RUNet
class SUAM(nn.Module):

    def __init__(
            self,
            in_channels,
            hidden_dim=64,
            num_tasks=2
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            hidden_dim,
            3,
            padding=1
        )

        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            3,
            padding=1
        )

        self.act2 = nn.GELU()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tasks)
        )

    def forward(
            self,
            global_feature,
            local_feature
    ):

        # feature fusion
        x = torch.cat(
            [global_feature, local_feature],
            dim=1
        )

        x = self.act1(
            self.conv1(x)
        )

        x = self.act2(
            self.conv2(x)
        )

        x = self.pool(x).flatten(1)

        # [B, num_tasks]
        log_vars = self.fc(x)

        # batch-level task uncertainty
        log_vars = log_vars.mean(dim=0)


        return log_vars


# =========================================================
# TUR Loss
# =========================================================
class TURLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
            self,
            losses,
            log_vars
    ):

        total_loss = 0
        weights = []

        for i, loss in enumerate(losses):

            precision = torch.exp(
                -log_vars[i]
            )

            total_loss += (
                    precision * loss
                    + log_vars[i]
            )

            weights.append(
                precision.detach()
            )

        return total_loss, weights

# =========================================================
# Main Network
# =========================================================
class Net(L.LightningModule):

    def __init__(
            self,
            lr=1e-4,
            save_val_debug=True
    ):
        super().__init__()

        # =========================================================
        # __init__
        # =========================================================

        self.restoration_net = RUNet(
            inp_channels=2,
            out_channels=1,
            dim=64,
            patch_size=4,
            window_size=8,
            use_checkpoint=False
        )

        # shared UEM
        self.suam = SUAM(
            in_channels=128,
            hidden_dim=64,
            num_tasks=2
        )

        self.loss_rec = nn.MSELoss()

        self.total_loss = TURLoss()

        self.lr = lr
        self.save_val_debug = save_val_debug

    # =====================================================
    # Slice helper
    # =====================================================
    @staticmethod
    def _slice_from_positions(
            feature_map,
            patch_hw,
            patch_pos
    ):
        ph, pw = patch_hw

        patches = []

        for i in range(feature_map.shape[0]):

            top = int(patch_pos[i, 0].item())
            left = int(patch_pos[i, 1].item())

            cur = feature_map[i:i + 1]

            _, _, h, w = cur.shape

            bottom = min(top + ph, h)
            right = min(left + pw, w)

            crop = cur[:, :, top:bottom, left:right]

            out = torch.zeros(
                (1, cur.shape[1], ph, pw),
                dtype=cur.dtype,
                device=cur.device
            )

            out[:, :, :crop.shape[2], :crop.shape[3]] = crop

            patches.append(out)

        return torch.cat(patches, dim=0)

    # =====================================================
    # Forward
    # =====================================================
    def forward(
            self,
            clean_small,
            noisy_small,
            noisy_patch,
            patch_pos,
            target_hw
    ):

        # ---------------------------------------------
        # global prior
        # ---------------------------------------------
        prior_small, prior_feature = self.restoration_net(
            noisy_small
        )

        prior_full = F.interpolate(
            prior_small,
            size=target_hw,
            mode='bilinear',
            align_corners=False
        )

        prior_patch = self._slice_from_positions(
            prior_full,
            noisy_patch.shape[-2:],
            patch_pos
        )

        # ---------------------------------------------
        # local restoration
        # ---------------------------------------------
        local_input = torch.cat(
            [noisy_patch, prior_patch],
            dim=1
        )

        denoised_patch, denoised_feature = self.restoration_net(
            local_input
        )

        return {
            'prior_small': prior_small,
            'prior_feature': prior_feature,
            'prior_full': prior_full,
            'prior_patch': prior_patch,
            'denoised_patch': denoised_patch,
            'denoised_feature': denoised_feature,
        }

    # =========================================================
    # _shared_step
    # =========================================================

    def _shared_step(self, batch):

        target_hw = (
            int(batch['patch_pos'][0, 2].item()),
            int(batch['patch_pos'][0, 3].item()),
        )

        output = self.forward(
            clean_small=batch['clean_small'],
            noisy_small=batch['noisy_small'],
            noisy_patch=batch['noisy_patch'],
            patch_pos=batch['patch_pos'],
            target_hw=target_hw,
        )

        # -------------------------------------------------
        # losses
        # -------------------------------------------------
        loss_global = self.loss_rec(
            output['prior_small'],
            batch['clean_small']
        )

        loss_local = self.loss_rec(
            output['denoised_patch'],
            batch['clean_patch']
        )

        # -------------------------------------------------
        # UEM
        # -------------------------------------------------
        log_vars = self.uem(
            output['prior_feature'],
            output['denoised_feature']
        )

        log_var_global = log_vars[0]
        log_var_local = log_vars[1]

        # -------------------------------------------------
        # TUR
        # -------------------------------------------------
        loss, weights = self.total_loss(
            [loss_global, loss_local],
            log_vars
        )

        weight_global = weights[0]
        weight_local = weights[1]

        return (
            output,
            loss,
            loss_global,
            loss_local,
            weight_global,
            weight_local,
            log_var_global,
            log_var_local
        )

    # =====================================================
    # Validation visualization
    # =====================================================
    def _save_val_debug_tensors(
            self,
            batch,
            output,
            batch_idx
    ):

        if not self.save_val_debug:
            return

        if batch_idx != 0:
            return

        save_dir = os.path.join(
            self.trainer.default_root_dir,
            "val_debug"
        )

        os.makedirs(save_dir, exist_ok=True)

        epoch_tag = f"epoch_{int(self.current_epoch):04d}"

        constraint_out_small = output['prior_small'][0, 0].detach().cpu().numpy().astype(np.float32)

        upsampled_then_slice = output['prior_patch'][0, 0].detach().cpu().numpy().astype(np.float32)

        noisy_slice = batch['noisy_patch'][0, 0].detach().cpu().numpy().astype(np.float32)

        clean_slice = batch['clean_patch'][0, 0].detach().cpu().numpy().astype(np.float32)

        denoised_slice = output['denoised_patch'][0, 0].detach().cpu().numpy().astype(np.float32)

        stacked = np.concatenate(
            [
                constraint_out_small.reshape(-1),
                upsampled_then_slice.reshape(-1),
                noisy_slice.reshape(-1),
                clean_slice.reshape(-1),
                denoised_slice.reshape(-1),
            ]
        )

        vmin = float(np.percentile(stacked, 1))
        vmax = float(np.percentile(stacked, 99))

        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin == vmax):

            vmin = float(stacked.min())
            vmax = float(stacked.max())

            if vmin == vmax:
                vmin -= 1e-6
                vmax += 1e-6

        def _save_png(img, filename):

            plt.figure(figsize=(6, 4))

            plt.imshow(
                img,
                cmap=plt.cm.seismic,
                vmin=vmin,
                vmax=vmax,
                aspect='auto'
            )

            plt.colorbar(
                fraction=0.046,
                pad=0.04
            )

            plt.axis("off")

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    save_dir,
                    f"{epoch_tag}_{filename}.png"
                ),
                dpi=200,
                bbox_inches="tight"
            )

            plt.close()

        _save_png(constraint_out_small, "constraint_out_small")
        _save_png(upsampled_then_slice, "upsampled_then_slice")
        _save_png(noisy_slice, "noisy_slice")
        _save_png(clean_slice, "clean_slice")
        _save_png(denoised_slice, "denoised_slice")

    # =========================================================
    # training_step
    # =========================================================

    def training_step(self, batch, batch_idx):

        (
            _,
            loss,
            loss_global,
            loss_local,
            weight_global,
            weight_local,
            log_var_global,
            log_var_local
        ) = self._shared_step(batch)

        self.log_dict(
            {
                'train_loss': loss,

                'train_loss_global': loss_global,
                'train_loss_local': loss_local,

                'weight_global': weight_global,
                'weight_local': weight_local,

                'log_var_global': log_var_global,
                'log_var_local': log_var_local,
            },
            prog_bar=True,
        )

        return loss

    # =========================================================
    # validation_step
    # =========================================================

    def validation_step(self, batch, batch_idx):

        (
            output,
            loss,
            loss_global,
            loss_local,
            weight_global,
            weight_local,
            log_var_global,
            log_var_local
        ) = self._shared_step(batch)

        self._save_val_debug_tensors(
            batch,
            output,
            batch_idx
        )

        pred = output['denoised_patch'][0][0].detach().cpu().numpy()

        gt = batch['clean_patch'][0][0].detach().cpu().numpy()

        self.log_dict(
            {
                'val_loss': loss,

                'val_loss_global': loss_global,
                'val_loss_local': loss_local,

                'weight_global': weight_global,
                'weight_local': weight_local,

                'log_var_global': log_var_global,
                'log_var_local': log_var_local,

                'psnr': psnr(
                    gt,
                    pred,
                    data_range=2
                ),

                'ssim': ssim(
                    gt,
                    pred,
                    data_range=2
                ),
            },
            prog_bar=True,
        )

    # =====================================================
    # Optimizer
    # =====================================================
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95)
        )

        return optimizer


# =========================================================
# Test
# =========================================================
if __name__ == "__main__":

    import time

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    clean_small = torch.zeros(
        (1, 1, 256, 256),
        dtype=torch.float32
    ).cuda()

    noisy_small = torch.zeros(
        (1, 2, 256, 256),
        dtype=torch.float32
    ).cuda()

    noisy_patch = torch.zeros(
        (1, 1, 256, 256),
        dtype=torch.float32
    ).cuda()

    pos = torch.tensor(
        [[0, 0, 256, 256]],
        dtype=torch.long
    ).cuda()

    model = Net().cuda()

    since = time.time()

    y = model(
        clean_small=clean_small,
        noisy_small=noisy_small,
        noisy_patch=noisy_patch,
        patch_pos=pos,
        target_hw=(256, 256),
    )

    print("time:", time.time() - since)

    print(
        y['denoised_patch'].shape
    )