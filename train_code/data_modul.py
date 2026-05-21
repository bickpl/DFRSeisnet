import os
import random

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def guiyihua(data, min, max):
    if min == max:
        return data
    return 2 * (data - min) / (max - min) - 1


def data_list(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["npy"]:
            results.append(full_path)
        elif os.path.isdir(full_path):
            results.extend(data_list(full_path))
    return results


def load_data(path, full_shape=None):
    m = np.fromfile(path, dtype=np.float32)
    if full_shape is not None:
        expected = int(full_shape[0]) * int(full_shape[1])
        if m.size == expected:
            return m.reshape((int(full_shape[0]), int(full_shape[1])))
        if int(full_shape[1]) > 0 and m.size % int(full_shape[1]) == 0:
            return m.reshape((m.size // int(full_shape[1]), int(full_shape[1])))

    edge = int(np.sqrt(m.size))
    if edge * edge == m.size:
        return m.reshape((edge, edge))

    raise ValueError(f"Cannot infer shape for {path}, element count={m.size}.")


def modify_filename(filename):
    # Split filename and extension.
    name, ext = filename.rsplit(".", 1)

    # Split all underscore-separated parts.
    parts = name.split("_")

    # Remove the second-last segment.
    if len(parts) > 2:
        parts = parts[:-2] + [parts[-1]]

    # Rebuild filename.
    new_name = "_".join(parts) + "." + ext
    return new_name


def selfguiyihua(data):
    q5 = np.percentile(data, 5)
    q95 = np.percentile(data, 95)
    delta = q95 - q5

    # Stricter threshold to avoid floating-point false positives.
    if delta < 1e-6:
        return data
    normalized_quantile = 2 * (data - q5) / (q95 - q5 + 1e-8) - 1
    return normalized_quantile


class NoiseDataset(Dataset):
    def __init__(
            self,
            data_paths,
            full_shape=None,
            downsample_size=(256, 256),
            patch_size=(256, 256),
            stride=(256, 256),
            random_flip=True,
            cache_in_memory=True,
    ):
        super().__init__()
        self.noise_datas = data_list(os.path.join(data_paths, "noisy"))
        self.full_shape = full_shape
        self.downsample_size = downsample_size
        self.patch_size = patch_size
        self.stride = stride
        self.random_flip = random_flip
        self.cache_in_memory = cache_in_memory
        self.sample_indices = self._build_sample_indices()
        self.file_cache = {}
        if self.cache_in_memory:
            try:
                self._warmup_cache()
            except MemoryError:
                # Fallback to on-demand loading when host memory is insufficient.
                self.cache_in_memory = False
                self.file_cache = {}
                print("[NoiseDataset] cache_in_memory disabled due to MemoryError, falling back to lazy loading.")

    def _build_sample_indices(self):
        indices = []
        ph, pw = self.patch_size
        sh, sw = self.stride

        for file_idx, noisy_path in enumerate(self.noise_datas):
            full = np.load(noisy_path, mmap_mode="r")
            h, w = full.shape
            tops = list(range(0, h, sh))
            lefts = list(range(0, w, sw))
            if not tops:
                tops = [0]
            if not lefts:
                lefts = [0]

            for top in tops:
                for left in lefts:
                    indices.append((file_idx, top, left))
        return indices

    @staticmethod
    def _crop_with_zero_pad(x, top, left, ph, pw):
        _, h, w = x.shape
        bottom = min(top + ph, h)
        right = min(left + pw, w)
        patch = x[:, top:bottom, left:right]
        out = torch.zeros((1, ph, pw), dtype=x.dtype)
        out[:, : patch.shape[1], : patch.shape[2]] = patch
        return out

    def _warmup_cache(self):
        for file_idx, noisy_path in enumerate(self.noise_datas):
            clean_path = noisy_path.replace("noisy", "clean")
            noise_data = torch.from_numpy(np.load(noisy_path).astype(np.float32)).unsqueeze(0)
            clean = torch.from_numpy(np.load(clean_path).astype(np.float32)).unsqueeze(0)

            clean_small = F.interpolate(
                clean.unsqueeze(0), size=self.downsample_size, mode="bilinear", align_corners=False
            ).squeeze(0)
            noisy_small = F.interpolate(
                noise_data.unsqueeze(0), size=self.downsample_size, mode="bilinear", align_corners=False
            ).squeeze(0)

            entry = {
                "noise_data": noise_data.contiguous(),
                "clean": clean.contiguous(),
                "clean_small": clean_small.contiguous(),
                "noisy_small": noisy_small.contiguous(),
            }

            if self.random_flip:
                noise_flip = torch.flip(noise_data, dims=[2]).contiguous()
                clean_flip = torch.flip(clean, dims=[2]).contiguous()
                clean_small_flip = F.interpolate(
                    clean_flip.unsqueeze(0), size=self.downsample_size, mode="bilinear", align_corners=False
                ).squeeze(0).contiguous()
                noisy_small_flip = F.interpolate(
                    noise_flip.unsqueeze(0), size=self.downsample_size, mode="bilinear", align_corners=False
                ).squeeze(0).contiguous()
                entry.update({
                    "noise_data_flip": noise_flip,
                    "clean_flip": clean_flip,
                    "clean_small_flip": clean_small_flip,
                    "noisy_small_flip": noisy_small_flip,
                })

            self.file_cache[file_idx] = entry

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        file_idx, top, left = self.sample_indices[idx]
        noisy_path = self.noise_datas[file_idx]
        clean_path = noisy_path.replace("noisy", "clean")

        do_flip = self.random_flip and random.random() < 0.5
        if self.cache_in_memory:
            cache = self.file_cache[file_idx]
            if do_flip and "noise_data_flip" in cache:
                noise_data = cache["noise_data_flip"]
                clean = cache["clean_flip"]
                clean_small = cache["clean_small_flip"]
                noisy_small = cache["noisy_small_flip"]
            else:
                noise_data = cache["noise_data"]
                clean = cache["clean"]
                clean_small = cache["clean_small"]
                noisy_small = cache["noisy_small"]
        else:
            noise_data = np.load(noisy_path)
            clean = np.load(clean_path)
            if do_flip:
                noise_data = np.flip(noise_data, axis=1).copy()
                clean = np.flip(clean, axis=1).copy()
            noise_data = torch.from_numpy(noise_data.astype(np.float32)).unsqueeze(0)
            clean = torch.from_numpy(clean.astype(np.float32)).unsqueeze(0)
            clean_small = F.interpolate(
                clean.unsqueeze(0), size=self.downsample_size, mode="bilinear", align_corners=False
            ).squeeze(0)
            noisy_small = F.interpolate(
                noise_data.unsqueeze(0), size=self.downsample_size, mode="bilinear", align_corners=False
            ).squeeze(0)

        ph, pw = self.patch_size
        clean_patch = self._crop_with_zero_pad(clean, top, left, ph, pw)
        noisy_patch = self._crop_with_zero_pad(noise_data, top, left, ph, pw)
        noisy_small = noisy_small.repeat(2, 1, 1)
        patch_pos = torch.tensor([top, left, clean.shape[-2], clean.shape[-1]], dtype=torch.long)
        return {
            "clean_small": clean_small,
            "noisy_small": noisy_small,
            "clean_patch": clean_patch,
            "noisy_patch": noisy_patch,
            "patch_pos": patch_pos,
        }


class DataModule(L.LightningDataModule):
    def __init__(
            self,
            train_dir: str = "./test_data",
            val_dir: str = "./test_data",
            batch_size: int = 4,
            random_flip=True,
            num_workers: int = 0,
            full_shape=None,
            downsample_size=(128, 128),
            patch_size=(256, 256),
            stride=(256, 256),
            cache_in_memory=True,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.random_flip = random_flip
        self.num_workers = num_workers
        self.full_shape = full_shape
        self.downsample_size = downsample_size
        self.patch_size = patch_size
        self.stride = stride
        self.cache_in_memory = cache_in_memory

    def setup(self, stage: str):
        self.train_set = NoiseDataset(
            data_paths=self.train_dir,
            full_shape=self.full_shape,
            downsample_size=self.downsample_size,
            patch_size=self.patch_size,
            stride=self.stride,
            random_flip=self.random_flip,
            cache_in_memory=self.cache_in_memory,
        )
        self.val_set = NoiseDataset(
            data_paths=self.val_dir,
            full_shape=self.full_shape,
            downsample_size=self.downsample_size,
            patch_size=self.patch_size,
            stride=self.stride,
            random_flip=False,
            cache_in_memory=self.cache_in_memory,
        )

    def train_dataloader(self):
        loader_kwargs = dict(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4
        ld_train = DataLoader(**loader_kwargs)
        return ld_train

    def val_dataloader(self):
        loader_kwargs = dict(
            dataset=self.val_set,
            num_workers=self.num_workers,
            pin_memory=True,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        ld_val = DataLoader(**loader_kwargs)
        return ld_val
