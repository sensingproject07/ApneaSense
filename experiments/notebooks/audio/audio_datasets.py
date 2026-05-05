"""
Shared Dataset classes for audio experiments.

Single canonical copy for all audio notebooks. Lives at the audio-notebook
root (experiments/notebooks/audio/) so any notebook below it can import it
by adding the audio root to sys.path:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd().parent))   # for notebooks one level deep
    from audio_datasets import MelSpectrogramDataset

Being importable from a .py file is what allows PyTorch DataLoader worker
processes to pickle the Dataset on Windows + Jupyter. Classes defined
directly in notebook cells cannot be pickled by spawned workers.
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MelSpectrogramDataset(Dataset):
    """Loads precomputed Mel spectrograms (float16) from per-subject .npy files.

    Expected layout:
        precomputed_dir / {subject_id} / {subject_id}_{file_type}_mel.npy

    Each file is a float16 array of shape (n_segments, n_mels, time_frames).
    Values are dB-scaled with a fixed reference (no per-sample normalization)
    and linearly mapped to [0, 1] from a global dB window — this preserves
    cross-segment loudness differences, which the EDA flagged as
    discriminative for apnea vs non-apnea.

    Metadata columns are pre-extracted into numpy arrays at init time to
    avoid slow pandas .iloc[] lookups in __getitem__.

    Augmentation (SpecAugment) is intentionally NOT applied here — it runs
    batch-wise on GPU inside the training loop for speed.
    """

    def __init__(self, dataframe, precomputed_dir, max_cache=16):
        self.precomputed_dir = Path(precomputed_dir)

        df = dataframe.reset_index(drop=True)
        self.subject_ids = df["subject_id"].values
        self.file_types = df["file_type"].values
        self.segment_idxs = df["segment_idx"].values.astype(np.int64)
        self.labels = df["label"].values.astype(np.int64)
        self._length = len(df)

        # LRU cache of loaded subject arrays. With num_workers > 0, each
        # worker has its own cache; size it generously since workers only
        # see a subset of subjects per shuffle epoch.
        self._cache = OrderedDict()
        self._max_cache = max_cache

    def _load_array(self, subject_id, file_type):
        key = (subject_id, file_type)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if len(self._cache) >= self._max_cache:
            self._cache.popitem(last=False)
        path = self.precomputed_dir / subject_id / f"{subject_id}_{file_type}_mel.npy"
        self._cache[key] = np.load(path, allow_pickle=False)
        return self._cache[key]

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        arr = self._load_array(self.subject_ids[idx], self.file_types[idx])
        mel = arr[self.segment_idxs[idx]]  # (n_mels, time_frames), float16
        tensor = torch.from_numpy(np.ascontiguousarray(mel, dtype=np.float32)).unsqueeze(0)
        return tensor, int(self.labels[idx])


class MFCCDataset(Dataset):
    """Loads precomputed MFCC features (float16) from per-subject .npy files.

    Expected layout:
        precomputed_dir / {subject_id} / {subject_id}_{file_type}_mfcc.npy

    Each file is a float16 array of shape (n_segments, n_mfcc, time_frames).
    Values are stored RAW (no per-sample normalization) — the model handles
    scaling via a BatchNorm2d(1) layer at its input. Per-sample mean/std
    normalization (a common MFCC default) destroys cross-segment loudness,
    which the EDA flagged as discriminative for apnea vs non-apnea.

    Metadata columns are pre-extracted into numpy arrays at init time to
    avoid slow pandas .iloc[] lookups in __getitem__.
    """

    def __init__(self, dataframe, precomputed_dir, max_cache=16):
        self.precomputed_dir = Path(precomputed_dir)

        df = dataframe.reset_index(drop=True)
        self.subject_ids = df["subject_id"].values
        self.file_types = df["file_type"].values
        self.segment_idxs = df["segment_idx"].values.astype(np.int64)
        self.labels = df["label"].values.astype(np.int64)
        self._length = len(df)

        self._cache = OrderedDict()
        self._max_cache = max_cache

    def _load_array(self, subject_id, file_type):
        key = (subject_id, file_type)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if len(self._cache) >= self._max_cache:
            self._cache.popitem(last=False)
        path = self.precomputed_dir / subject_id / f"{subject_id}_{file_type}_mfcc.npy"
        self._cache[key] = np.load(path, allow_pickle=False)
        return self._cache[key]

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        arr = self._load_array(self.subject_ids[idx], self.file_types[idx])
        mfcc = arr[self.segment_idxs[idx]]  # (n_mfcc, time_frames), float16
        tensor = torch.from_numpy(np.ascontiguousarray(mfcc, dtype=np.float32)).unsqueeze(0)
        return tensor, int(self.labels[idx])


class RawWaveformDataset(Dataset):
    """Loads raw audio segments (float32) from per-subject .npy files.

    Expected layout:
        psg_dir / {subject_id} / {subject_id}_{file_type}.npy

    Each file is shape (n_segments, n_samples) at 16 kHz. Used by backbones
    that compute their own features inside `forward()` (e.g. AST, PANN).

    Metadata columns are pre-extracted into numpy arrays at init time.
    """

    def __init__(self, dataframe, psg_dir, max_cache=16):
        self.psg_dir = Path(psg_dir)

        df = dataframe.reset_index(drop=True)
        self.subject_ids = df["subject_id"].values
        self.file_types = df["file_type"].values
        self.segment_idxs = df["segment_idx"].values.astype(np.int64)
        self.labels = df["label"].values.astype(np.int64)
        self._length = len(df)

        self._cache = OrderedDict()
        self._max_cache = max_cache

    def _load_array(self, subject_id, file_type):
        key = (subject_id, file_type)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if len(self._cache) >= self._max_cache:
            self._cache.popitem(last=False)
        path = self.psg_dir / subject_id / f"{subject_id}_{file_type}.npy"
        self._cache[key] = np.load(path, allow_pickle=False)
        return self._cache[key]

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        arr = self._load_array(self.subject_ids[idx], self.file_types[idx])
        wave = arr[self.segment_idxs[idx]]  # (n_samples,)
        tensor = torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32))
        return tensor, int(self.labels[idx])
