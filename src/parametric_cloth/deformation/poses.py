"""Pose sourcing for training-data generation.

Realistic pose distributions come from AMASS (~10k SMPL poses from motion
capture). Using 100-500 representative poses across a few body shapes yields
enough samples for a per-garment MLP. These helpers subsample and normalize pose
arrays; the actual draping happens in the Blender training-data script.

SMPL-X body pose here is 21 joints x 3 axis-angle = 63 values per pose.
"""

from __future__ import annotations

import numpy as np

N_BODY_JOINTS = 21
POSE_DIM = N_BODY_JOINTS * 3


def zero_pose() -> np.ndarray:
    """The rest (T/A) pose: all joint rotations zero."""
    return np.zeros(POSE_DIM)


def random_poses(n: int, *, scale: float = 0.2, seed: int = 0) -> np.ndarray:
    """Small random axis-angle poses (radians) for smoke tests. (n, 63)."""
    rng = np.random.default_rng(seed)
    return rng.normal(scale=scale, size=(n, POSE_DIM))


def subsample_poses(poses: np.ndarray, n: int, *, seed: int = 0) -> np.ndarray:
    """Pick ``n`` poses spread across the set (evenly if possible, else random)."""
    poses = _to_pose_matrix(poses)
    total = len(poses)
    if n >= total:
        return poses
    if n <= 0:
        raise ValueError("n must be >= 1")
    # Even strides give better coverage of an ordered motion sequence than
    # random draws; fall back to random only if the array is unordered noise.
    idx = np.linspace(0, total - 1, n).round().astype(int)
    idx = np.unique(idx)
    if len(idx) < n:  # collisions from rounding; top up randomly
        rng = np.random.default_rng(seed)
        extra = rng.choice(np.setdiff1d(np.arange(total), idx),
                           size=n - len(idx), replace=False)
        idx = np.sort(np.concatenate([idx, extra]))
    return poses[idx]


def load_amass(path: str, *, n: int | None = None, seed: int = 0) -> np.ndarray:
    """Load poses from an AMASS-style ``.npz`` (expects a ``poses`` array).

    AMASS stores ``poses`` as (frames, >=66); the first 3 are global orientation
    and the next 63 are the body joints we use. Optionally subsample to ``n``.
    """
    data = np.load(path)
    if "poses" not in data:
        raise KeyError(f"{path} has no 'poses' array (keys: {list(data.keys())})")
    raw = np.asarray(data["poses"], dtype=float)
    body = raw[:, 3:3 + POSE_DIM] if raw.shape[1] >= 3 + POSE_DIM else raw[:, :POSE_DIM]
    return subsample_poses(body, n, seed=seed) if n else body


def _to_pose_matrix(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=float)
    if poses.ndim == 1:
        poses = poses[None]
    return poses
