"""Pre-generated pool of validated layouts (with free-space masks) for the 2D training env.
Training uses seeds >= 1000; seeds 0..199 are reserved for evaluation so the two never overlap."""
import pathlib
import sys
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
import layout as L

K_MAX = L.OBSTACLE_COUNT[1]
ROWS, COLS = int(L.ARENA_H / L.GRID_RES), int(L.ARENA_W / L.GRID_RES)


def _one(seed):
    lay = L.generate_layout(seed)
    obst = np.tile(np.array([100.0, 100.0, 0.01, 0.01, 0.0], np.float32), (K_MAX, 1))  # padding: far-away dummy box
    obst[: len(lay.obstacles)] = lay.obstacles
    labels, _ = ndimage.label(~L.blocked_map(lay, L.ROBOT_RADIUS + L.CLEARANCE_MARGIN))
    r, c = L._cell(lay, lay.starts[0][0], lay.starts[0][1])
    return obst, labels == labels[r, c], np.array(lay.starts, np.float32), np.array(lay.goals, np.float32)


def build(seeds, path=None, procs=10):
    path = pathlib.Path(path) if path else None
    if path and path.exists():
        d = np.load(path)
        return {k: d[k] for k in d.files}
    with Pool(procs) as p:
        res = p.map(_one, list(seeds), chunksize=8)
    out = {"obst": np.stack([r[0] for r in res]), "free": np.stack([r[1] for r in res]),
           "starts": np.stack([r[2] for r in res]), "goals": np.stack([r[3] for r in res]), "seeds": np.array(list(seeds))}
    if path:
        np.savez_compressed(path, **out)
    return out


TRAIN_SEEDS, EVAL_SEEDS = range(1000, 3000), range(0, 200)
