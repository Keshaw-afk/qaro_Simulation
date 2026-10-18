"""Shared discrete action table: every policy output is one of N_ACT (speed, turn-rate) pairs. Numpy only, so deploy can import it."""
import numpy as np

F32 = np.float32
VMAX, WMAX = 0.20, 2.5
VS = np.array([0.0, 0.05, 0.10, 0.15, 0.20], F32)                         # speed choices (m/s)
WS = np.array([-2.0, -1.2, -0.7, -0.3, 0.0, 0.3, 0.7, 1.2, 2.0], F32)      # turn-rate choices (rad/s)
CV, CW = (a.ravel().astype(F32) for a in np.meshgrid(VS, WS, indexing="ij"))
N_ACT = len(CV)
