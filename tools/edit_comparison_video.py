"""Join one controller's demo clips into a short video (used by tools/make_comparison_videos.sh).

Per arena: a title card (CARD s), about 0.5 s of stillness, the run, about 1 s holding the final positions. Stretches
where nothing moves for more than 1.5 s are shortened to 0.8 s; runs longer than LONG s play normally for HEAD s, then at
SPEEDx (labelled on screen) until their last TAIL s. The clips' overlay clock shows true simulation time, so every cut
or speed-up stays visible. Motion is detected by frame differencing outside the overlay panel.

    python3 tools/edit_comparison_video.py CLIP_ROOT OUT_DIR SEED [SEED ...]
reads CLIP_ROOT/{ppo,dqn,rule}/demo_seed{SEED}.mp4, writes OUT_DIR/{ppo,dqn,rule}.avi (MJPG; convert with ffmpeg)."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import designed_layouts as D  # noqa: E402

FPS, LAG = 30, 5
LONG, HEAD, TAIL, SPEED, CARD = 20, 12, 2, 6, 1.0
NAMES = {"ppo": "PPO", "dqn": "DQN", "rule": "Rule controller (no learning)"}


def label(seed):
    return f"designed arena {seed}: {D.DESIGNS[seed][0]}" if seed in D.DESIGNS else f"layout {seed}"


def motion(path):
    """Per-frame flag: pixels changed vs LAG frames earlier, ignoring the overlay panel (its clock changes every frame)."""
    cap, hist, flags = cv2.VideoCapture(str(path)), [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.int16)
        g[128:, :200] = 0
        hist.append(g)
        prev = hist[-1 - LAG] if len(hist) > LAG else hist[0]
        flags.append(int((np.abs(g - prev) > 12).sum()) > 15)
        if len(hist) > LAG + 1:
            hist.pop(0)
    return np.array(flags)


def keep_frames(flags):
    act = np.flatnonzero(flags)
    if not len(act):
        return list(range(min(len(flags), 60)))
    start, end = max(0, act[0] - LAG - 15), min(len(flags) - 1, act[-1] + 30)
    keep, i = [], start
    while i <= end:
        if not flags[i]:
            j = i
            while j <= end and not flags[j]:
                j += 1
            run = list(range(i, j))
            keep += run if len(run) <= 45 or i == start or j > end else run[:12] + run[-12:]
            i = j
        else:
            keep.append(i)
            i += 1
    return keep


def card(w, h, title, sub, n, total):
    img = np.full((h, w, 3), (38, 34, 30), np.uint8)
    for text, y, sc, th, col in ((title, h // 2 - 20, 1.6, 3, (255, 255, 255)), (sub, h // 2 + 40, 1.0, 2, (200, 200, 200)),
                                 (f"arena {n} of {total}", h // 2 + 90, 0.7, 1, (160, 160, 160))):
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, th)[0][0]
        cv2.putText(img, text, ((w - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)
    return img


def main(src, out, seeds):
    out.mkdir(parents=True, exist_ok=True)
    for m, name in NAMES.items():
        writer, total_in, total_out = None, 0, 0
        for n, seed in enumerate(seeds, 1):
            path = src / m / f"demo_seed{seed}.mp4"
            if not path.exists():
                print(f"missing {path}: skipped", flush=True)
                continue
            order = keep_frames(motion(path))
            keep, fast, mid = set(order), set(), set()
            if len(order) > LONG * FPS:
                m_ = order[HEAD * FPS:len(order) - TAIL * FPS]
                mid, fast = set(m_), set(m_) - set(m_[::SPEED])
            cap = cv2.VideoCapture(str(path))
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if writer is None:
                writer = cv2.VideoWriter(str(out / f"{m}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (w, h))
            c = card(w, h, name, label(seed), n, len(seeds))
            for _ in range(int(CARD * FPS)):
                writer.write(c)
            k, written = 0, 0
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                if k in keep and k not in fast:
                    if k in mid:
                        cv2.rectangle(f, (w - 215, 20), (w - 20, 75), (25, 25, 25), -1)
                        cv2.putText(f, f"{SPEED}x speed", (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (80, 200, 255), 2, cv2.LINE_AA)
                    writer.write(f)
                    written += 1
                k += 1
            total_in += k
            total_out += written
            print(f"{m} {seed}: {k / FPS:.1f} s -> {written / FPS:.1f} s", flush=True)
        if writer is not None:
            writer.release()
            print(f"{m}: clips {total_in / FPS:.0f} s -> {total_out / FPS:.0f} s (+ title cards)", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), [int(s) for s in sys.argv[3:]])
