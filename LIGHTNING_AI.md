# Training on Lightning AI

## 1. Create a Studio

- Go to lightning.ai → New Studio.
- Pick a GPU: **L4** or **A10G** is a solid, affordable choice for `yolo11l`
  at 150 epochs on ~6,700 images (a few hours). Use **A100** if you want to
  try `yolo11x` or train faster.

## 2. Get the project onto the Studio

Either upload the `project/` folder via the Studio file browser, or push it
to a git repo and clone it in the Studio terminal:

```bash
git clone <your-repo-url>
cd project
```

## 3. Install dependencies

```bash
pip install -r requirements-training.txt
```

## 4. Set your Roboflow API key (optional, avoids the interactive prompt)

```bash
export ROBOFLOW_API_KEY=your_key_here
```

(Find it under Roboflow → Settings → API Keys.)

## 5. Run training in the background

Studio browser tabs can disconnect — always launch long runs with `nohup`
so they keep going:

```bash
nohup python train_cheating_model.py > train.log 2>&1 &
tail -f train.log        # watch progress; Ctrl+C just stops watching, not the run
```

Check progress any time with:

```bash
tail -n 50 train.log
```

## 6. Persist your results

Lightning Studios keep your home directory across restarts, so
`runs_cheating_detection/` (set by `PROJECT_DIR` in the script) will still
be there next time you open the Studio. If you want it backed up elsewhere
too, download `runs_cheating_detection/yolo11l_v5/weights/best.pt` via the
file browser once training finishes.

## 7. Bring the trained model back to the app

Copy `best.pt` into `backend/` and update `backend/config.json`:

```json
{
  "model_path": "best.pt",
  ...
}
```

## Tuning for accuracy vs. time

| If you want... | Change |
|---|---|
| Faster iteration to sanity-check the pipeline | `BASE_MODEL="yolo11m.pt"`, `EPOCHS=50` |
| Best possible accuracy, time is not a constraint | `BASE_MODEL="yolo11x.pt"` on an A100, `EPOCHS=200` |
| Better detection of small objects (phones, notes) | Raise `IMG_SIZE` to 960 or 1280 (slower, more GPU memory) |
| To resume an interrupted run | Set `RESUME_FROM` to the `last.pt` path shown in `train.log` |

After training, compare the new `mAP50` / `mAP50-95` printed at the end
against the notebook's baseline (mAP50 ≈ 0.71, mAP50-95 ≈ 0.43) to confirm
the larger model + augmentation actually improved things before you deploy it.
