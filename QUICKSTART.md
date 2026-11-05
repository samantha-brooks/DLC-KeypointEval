# DLC-KeypointEval — Quickstart Guide

Welcome to **DLC-KeypointEval**, a small helper tool that shows how accurate
your DeepLabCut model is **for each individual landmark**, not just on
average. This guide walks through the whole workflow in plain terms — no
coding experience required, just comfort typing a couple of commands into a
terminal window.

---

## Why this tool exists

When you click **Evaluate network** in DeepLabCut, it gives you just two
numbers: one error for the training images and one for the test images.
Each is an **average over every landmark at once**.

That average can hide a lot. A model can be excellent on the hooves and
poor on the nostril, and the single number blends the two together. If you
want to know *which* landmarks you can trust, and which need more labeled
examples, you need the numbers broken down per landmark.

**DLC-KeypointEval does that.** It doesn't re-run the model or touch your
videos. It reads the results DeepLabCut already saved when you ran
**Evaluate network**, compares them to your hand-labeled frames, and gives
you a table, a chart, and a list of the worst misses — one row per
landmark.

---

## What You Need

- A DeepLabCut project where you have **already run Evaluate network** (that
  step creates the results this tool reads)
- Python installed via the same conda environment used for DeepLabCut
  (see [DLC-Analyst's QUICKSTART](../dlc-analyst/QUICKSTART.md) if you don't
  have this set up yet)
- A way to open `.xlsx` or `.csv` files — Excel works fine

---

## The Interface at a Glance

DLC-KeypointEval doesn't have a window — you run it by typing one short
command into a terminal (Anaconda Prompt on Windows). This guide gives you
the exact text to copy.

---

## Step-by-Step Workflow

> **⚠ If any folder in your path has a space in it, quote the whole path.**
> Windows folder names often have spaces (`My Videos`,
> `OneDrive - University of Florida`, `DLC Code Workspace`). Every command
> below shows paths wrapped in double quotes (`"path\to\thing"`) — **always
> keep those quotes**, even when you swap in your own path. Without them
> the terminal reads everything after the first space as a separate
> argument and gives a confusing "path not found"-style error.
>
> ✅ `--config "C:\Users\you\Desktop\DLC Code Workspace\my_project\config.yaml"`
> ❌ `--config C:\Users\you\Desktop\DLC Code Workspace\my_project\config.yaml`
>
> **Tip:** you don't have to type paths by hand — drag the file from File
> Explorer and drop it into the terminal window; it pastes in correctly
> quoted.

### 0. Open a terminal and get to the right folder

Open **Anaconda Prompt** (Windows), activate your DeepLabCut environment,
then move into the DLC-KeypointEval folder (the one that directly contains
`keypoint_eval.py` and `README.md`):

```
conda activate deeplabcut
cd "path\to\DLC-KeypointEval"
```

> **Note on folder names:** if you downloaded the code with GitHub's green
> **Code → Download ZIP** button, the folder will be named
> **`DLC-KeypointEval-main`**. That's normal — just use that name in the
> `cd` command.

**How to check you're in the right place:** type `dir` and press Enter. You
should see `keypoint_eval.py` in the list.

There is **no install step** — it's a single script that uses packages your
DeepLabCut environment already has.

### 1. Make sure Evaluate network has been run

In DeepLabCut, run **Evaluate network** for the model (shuffle) you want to
check, as you normally would. This saves the model's predictions on all your
labeled images — the thing DLC-KeypointEval reads.

### 2. Run DLC-KeypointEval

```
python keypoint_eval.py --config "path\to\your_project\config.yaml"
```

- `--config` — your DeepLabCut project's `config.yaml` file

That's all that's required. It automatically:

- finds the evaluation results for your project,
- uses the **best** model snapshot — the same one DeepLabCut uses when it
  analyzes your videos in the next step (if no "best" snapshot exists, it
  uses the one trained longest and tells you so),
- separates your **training** images from your **test** images exactly the
  way DeepLabCut did.

Optional extras (add to the end of the command):

| Option | What it does |
|---|---|
| `--shuffle 2` | Check shuffle 2 instead of shuffle 1 (the default) |
| `--pcutoff 0.8` | Use a different confidence cutoff (default: the one in your config.yaml) |
| `--snapshot 200` | Check a specific snapshot instead of the best — give any part of its file name |
| `--thresholds 5 10 20` | The pixel distances used for the "% within N pixels" columns |
| `--worst 10` | List the 10 worst misses per landmark (default 5) |
| `--out "path\to\folder"` | Save the results somewhere other than the default |

### 3. Read the results

The terminal prints a table for your **test** images (the honest measure —
the model never saw these during training), with the worst landmark at the
top, and then lists where it saved everything:

| File | What it is |
|---|---|
| `keypoint_eval.xlsx` | Everything in one Excel file: a sheet for train, test, and all images, plus the worst misses |
| `per_keypoint_errors.png` | A chart: error per landmark, and the spread of those errors |
| `worst_predictions.csv` | The biggest individual misses, with image names, so you can open those frames and see what went wrong |
| `per_keypoint_train.csv`, `_test.csv`, `_all.csv` | The same tables as plain CSV files |

They are saved in a `keypoint-eval` folder inside your project's
evaluation-results folder (the terminal prints the exact path).

---

## How to Read the Numbers

All errors are in **pixels** — the distance between where you labeled a
landmark and where the model put it. Smaller is better.

| Column | What it tells you |
|---|---|
| **Mean_error_px** | The average error. **This is the number DeepLabCut itself labels "RMSE"** — the `ALL` row should match what DeepLabCut printed for train/test. |
| **RMSE_px** | A stricter version of the average that punishes big misses more. If it's much larger than Mean_error_px, a few large failures are dragging that landmark down. |
| **Median_error_px** | The typical error. Compared with the mean: a small median but a large mean means "usually good, occasionally way off." |
| **P90 / P95 / Max_error_px** | How bad the worst cases get. |
| **Pct_within_5px** (etc.) | Percent of labeled points the model placed within that many pixels. Easy to explain in a paper: "92% of hoof points within 10 px." |
| **Bias_x_px / Bias_y_px** | The average *direction* of the error. Near 0 is good. A big value means the model is consistently shifted — e.g. always placing a landmark too high. |
| **N_labeled / N_predicted** | How many labeled points there were, and how many the model gave an answer for. Few labels for a landmark often explains a poor score. |
| **Pct_above_pcutoff** | Percent of points the model was confident about (above your confidence cutoff). |
| **Mean_error_px_pcutoff** | The average error counting only those confident points. |
| **Rank_worst_first** | 1 = the landmark with the highest error. |

**Comparing train and test:** if a landmark does well on training images but
poorly on test images, the model has memorized rather than learned it —
usually fixed by labeling more varied examples of that landmark. If it's poor
on *both*, the landmark may be hard to see or inconsistently labeled.

**Note on units:** errors are in image pixels, so compare landmarks within
the same video resolution. Don't compare numbers from videos filmed at
different resolutions without converting.

---

## Tips for Best Results

- **Focus on the test table.** Training error will nearly always look better.
- **Open the worst frames.** `worst_predictions.csv` tells you exactly which
  image to look at for each landmark. Often the cause is a labeling slip
  (a mislabeled frame) rather than a model problem — and a quick fix there
  helps every future model.
- **Check the `N_labeled` column** before drawing conclusions about a
  landmark with only a handful of labeled points.
- **Re-run after each new model.** Each run is independent and nothing is
  modified in your project, so it's always safe to repeat.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python: can't open file 'keypoint_eval.py'` | You're not in the DLC-KeypointEval folder. Re-do Step 0 and check that `dir` shows `keypoint_eval.py`. |
| `ERROR: No evaluation folder found ...` | Evaluate network hasn't been run for that shuffle yet — run it in DeepLabCut first. If you evaluated a different shuffle, add `--shuffle N`. |
| `ERROR: No predictions .h5 found ...` | The evaluation folder exists but is empty, or your `--snapshot` text doesn't match any file name. Re-run Evaluate network, or drop `--snapshot`. |
| `ERROR: Ground-truth file not found` | The project's `training-datasets` folder is missing the `CollectedData_<name>.h5` file — it's created when you build a training dataset. |
| `WARNING: train/test split file not found` | The tool couldn't tell which images were train vs. test, so it reports all images together (the `all` sheet only). |
| `No 'best' snapshot was evaluated; using the highest-epoch one` | Not an error. DeepLabCut didn't save a "best" model for this run, so the one trained longest is used (this is also what DeepLabCut would use). |
| "other evaluated snapshots ignored" | Not an error. You've evaluated more than one snapshot; add `--snapshot` with part of a file name to look at a different one. |
| `No module named yaml` (or `tables`, `openpyxl`) | You're not in the DeepLabCut conda environment. Run `conda activate deeplabcut` and try again. |
| Error mentions a path not found, but the file is definitely there | Almost always a missing quote around a path with a space in it — see the warning at the top of the workflow section. |
| No `.xlsx` or `.png` produced | The `openpyxl` / `matplotlib` package is missing; the CSV files are still written. Use the DeepLabCut environment, which has both. |

---

*DLC-KeypointEval was built to complement [DLC-Analyst](../dlc-analyst), [DLC-Mirror](../DLC-Mirror) and [DeepLabCut-Display](../DeepLabCut-Display). DeepLabCut is developed by the Mathis Lab: https://github.com/DeepLabCut/DeepLabCut*
