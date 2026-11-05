# DLC-KeypointEval

DeepLabCut's *Evaluate network* step gives one train error and one test error averaged over **all** keypoints. This tool breaks those numbers down **per keypoint**, so you can see which points the model has learned well and which it hasn't.

It does not re-run the model. It reads the predictions file that `evaluate_network` already saved, and compares it with your labeled data.

## Use

1. In DeepLabCut, run **Evaluate network** as usual.
2. Then run:

```bash
python keypoint_eval.py --config "C:\path\to\your\project\config.yaml"
```

Options (all optional): `--shuffle 1`, `--trainingsetindex 0`, `--pcutoff 0.6` (default: from config.yaml), `--snapshot <text>` (pick a particular evaluated snapshot; by default the **best** snapshot is used, the one DLC uses to analyze videos, or the highest-epoch one if no best exists), `--thresholds 5 10 20`, `--worst 5`, `--out <folder>`.

Run it in the DEEPLABCUT environment (needs pandas, pyyaml, pytables; matplotlib and openpyxl for the plot and Excel file).

## Output

Written to `<project>/evaluation-results-pytorch/.../keypoint-eval/` (or `--out`):

| File | Contents |
|---|---|
| `keypoint_eval.xlsx` | One sheet each for train / test / all images, plus `worst_predictions` |
| `per_keypoint_train/test/all.csv` | The same tables as CSV |
| `worst_predictions.csv` | The largest individual errors per keypoint, with image names, so you can go look at them |
| `per_keypoint_errors.png` | Bar chart and box plot of error per keypoint |

## Columns

- **Mean_error_px** – mean distance between prediction and label. **This is what DeepLabCut reports as "RMSE"/"error (px)"**; the `ALL` row should match DLC's printed train/test error.
- **RMSE_px** – true root-mean-square error. Larger than the mean when there are a few big misses.
- **Median / SD / P90 / P95 / Max_error_px** – spread. A good median with a large max means occasional big failures rather than general inaccuracy.
- **Pct_within_Npx** – % of labeled points predicted within N pixels.
- **Bias_x_px / Bias_y_px** – average signed offset (prediction − label). A value well away from 0 means a systematic shift, e.g. the model consistently places a point too high.
- **N_labeled / N_predicted** – labeled points, and how many got a prediction.
- **Mean_likelihood, Pct_above_pcutoff, Mean_error_px_pcutoff, RMSE_px_pcutoff** – the same idea as DLC's "with p-cutoff" numbers: only predictions the model was confident about.
- **Rank_worst_first** – 1 = keypoint with the highest mean error.

Errors are in image pixels, so they are only comparable between keypoints of the same video resolution.
