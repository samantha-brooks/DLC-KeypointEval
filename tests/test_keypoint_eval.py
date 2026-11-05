import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import keypoint_eval as ke  # noqa: E402

BODYPARTS = ["nose", "withers", "hoof"]


def _table(scorer, values, likelihood=None):
    """values: (n_images, n_bodyparts, 2) -> DLC-style table."""
    cols = []
    if likelihood is None:
        coords = ["x", "y"]
    else:
        coords = ["x", "y", "likelihood"]
    cols = pd.MultiIndex.from_product([[scorer], BODYPARTS, coords], names=["scorer", "bodyparts", "coords"])
    n = values.shape[0]
    data = np.zeros((n, len(BODYPARTS), len(coords)))
    data[:, :, :2] = values
    if likelihood is not None:
        data[:, :, 2] = likelihood
    idx = pd.MultiIndex.from_tuples([("labeled-data", "vid", f"img{i:03d}.png") for i in range(n)])
    return pd.DataFrame(data.reshape(n, -1), index=idx, columns=cols)


@pytest.fixture
def project(tmp_path):
    cfg = {"Task": "Horse", "date": "Nov4", "scorer": "Sam", "TrainingFraction": [0.8],
           "iteration": 0, "pcutoff": 0.6}
    import yaml

    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    tdir = tmp_path / "training-datasets" / "iteration-0" / "UnaugmentedDataSet_HorseNov4"
    tdir.mkdir(parents=True)

    n = 10
    gt_vals = np.full((n, 3, 2), 100.0)
    gt_vals[3, 1] = np.nan  # withers unlabeled in image 3
    gt = _table("Sam", gt_vals)
    gt.to_hdf(tdir / "CollectedData_Sam.h5", key="df_with_missing")

    # Errors: nose always 3px in x (error 3); withers: 4px in y; hoof: images 0-7 error 1, 8-9 error 50
    pred_vals = gt_vals.copy()
    pred_vals[:, 0, 0] += 3
    pred_vals[:, 1, 1] += 4
    pred_vals[:, 2, 0] += 1
    pred_vals[8:, 2, 0] += 49
    lik = np.full((n, 3), 0.9)
    lik[8:, 2] = 0.2  # the two bad hoof predictions have low confidence
    pred = _table("DLC_scorer", pred_vals, lik)
    # train = images 0-7, test = images 8-9
    with open(tdir / "Documentation_data-Horse_80shuffle1.pickle", "wb") as f:
        pickle.dump([None, list(range(8)), [8, 9], 0.8], f)
    ev = tmp_path / "evaluation-results-pytorch" / "iteration-0" / "HorseNov4-trainset80shuffle1"
    ev.mkdir(parents=True)
    pred.to_hdf(ev / "DLC_scorer.h5", key="df_with_missing")
    return tmp_path


def test_end_to_end(project):
    res = ke.run(project / "config.yaml", quiet=True)
    train, test, allt = res["tables"]["train"], res["tables"]["test"], res["tables"]["all"]

    assert train.loc["nose", "Mean_error_px"] == pytest.approx(3.0)
    assert train.loc["withers", "Mean_error_px"] == pytest.approx(4.0)
    assert train.loc["hoof", "Mean_error_px"] == pytest.approx(1.0)
    assert test.loc["hoof", "Mean_error_px"] == pytest.approx(50.0)
    assert test.loc["hoof", "Pct_above_pcutoff"] == 0.0
    assert np.isnan(test.loc["hoof", "Mean_error_px_pcutoff"])
    # unlabeled withers point (image 3) is excluded from the count
    assert train.loc["withers", "N_labeled"] == 7
    assert allt.loc["hoof", "N_labeled"] == 10
    # mean vs true RMSE differ for hoof across all images: mean=(8*1+2*50)/10
    assert allt.loc["hoof", "Mean_error_px"] == pytest.approx(10.8)
    assert allt.loc["hoof", "RMSE_px"] == pytest.approx(np.sqrt((8 * 1 + 2 * 2500) / 10))
    assert allt.loc["hoof", "Pct_within_5px"] == pytest.approx(80.0)
    assert allt.loc["hoof", "Bias_x_px"] == pytest.approx(10.8)
    assert allt.loc["hoof", "Rank_worst_first"] == 1
    # ALL row pools every point (DLC's headline number)
    pooled = res["errors"]["error"].mean()
    assert allt.loc["ALL", "Mean_error_px"] == pytest.approx(pooled)

    out = res["out_dir"]
    for name in ("per_keypoint_train.csv", "per_keypoint_test.csv", "per_keypoint_all.csv",
                 "worst_predictions.csv", "keypoint_eval.xlsx", "per_keypoint_errors.png"):
        assert (out / name).exists(), name
    worst = res["worst"]
    assert worst[worst.keypoint == "hoof"].iloc[0]["error"] == pytest.approx(50.0)


def test_missing_prediction_counted(project):
    ev = project / "evaluation-results-pytorch" / "iteration-0" / "HorseNov4-trainset80shuffle1" / "DLC_scorer.h5"
    pred = pd.read_hdf(ev)
    pred.iloc[0, 0:3] = np.nan  # nose in image 0 undetected
    pred.to_hdf(ev, key="df_with_missing")
    res = ke.run(project / "config.yaml", quiet=True)
    t = res["tables"]["train"]
    assert t.loc["nose", "N_labeled"] == 8
    assert t.loc["nose", "N_predicted"] == 7


def test_no_split_file_falls_back_to_all(project):
    (project / "training-datasets/iteration-0/UnaugmentedDataSet_HorseNov4/Documentation_data-Horse_80shuffle1.pickle").unlink()
    res = ke.run(project / "config.yaml", quiet=True)
    assert set(res["tables"]) == {"all"}


def _touch(tmp_path, names):
    out = []
    for n in names:
        p = tmp_path / n
        p.write_text("x")
        out.append(p)
    return out


def test_prefers_best_snapshot_over_newer_or_higher(tmp_path):
    files = _touch(tmp_path, ["DLC_x_snapshot_best-120.h5", "DLC_x_snapshot_200.h5"])
    assert ke.choose_predictions_file(files).name == "DLC_x_snapshot_best-120.h5"


def test_no_best_uses_highest_epoch(tmp_path):
    files = _touch(tmp_path, ["DLC_x_snapshot_200.h5", "DLC_x_snapshot_50.h5", "DLC_x_snapshot_100.h5"])
    assert ke.choose_predictions_file(files).name == "DLC_x_snapshot_200.h5"


def test_snapshot_override(tmp_path):
    files = _touch(tmp_path, ["DLC_x_snapshot_best-120.h5", "DLC_x_snapshot_200.h5"])
    assert ke.choose_predictions_file(files, "_200").name == "DLC_x_snapshot_200.h5"
    with pytest.raises(FileNotFoundError):
        ke.choose_predictions_file(files, "nope")


def test_missing_evaluation_folder_errors(tmp_path):
    import yaml

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"Task": "T", "date": "d", "scorer": "S", "TrainingFraction": [0.95]})
    )
    with pytest.raises(FileNotFoundError):
        ke.run(tmp_path / "config.yaml", quiet=True)
