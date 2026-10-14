"""Ten-sensor acquisition contract, supervised training and advisory replay.

No hardware commands. Real data must include independent, complete event labels.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
from week56_realtime import StreamWindow, parse_frame

SENSORS = [
    ("ground_potential_a", .1, 0., 50., "V AC"),
    ("ground_potential_b", .1, 0., 50., "V AC"),
    ("leakage_current", 1., .001, 1., "A"),
    ("element_temperature_a", 60., 25., 150., "degC"),
    ("element_temperature_b", 60., 25., 150., "degC"),
    ("element_temperature_c", 60., 25., 150., "degC"),
    ("relative_humidity", 3600., 20., 90., "%"),
    ("ambient_temperature", 3600., -20., 80., "degC"),
    ("input_voltage", 1., 0., 1500., "V"),
    ("lightning_sensor", .1, 0., 1., "H/L"),
]
FEATURES = [s[0] for s in SENSORS]
AGES = ["age_s_" + s for s in FEATURES]
HEADER = ["time_s"] + FEATURES + AGES
LOOKBACK, HORIZON = 30, 10


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def new_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_data(path, manifest_path, require_labels=True):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    if manifest.get("features") != FEATURES or manifest.get("step_seconds") != 1:
        raise ValueError("Exactly the declared ten features and one-second grid are required")
    if manifest.get("data_origin") not in ("synthetic", "real"):
        raise ValueError("Declare data_origin as synthetic or real")
    if manifest["data_origin"] == "real":
        for key in ("site_id", "time_reference", "sensor_mapping_source"):
            if not manifest.get(key) or str(manifest[key]).startswith("TODO"):
                raise ValueError(f"Real data requires verified metadata: {key}")
        if require_labels and (manifest.get("independent_event_labels") is not True
                               or manifest.get("event_catalog_complete") is not True
                               or not manifest.get("event_source")
                               or str(manifest["event_source"]).startswith("TODO")):
            raise ValueError("Real training requires independent event labels and complete observation coverage")
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != HEADER:
            raise ValueError("CSV columns/order must equal the acquisition template (including ten ages)")
        rows = []
        for r in reader:
            if None in r or any(r[k] in (None, "") for k in HEADER):
                raise ValueError("Ragged CSV row or missing value")
            rows.append([float(r[k]) for k in HEADER])
    a = np.asarray(rows, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 21 or not np.isfinite(a).all():
        raise ValueError("Missing, nonnumeric or nonfinite sensor data")
    t, raw, ages = a[:, 0], a[:, 1:11], a[:, 11:21]
    if np.any(t < 0) or np.any(t != np.floor(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("time_s must be nonnegative integer seconds, strictly increasing")
    if np.any(ages < 0):
        raise ValueError("Negative observation age")
    for j, (name, period, low, high, _) in enumerate(SENSORS):
        if np.any((raw[:, j] < low) | (raw[:, j] > high)):
            raise ValueError(f"Out-of-range sensor: {name}; confirm units, do not silently clip")
        if np.any(ages[:, j] > period * 2 + 1e-6):
            raise ValueError(f"Stale sensor: {name}; leave a time gap instead of filling unavailable data")
    if not np.isin(raw[:, -1], [0, 1]).all():
        raise ValueError("lightning_sensor must be binary")
    return t, raw.astype(np.float32), manifest


def load_events(path, t):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["time_s", "event_id"]:
            raise ValueError("Event columns must be time_s,event_id")
        rows = [(float(r["time_s"]), r["event_id"]) for r in reader]
    events = np.asarray([r[0] for r in rows], dtype=float)
    ids = [r[1] for r in rows]
    if not np.isfinite(events).all() or np.any(np.diff(events) <= 0):
        raise ValueError("Events must have unique, finite, increasing timestamps")
    if any(not s for s in ids) or len(set(ids)) != len(ids):
        raise ValueError("Event identifiers must be nonempty and unique")
    if len(events) and (events[0] < t[0] or events[-1] > t[-1]):
        raise ValueError("Event catalog must be limited to this sensor observation period")
    return events


def make_windows(t, raw, events, start, stop):
    # Each window AND its complete future label interval must lie in its split.
    anchors = np.arange(start + LOOKBACK - 1, stop - HORIZON)
    if len(anchors):
        gaps = np.r_[0, np.cumsum(np.diff(t) != 1)]
        anchors = anchors[(gaps[anchors + HORIZON] - gaps[anchors - LOOKBACK + 1]) == 0]
    if not len(anchors):
        raise ValueError("No contiguous 30-second input + 10-second label windows")
    x = np.stack([raw[i - 29:i + 1] for i in anchors])
    at = t[anchors]
    y = (np.searchsorted(events, at + HORIZON, side="right")
         > np.searchsorted(events, at, side="right")).astype(np.int32)
    return x, y, anchors


def binary_metrics(y, score, threshold):
    pred = np.asarray(score) >= threshold
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}


def choose_threshold(y, score, max_fpr):
    # Exact validation-score boundaries; test values never affect selection.
    score = np.asarray(score)
    order = np.argsort(-score, kind="stable")
    s, labels = score[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(labels == 1)[ends]; fp = np.cumsum(labels == 0)[ends]
    positives, negatives = int(np.sum(y == 1)), int(np.sum(y == 0))
    fn = positives - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    eligible = fp / max(negatives, 1) <= max_fpr
    # A no-alert threshold remains a valid option if every alert exceeds the limit.
    if not np.any(eligible):
        return float(np.nextafter(score.max(), np.inf))
    candidates = np.flatnonzero(eligible)
    best = candidates[np.argmax(f1[candidates])]
    return float(s[ends[best]])


def event_metrics(t, raw, anchors, events, score, threshold):
    eligible = detected = quiet_total = quiet_detected = 0
    leads = []
    at = t[anchors]
    for event in events:
        indices = np.flatnonzero((at >= event - HORIZON) & (at < event))
        # Require complete prediction coverage before counting an event.
        if not len(indices) or not np.array_equal(at[indices], np.arange(math.ceil(event - HORIZON), math.ceil(event))):
            continue
        eligible += 1
        alert = indices[score[indices] >= threshold]
        if len(alert):
            detected += 1; leads.append(float(event - at[alert[0]]))
        # Only score strict pre-detection warnings when each input history is quiet.
        quiet = np.array([not raw[anchors[k] - 29:anchors[k] + 1, -1].any() for k in indices])
        if quiet.all():
            quiet_total += 1
            quiet_detected += int(bool(len(alert)))
    return {"eligible_events": eligible, "warned_events": detected,
            "event_recall": detected / eligible if eligible else None,
            "earliest_warning_lead_seconds": leads,
            "quiet_history_eligible_events": quiet_total,
            "quiet_history_warned_events": quiet_detected,
            "quiet_history_recall": quiet_detected / quiet_total if quiet_total else None,
            "quiet_history_definition": "all ten pre-event windows have no lightning_sensor=1 in their previous 30 seconds; not proof of first-ever strike"}


def tf_setup():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    return tf


def train(args):
    t, raw, source = load_data(args.csv, args.manifest)
    events = load_events(args.events, t)
    n = len(t); cuts = [0, int(n * .7), int(n * .85), n]
    if source["data_origin"] == "real":
        days = np.unique(np.floor(t / 86400))
        if len(days) < 3:
            raise ValueError("Real data requires at least three UTC observation dates for separate splits")
        first = max(1, min(len(days) - 2, int(len(days) * .7)))
        second = max(first + 1, min(len(days) - 1, int(len(days) * .85)))
        cuts = [0, int(np.searchsorted(t, days[first] * 86400)),
                int(np.searchsorted(t, days[second] * 86400)), n]
    splits = {name: make_windows(t, raw, events, a, b)
              for name, a, b in zip(("train", "validation", "test"), cuts[:-1], cuts[1:])}
    for name, (_, y, _) in splits.items():
        if set(y) != {0, 1}:
            raise ValueError(f"{name}: both classes required; collect more event and normal observations")
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LogisticRegression
    import joblib
    scaler = MinMaxScaler().fit(raw[:cuts[1]])
    transformed = {k: (scaler.transform(x.reshape(-1, 10)).reshape(x.shape).astype(np.float32), y, a)
                   for k, (x, y, a) in splits.items()}
    out = new_directory(args.output)
    tf = tf_setup()
    xt, yt, _ = transformed["train"]; xv, yv, _ = transformed["validation"]
    model = tf.keras.Sequential([tf.keras.Input((30, 10)), tf.keras.layers.LSTM(32),
                                 tf.keras.layers.Dense(1, activation="sigmoid")])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    weight = {0: 1., 1: float(np.sum(yt == 0) / np.sum(yt == 1))}
    def dataset(x, y=None, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices(x if y is None else (x, y))
        if shuffle:
            ds = ds.shuffle(len(x), seed=42)
        options = tf.data.Options(); options.threading.private_threadpool_size = 2
        return ds.batch(64).with_options(options).prefetch(1)
    history = model.fit(dataset(xt, yt, True), validation_data=dataset(xv, yv), epochs=args.epochs, shuffle=False,
                        class_weight=weight, verbose=2,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
    baseline = LogisticRegression(C=.1, class_weight="balanced", max_iter=1000, random_state=42)
    baseline.fit(xt.reshape(len(xt), -1), yt)
    val_lstm = model.predict(dataset(xv), verbose=0).ravel()
    val_log = baseline.predict_proba(xv.reshape(len(xv), -1))[:, 1]
    thresholds = {"lstm_10": choose_threshold(yv, val_lstm, args.max_fpr),
                  "logistic_10": choose_threshold(yv, val_log, args.max_fpr), "recent_lightning": .5}
    # Persist policy before scoring the untouched chronological test partition.
    write_json(out / "validation_policy.json", {"thresholds": thresholds, "max_validation_fpr": args.max_fpr,
                                               "primary_model": "lstm_10", "selection_data": "validation only"})
    model.save(out / "lstm_10.keras")
    np.savez(out / "scaler.npz", scale=scaler.scale_, offset=scaler.min_, features=np.array(FEATURES))
    joblib.dump(baseline, out / "logistic_baseline.joblib")
    report = {}
    for name, (x, y, anchors) in transformed.items():
        scores = {"lstm_10": model.predict(dataset(x), verbose=0).ravel(),
                  "logistic_10": baseline.predict_proba(x.reshape(len(x), -1))[:, 1],
                  "recent_lightning": raw[anchors - 29 + np.arange(30)[:, None], -1].max(axis=0)}
        report[name] = {"window_count": len(y), "positive_windows": int(y.sum()), "models": {}}
        for key, score in scores.items():
            report[name]["models"][key] = {"threshold": thresholds[key],
                "windows": binary_metrics(y, score, thresholds[key]),
                "events": event_metrics(t, raw, anchors, events, score, thresholds[key])}
        with (out / f"{name}_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f); writer.writerow(["time_s", "label"] + list(scores))
            writer.writerows(zip(t[anchors], y, *scores.values()))
    write_json(out / "evaluation.json", report)
    write_json(out / "history.json", history.history)
    manifest = {"data_origin": source["data_origin"], "features": FEATURES, "sensor_schema": SENSORS,
                "step_seconds": 1, "lookback_seconds": 30, "forecast_seconds": 10,
                "threshold": thresholds["lstm_10"], "calibrated_probability": False,
                "primary_model": "lstm_10", "automatic_control_allowed": False,
                "production_ready": False, "usage": "RESEARCH_ONLY",
                "source_manifest": source, "csv_sha256": digest(args.csv), "events_sha256": digest(args.events),
                "artifacts": {p: digest(out / p) for p in ("lstm_10.keras", "scaler.npz")},
                "split_bounds_seconds": {k: [float(t[a]), float(t[b - 1])] for k, a, b in
                                         zip(splits, cuts[:-1], cuts[1:])},
                "limits": "Overlapping windows are correlated. Quiet-history events are not independently verified first strikes. No hardware validation."}
    write_json(out / "manifest.json", manifest)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(); ax.plot(history.history["loss"], label="train")
    ax.plot(history.history["val_loss"], label="validation"); ax.legend()
    ax.set(xlabel="epoch (zero-based)", ylabel="binary cross entropy", title="10-input LSTM")
    fig.savefig(out / "loss.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print(json.dumps({"output": str(out.resolve()), "test": report["test"], "data_origin": source["data_origin"]}))


def replay(args):
    t, raw, source = load_data(args.csv, args.manifest, require_labels=False)
    bundle = Path(args.model_dir)
    m = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if m.get("features") != FEATURES or m.get("lookback_seconds") != 30 or m.get("forecast_seconds") != 10:
        raise ValueError("Model contract mismatch")
    for name in ("lstm_10.keras", "scaler.npz"):
        if digest(bundle / name) != m["artifacts"][name]:
            raise ValueError("Model/scaler checksum mismatch")
    tf = tf_setup(); model = tf.keras.models.load_model(bundle / "lstm_10.keras", compile=False)
    if model.input_shape != (None, 30, 10):
        raise ValueError("Invalid model input")
    with np.load(bundle / "scaler.npz", allow_pickle=False) as coefficients:
        if list(coefficients["features"]) != FEATURES:
            raise ValueError("Normalization feature order mismatch")
        scale, offset = coefficients["scale"], coefficients["offset"]
    if scale.shape != (10,) or offset.shape != (10,) or not np.isfinite(scale).all() or not np.isfinite(offset).all():
        raise ValueError("Invalid normalization")
    @tf.function(input_signature=[tf.TensorSpec((1, 30, 10), tf.float32)])
    def predict(x):
        return model(x, training=False)
    output = Path(args.output)
    with output.open("x", encoding="utf-8") as f:
        start = 0
        for i, now in enumerate(t):
            if i and now != t[i - 1] + 1:
                start = i
            score = None; state = "WARMUP"
            if i - start >= 29:
                score = float(predict((raw[i - 29:i + 1] * scale + offset)[None]).numpy()[0, 0])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Nonfinite/out-of-range model score")
                state = "RESEARCH_ALERT" if score >= m["threshold"] else "RESEARCH_LOW_SCORE"
            f.write(json.dumps({"time_s": float(now), "state": state, "risk_score": score,
                                "threshold": m["threshold"], "score_is_calibrated_probability": False,
                                "input_features": 10, "input_origin": source["data_origin"],
                                "model_training_origin": m["data_origin"], "usage": "RESEARCH_ONLY",
                                "automatic_control_allowed": False, "hardware_command": None}) + "\n")
    print(f"Replay: {output.resolve()}")


def capture(args):
    # Accept only the documented provisional protocol; never guess vendor fields.
    out = new_directory(args.output)
    w = StreamWindow(SENSORS); origin = args.origin_seconds
    stats = {"valid_rows": 0, "rejected_frames": 0, "fault_ticks": 0, "completed": False}
    source = sys.stdin if args.frames == "-" else Path(args.frames).open(encoding="utf-8-sig")
    with (out / "aligned.csv").open("w", newline="", encoding="utf-8-sig") as aligned, \
            (out / "raw_frames.jsonl").open("w", encoding="utf-8") as log:
        writer = csv.writer(aligned); writer.writerow(HEADER)
        try:
            for line in source:
                log.write(line.rstrip("\n") + "\n"); log.flush()
                try:
                    wrapped = json.loads(line)
                    wire = wrapped["raw"] if isinstance(wrapped, dict) and "raw" in wrapped else line
                    if not isinstance(wire, str):
                        raise ValueError("Wrapped raw frame must be a string")
                    frame = parse_frame(wire.encode("utf-8"), w.schema)
                    if frame["type"] == "end":
                        stats["completed"] = True; break
                    if frame["type"] == "sample":
                        w.sample(frame); continue
                    _, fault, _ = w.tick(frame["t_s"])
                    if fault not in (None, "WARMUP", "DATA_GAP"):
                        stats["fault_ticks"] += 1; continue
                    writer.writerow([origin + frame["t_s"]] + list(w.rows[-1]) +
                                    [frame["t_s"] - w.latest[name][0] for name in FEATURES])
                    stats["valid_rows"] += 1
                except (ValueError, TypeError, KeyError, IndexError) as error:
                    stats["rejected_frames"] += 1
                    # Stop: rejected ordering could invalidate causal interpretation.
                    stats["error"] = str(error); break
        finally:
            if source is not sys.stdin:
                source.close()
            write_json(out / "capture_summary.json", stats)
    print(json.dumps(stats))
    if stats["rejected_frames"]:
        raise ValueError("Capture rejected invalid input; retained raw log for diagnosis")


def templates(args):
    out = new_directory(args.output)
    with (out / "sensors.csv").open("w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerow(HEADER)
    with (out / "events.csv").open("w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerow(["time_s", "event_id"])
    write_json(out / "data_manifest.json", {"data_origin": "real", "features": FEATURES,
        "step_seconds": 1, "site_id": "TODO site identifier", "time_reference": "TODO UTC Unix seconds",
        "sensor_mapping_source": "TODO vendor-confirmed channel/unit mapping",
        "independent_event_labels": False, "event_catalog_complete": False,
        "event_source": "TODO independent lightning observations and covered radius/type"})


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="mode", required=True)
    q = sub.add_parser("templates"); q.add_argument("--output", required=True)
    q = sub.add_parser("capture"); q.add_argument("--frames", required=True, help="NDJSON file or - for stdin")
    q.add_argument("--origin-seconds", type=int, default=0); q.add_argument("--output", required=True)
    for mode in ("validate", "train", "replay"):
        q = sub.add_parser(mode); q.add_argument("--csv", required=True); q.add_argument("--manifest", required=True)
        if mode in ("validate", "train"):
            q.add_argument("--events", required=True)
        if mode == "train":
            q.add_argument("--epochs", type=int, default=40); q.add_argument("--max-fpr", type=float, default=.01)
            q.add_argument("--output", required=True)
        if mode == "replay":
            q.add_argument("--model-dir", required=True); q.add_argument("--output", required=True)
    args = p.parse_args()
    if args.mode == "templates": templates(args)
    elif args.mode == "capture":
        if args.origin_seconds < 0:
            p.error("origin-seconds must be nonnegative")
        capture(args)
    elif args.mode == "validate":
        t, raw, source = load_data(args.csv, args.manifest)
        events = load_events(args.events, t)
        print(json.dumps({"rows": len(t), "features": raw.shape[1], "events": len(events),
                          "gaps": int(np.sum(np.diff(t) != 1)), "origin": source["data_origin"]}))
    elif args.mode == "train":
        if args.epochs < 1 or not 0 <= args.max_fpr <= 1:
            p.error("epochs >= 1 and 0 <= max-fpr <= 1 required")
        train(args)
    else: replay(args)


if __name__ == "__main__":
    main()
