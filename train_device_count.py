"""Train a NEW count-v2 model; never reuse legacy binary-sensor weights."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from device_server import DeviceProcessor, RAW_FEATURES, MODEL_FEATURES, TRANSFORM_VERSION
from sensor10_pipeline import (tf_setup, make_windows, choose_threshold, binary_metrics,
                               digest, new_directory, write_json)


class CountPredictor:
    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (self.manifest.get("transform_version") != TRANSFORM_VERSION
                or self.manifest.get("model_features") != MODEL_FEATURES
                or self.manifest.get("raw_features") != RAW_FEATURES
                or self.manifest.get("score_decimals") != 4):
            raise ValueError("Count-v2 model required: legacy H/L weights are incompatible")
        for name in ("count_model.keras", "scaler.npz"):
            if digest(directory / name) != self.manifest["artifacts"][name]:
                raise ValueError("Model/scaler checksum mismatch")
        with np.load(directory / "scaler.npz", allow_pickle=False) as co:
            if list(co["features"]) != MODEL_FEATURES: raise ValueError("Scaler feature order mismatch")
            self.scale, self.offset = co["scale"], co["offset"]
        if (self.scale.shape != (10,) or self.offset.shape != (10,)
                or not np.isfinite(self.scale).all() or not np.isfinite(self.offset).all()):
            raise ValueError("Invalid Min-Max coefficients")
        tf = tf_setup()
        self.model = tf.keras.models.load_model(directory / "count_model.keras", compile=False)
        if self.model.input_shape != (None, 30, 10): raise ValueError("Wrong Time-Step/model input")
        @tf.function(input_signature=[tf.TensorSpec((1, 30, 10), tf.float32)])
        def infer(x): return self.model(x, training=False)
        self.infer = infer

    def predict(self, rows):
        p = round(float(self.infer((rows * self.scale + self.offset)[None]).numpy()[0, 0]), 4)
        if not np.isfinite(p) or not 0 <= p <= 1: raise ValueError("Invalid model score")
        return {"risk_score": p, "model_state": "RESEARCH_ALERT" if p >= self.manifest["threshold"] else "RESEARCH_LOW_SCORE",
                "model_training_origin": self.manifest["data_origin"], "score_is_calibrated_probability": False,
                "forecast_seconds": 10, "automatic_control_allowed": False}

    def predict_many(self, rows, batch_size=128):
        """Frozen-model offline inference; same scaler and rounding as streaming."""
        rows = np.asarray(rows, dtype=np.float32)
        if rows.ndim != 3 or rows.shape[1:] != (30, 10) or not np.isfinite(rows).all():
            raise ValueError('Expected finite windows of shape (N, 30, 10)')
        if batch_size < 1: raise ValueError('batch_size must be positive')
        scores = []
        for at in range(0, len(rows), batch_size):
            x = rows[at:at+batch_size] * self.scale + self.offset
            scores.extend(self.model(x, training=False).numpy().ravel().tolist())
        scores = np.round(np.asarray(scores, dtype=float), 4)
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError('Invalid model scores')
        return scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("frames", "events", "manifest", "output"): p.add_argument("--" + name, required=True)
    p.add_argument("--epochs", type=int, default=40); p.add_argument("--allow-synthetic", action="store_true")
    args = p.parse_args()
    if args.epochs < 1: p.error("epochs must be positive")
    source = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if source.get("raw_features") != RAW_FEATURES: raise ValueError("Cumulative-count schema required")
    if source.get("data_origin") == "synthetic":
        if not args.allow_synthetic: raise ValueError("Synthetic training requires --allow-synthetic")
    elif source.get("data_origin") == "real":
        if (source.get("independent_event_labels") is not True or source.get("event_catalog_complete") is not True
                or not source.get("event_source") or str(source["event_source"]).startswith("TODO")
                or source.get("sensor_mapping_confirmed") is not True or not source.get("event_scope")):
            raise ValueError("Real training requires independent complete event labels, scope and verified channel mapping")
    else: raise ValueError("Declare real/synthetic provenance")
    processor = DeviceProcessor(); times = []; rows = []
    with Path(args.frames).open(encoding="utf-8-sig") as f:
        for line in f:
            frame = json.loads(line); result = processor.process(frame)
            if result["data_state"] == "SENSOR_FAULT": raise ValueError(result["detail"])
            if result.get("model_input") is not None:
                times.append(result["time_s"]); rows.append(result["model_input"])
    t = np.asarray(times); raw = np.asarray(rows, dtype=np.float32)
    if len(t) < 1000 or np.any(np.diff(t) <= 0): raise ValueError("Need >=1000 strictly ordered usable observations")
    from sensor10_pipeline import load_events
    events = load_events(args.events, t)
    n = len(t); cuts = [0, int(n * .7), int(n * .85), n]
    if source["data_origin"] == "real":
        days = np.unique(t // 86400)
        if len(days) < 3: raise ValueError("Need at least three separate UTC dates")
        a = max(1, min(len(days) - 2, int(len(days) * .7)))
        b = max(a + 1, min(len(days) - 1, int(len(days) * .85)))
        cuts = [0, int(np.searchsorted(t, days[a] * 86400)), int(np.searchsorted(t, days[b] * 86400)), n]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler().fit(raw[:cuts[1]])
    scaled = scaler.transform(raw).astype(np.float32)
    splits = {name: make_windows(t, scaled, events, a, b) for name, a, b in
              zip(("train", "validation", "test"), cuts[:-1], cuts[1:])}
    if any(set(y) != {0, 1} for _, y, _ in splits.values()): raise ValueError("Both classes required in each split")
    tf = tf_setup()
    model = tf.keras.Sequential([tf.keras.Input((30, 10)), tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32), tf.keras.layers.Dense(16, activation="relu"), tf.keras.layers.Dense(1, activation="sigmoid")])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")])
    def ds(x, y, shuffle=False):
        data = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle: data = data.shuffle(len(x), seed=42)
        opt = tf.data.Options(); opt.threading.private_threadpool_size = 2
        return data.batch(64).with_options(opt).prefetch(1)
    xt, yt, _ = splits["train"]; xv, yv, _ = splits["validation"]
    history = model.fit(ds(xt, yt, True), validation_data=ds(xv, yv), epochs=args.epochs, shuffle=False,
        class_weight={0: 1., 1: float(np.sum(yt == 0) / np.sum(yt == 1))}, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=5, restore_best_weights=True)])
    def score(x):
        values = np.concatenate([model(x[i:i + 128], training=False).numpy().ravel() for i in range(0, len(x), 128)])
        return np.round(values.astype(np.float64), 4)
    val_scores = score(xv)
    threshold = choose_threshold(yv, val_scores, .01)
    if binary_metrics(yv, val_scores, threshold)["tp"] == 0:
        threshold = 1.0001  # Explicit no-alert policy; do not exploit floating-point noise.
    out = new_directory(args.output)
    write_json(out / "validation_policy.json", {"threshold": threshold, "selection_data": "validation only", "max_fpr": .01})
    model.save(out / "count_model.keras")
    np.savez(out / "scaler.npz", scale=scaler.scale_, offset=scaler.min_, features=np.array(MODEL_FEATURES))
    report = {}
    for name, (x, y, anchors) in splits.items():
        scores = score(x); report[name] = binary_metrics(y, scores, threshold)
        with (out / f"{name}_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f); w.writerow(["time_s", "label", "score"]); w.writerows(zip(t[anchors], y, scores))
    write_json(out / "evaluation.json", report); write_json(out / "history.json", history.history)
    write_json(out / "manifest.json", {"raw_features": RAW_FEATURES, "model_features": MODEL_FEATURES,
        "transform_version": TRANSFORM_VERSION, "threshold": threshold, "score_decimals": 4, "lookback_seconds": 30,
        "forecast_seconds": 10, "target": "independent_event_in_next_10s; not validated first-strike prediction",
        "data_origin": source["data_origin"], "source_manifest": source,
        "criteria_are_lightning_labels": False, "real_lightning_prediction_validated": False,
        "production_ready": False, "automatic_control_allowed": False,
        "artifacts": {name: digest(out / name) for name in ("count_model.keras", "scaler.npz")},
        "frames_sha256": digest(args.frames), "events_sha256": digest(args.events)})
    print(out.resolve())


if __name__ == "__main__": main()
