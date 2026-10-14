"""Server-side ten-sensor processing using user device criteria and cumulative counts."""
import argparse
from collections import deque
from datetime import datetime
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
CRITERIA = json.loads((ROOT / "device_criteria.json").read_text(encoding="utf-8"))
RAW_FEATURES = ["ground_potential_a", "ground_potential_b", "leakage_current",
                "element_temperature_a", "element_temperature_b", "element_temperature_c",
                "relative_humidity", "ambient_temperature", "input_voltage", "lightning_count_total"]
MODEL_FEATURES = RAW_FEATURES[:-1] + ["lightning_activity_log1p"]
PERIODS = [.1, .1, 1., 60., 60., 60., 3600., 3600., 1., .1]
RANGES = [(0, 50), (0, 50), (.001, 1), (25, 150), (25, 150), (25, 150),
          (20, 90), (-20, 80), (0, 1500)]
TRANSFORM_VERSION = "count-delta-log1p-per-second-v2"


class DeviceProcessor:
    def __init__(self):
        self.last_time = None
        self.last_count = None
        self.rows = deque(maxlen=30)
        self.activity = deque()

    def clear(self):
        self.last_time = self.last_count = None
        self.rows.clear(); self.activity.clear()

    def process(self, frame):
        try:
            t = frame["time_s"]
            if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 or t != int(t):
                raise ValueError("time_s must be nonnegative integer seconds")
            values = [frame[k] for k in RAW_FEATURES]
            for j, v in enumerate(values):
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                    raise ValueError("Missing/nonfinite numeric sensor")
                if j < 9 and not RANGES[j][0] <= v <= RANGES[j][1]:
                    raise ValueError("Sensor outside declared acquisition range: " + RAW_FEATURES[j])
            count = values[-1]
            if count < 0 or count != int(count):
                raise ValueError("Cumulative lightning count must be a nonnegative integer")
            ages = frame["age_seconds"]
            for name, period in zip(RAW_FEATURES, PERIODS):
                age = ages[name]
                if isinstance(age, bool) or not isinstance(age, (int, float)) or not math.isfinite(age) or not 0 <= age <= 2 * period + 1e-6:
                    raise ValueError("Missing/stale observation: " + name)
            if self.last_time is not None and t <= self.last_time:
                raise ValueError("Duplicate/reversed timestamp")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            self.clear()
            safe_time = frame.get("time_s") if isinstance(frame, dict) else None
            if isinstance(safe_time, bool) or not isinstance(safe_time, (int, float)) or not math.isfinite(safe_time):
                safe_time = None
            return {"time_s": safe_time,
                    "data_state": "SENSOR_FAULT", "detail": str(error), "risk_score": None,
                    "model_state": "INFERENCE_BLOCKED", "hardware_command": None,
                    "automatic_control_allowed": False, "output_voltage_verified_zero": False}
        gap = self.last_time is not None and t != self.last_time + 1
        reset = self.last_count is not None and count < self.last_count
        initial = self.last_count is None or gap or reset
        delta = None if initial else int(count) - self.last_count
        if initial:
            self.rows.clear(); self.activity.clear()
        self.last_time, self.last_count = t, int(count)
        if delta is not None:
            self.activity.append((t, delta))
        while self.activity and self.activity[0][0] <= t - 300:
            self.activity.popleft()
        transformed = None if delta is None else math.log1p(delta)
        if transformed is not None:
            self.rows.append(np.asarray(values[:9] + [transformed], dtype=np.float32))
        equipment = {k: values[RAW_FEATURES.index(k)] >= limit
                     for k, limit in CRITERIA["equipment_thresholds"].items()}
        environment = {k: values[RAW_FEATURES.index(k)] >= limit
                       for k, limit in CRITERIA["environment_reference_values"].items()}
        thermal = {phase: {"temperature_C": frame["element_temperature_" + phase],
                           "above_45_C": equipment["element_temperature_" + phase],
                           "above_ambient_C": frame["element_temperature_" + phase] - frame["ambient_temperature"]}
                   for phase in ("a", "b", "c")}
        state = "COUNTER_RESET" if reset else "DATA_GAP" if gap else "COUNTER_BASELINE" if initial else "VALID"
        return {"time_s": t, "data_state": state,
                "any_user_criterion_exceeded": any(equipment.values()) or any(environment.values()),
                "equipment_state": "CRITERION_EXCEEDED" if any(equipment.values()) else "BELOW_CRITERIA",
                "equipment_flags": equipment, "environment_flags": environment, "thermal_channels": thermal,
                "raw_lightning_count_total": int(count), "new_lightning_count": delta,
                "lightning_activity_log1p": transformed,
                "lightning_count_observed_60s": sum(n for at, n in self.activity if at > t - 60),
                "lightning_count_observed_300s": sum(n for _, n in self.activity),
                "activity_history_observed_seconds": len(self.activity),
                "lightning_evidence": "COUNT_INCREASE_OBSERVED" if delta is not None and delta > 0 else "NO_NEW_COUNT_OBSERVED",
                "temperature_is_lightning_evidence": False,
                "model_ready": len(self.rows) == 30, "model_state": "COUNT_INPUT_MODEL_NOT_TRAINED",
                "risk_score": None, "model_features": MODEL_FEATURES,
                "model_input": self.rows[-1].tolist() if transformed is not None else None,
                "input_voltage_V": frame["input_voltage"], "input_voltage_limit": None,
                "simulation_conversion_requested": delta is not None and delta > 0,
                "simulation_target_output_voltage_V": CRITERIA["voltage_goal_V"],
                "measured_output_voltage_V": None, "output_voltage_verified_zero": False,
                "hardware_command": None, "automatic_control_allowed": False}


def sample_frames(case, seconds=150):
    base = [.8, .9, .0015, 35., 36., 34., 65., 23., 220., 1000]
    if case == "thermal_only": base[3] = 48.
    if case == "leakage_only": base[2] = .002
    if case == "environment_only": base[6:8] = [72., 27.]
    if case == "combined": base[:6] = [2.2, 2.1, .0023, 47., 46., 48.]
    count = 1000
    for t in range(seconds):
        if case in ("lightning_activity", "combined") and t in (40, 41, 75): count += 1
        if case == "counter_reset":
            if t == 50: count = 0
            if t == 70: count += 1
        frame = dict(zip(RAW_FEATURES, base)); frame["lightning_count_total"] = count
        frame["time_s"] = t
        frame["age_seconds"] = {name: float(t % period) if period >= 1 else 0.
                                for name, period in zip(RAW_FEATURES, PERIODS)}
        yield frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["demo", "stdin"], nargs="?", default="demo")
    p.add_argument("--output", type=Path)
    p.add_argument("--model-dir", type=Path, help="Optional newly trained count-v2 model, never a legacy H/L model")
    p.add_argument("--external-context", type=Path, help="Optional source-scoped UTC context JSON")
    p.add_argument("--weather-model-dir", type=Path, help="Optional separately validated public-weather research model")
    p.add_argument("--site-lat", type=float)
    p.add_argument("--site-lon", type=float)
    args = p.parse_args()
    out = args.output or ROOT / "device_server_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    model = None
    if args.model_dir:
        from train_device_count import CountPredictor
        model = CountPredictor(args.model_dir)
    from external_context import ExternalContext
    context_document = json.loads(args.external_context.read_text(encoding="utf-8")) if args.external_context else None
    context = ExternalContext(context_document, args.weather_model_dir)
    print("10-SENSOR SERVER | user deterioration criteria | cumulative lightning counts | NO HARDWARE COMMANDS")
    print("2mA = 0.002A; each element temperature threshold = 45C; voltage verification unavailable")
    cases = ("normal", "thermal_only", "leakage_only", "environment_only", "lightning_activity", "counter_reset", "combined")
    summaries = []
    with (out / "raw.jsonl").open("w", encoding="utf-8") as raw, (out / "status.jsonl").open("w", encoding="utf-8") as status:
        for case in cases if args.mode == "demo" else ("incoming",):
            processor = DeviceProcessor(); n = degraded = detected = resets = faults = 0
            frames = sample_frames(case) if args.mode == "demo" else (json.loads(line) for line in sys.stdin if line.strip())
            for frame in frames:
                raw.write(json.dumps({"case": case, "frame": frame}) + "\n"); raw.flush()
                result = processor.process(frame); result["case"] = case
                result["input_origin"] = "synthetic" if args.mode == "demo" else "unverified_external_stream"
                if model and result.get("model_ready"):
                    result.update(model.predict(np.stack(processor.rows)))
                if isinstance(frame, dict) and result["data_state"] != "SENSOR_FAULT":
                    result["external_context"] = context.assess(frame, args.site_lat, args.site_lon)
                status.write(json.dumps(result, allow_nan=False) + "\n"); status.flush()
                n += 1; degraded += result.get("any_user_criterion_exceeded", False)
                detected += result.get("new_lightning_count") or 0
                resets += result["data_state"] == "COUNTER_RESET"; faults += result["data_state"] == "SENSOR_FAULT"
            summary = {"case": case, "rows": n, "criterion_exceeded_rows": degraded,
                       "new_counts_observed": detected, "counter_resets": resets, "sensor_faults": faults}
            summaries.append(summary); print(json.dumps(summary))
    (out / "summary.json").write_text(json.dumps({"criteria": CRITERIA, "cases": summaries,
        "real_lightning_prediction_validated": False, "hardware_commands_sent": 0}, indent=2), encoding="utf-8")
    print(f"Results: {out.resolve()}")


if __name__ == "__main__": main()
