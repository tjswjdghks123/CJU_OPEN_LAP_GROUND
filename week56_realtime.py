"""Local TCP sensor demo. Protocol is provisional NDJSON, not the vendor protocol.

demo: sender + receiver; send: synthetic TCP server; receive: inference client.
Only simulated warning/breaker outputs are produced. No device commands.
"""
import argparse
from collections import deque
import csv
import hashlib
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import threading
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np


class Controller:
    def __init__(self, threshold=.5, trip_count=3):
        self.threshold, self.trip_count = threshold, trip_count
        self.count, self.opened = 0, False

    def step(self, probability=None, fault=None):
        if fault:
            self.count = 0
            return fault, True, self.opened
        risk = probability >= self.threshold
        self.count = self.count + 1 if risk else 0
        if self.count >= self.trip_count:
            self.opened = True
        return ("OPEN_LATCHED" if self.opened else "WARNING" if risk else "NORMAL",
                bool(risk or self.opened), self.opened)


def parse_frame(line, schema):
    if len(line) > 4096:
        raise ValueError("Frame exceeds 4096 bytes")
    frame = json.loads(line)
    if not isinstance(frame, dict):
        raise ValueError("Frame must be an object")
    kind = frame.get("type")
    if kind == "end":
        return frame
    if kind not in ("sample", "tick"):
        raise ValueError("Unknown frame type")
    t = frame.get("t_s")
    if isinstance(t, bool) or not isinstance(t, (float, int)) or not math.isfinite(t) or t < 0:
        raise ValueError("Invalid timestamp")
    if kind == "tick" and not float(t).is_integer():
        raise ValueError("Tick must represent an integer second")
    if kind == "sample":
        name, value = frame.get("sensor"), frame.get("value")
        if name not in schema or isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError("Unknown sensor or invalid value")
        _, _, low, high, _ = schema[name]
        if not low <= value <= high or (name == "lightning_sensor" and value not in (0, 1)):
            raise ValueError("Sensor value outside declared range")
    return frame


class StreamWindow:
    def __init__(self, sensor_schema):
        self.schema = {s[0]: s for s in sensor_schema}
        self.features = list(self.schema)
        self.latest = {}
        self.pulses = deque()
        self.rows = deque(maxlen=30)
        self.last_tick = None

    def sample(self, frame):
        name, t, value = frame["sensor"], frame["t_s"], frame["value"]
        if self.last_tick is not None and t <= self.last_tick:
            raise ValueError("Late sample arrived after its completed tick")
        if name in self.latest and t <= self.latest[name][0]:
            raise ValueError("Duplicate/out-of-order sensor timestamp")
        self.latest[name] = (t, value)
        if name == "lightning_sensor" and value == 1:
            self.pulses.append(t)

    def tick(self, t):
        gap = self.last_tick is not None and t != self.last_tick + 1
        if self.last_tick is not None and t <= self.last_tick:
            raise ValueError("Duplicate/out-of-order tick")
        self.last_tick = t
        if gap:
            self.rows.clear()
        issues, values = [], []
        for name, (_, period, _, _, _) in self.schema.items():
            sample = self.latest.get(name)
            if sample is None:
                issues.append(name + ":missing"); values.append(0.)
            else:
                observed, value = sample
                if observed > t + 1e-8:
                    raise ValueError("Future sample cannot enter a completed tick")
                if t - observed > period * 2 + 1e-8:
                    issues.append(name + ":stale")
                values.append(value)
        while self.pulses and self.pulses[0] <= t - 1:
            self.pulses.popleft()
        values[self.features.index("lightning_sensor")] = float(bool(self.pulses))
        if issues:
            self.rows.clear()
            return None, "SENSOR_FAULT", ";".join(issues)
        self.rows.append(np.asarray(values, dtype=np.float32))
        if gap:
            return None, "DATA_GAP", "Non-contiguous timeline; warmup restarted"
        if len(self.rows) < 30:
            return None, "WARMUP", f"{len(self.rows)}/30 valid seconds"
        return np.stack(self.rows), None, ""


class Journal:
    def __init__(self, directory, model_info=None):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.model_info = model_info or {}
        self.raw = (directory / "received_frames.jsonl").open("w", encoding="utf-8")
        self.status = (directory / "status.csv").open("w", newline="", encoding="utf-8-sig")
        self.writer = csv.writer(self.status)
        self.writer.writerow(["received_utc", "sensor_time_s", "risk_score", "state", "warning",
                              "simulated_breaker_open", "inference_ms", "detail",
                              "prediction_validated_for_device", "usage"])
        self.last_state = None
        self.counts = {}
        self.inferences = 0
        self.latencies = []

    def record(self, t, p, state, warning, opened, ms=0., detail=""):
        self.counts[state] = self.counts.get(state, 0) + 1
        if p is not None:
            self.inferences += 1; self.latencies.append(ms)
        self.writer.writerow([datetime.now(timezone.utc).isoformat(), t, p, state,
                              int(warning), int(opened), round(ms, 3), detail, False, "SYNTHETIC_DEMO"])
        self.status.flush()
        if state != self.last_state or (isinstance(t, (int, float)) and t % 10 == 0):
            score = "-" if p is None else f"{p:.3f}"
            print(f"SIMULATION t={t}  {state:16s} score={score}  warning={int(warning)}  simulated_open={int(opened)}  {detail}", flush=True)
            self.last_state = state

    def close(self, completed):
        summary = {"completed": completed, "state_counts": self.counts,
                   "inference_count": self.inferences,
                   "mean_inference_ms": float(np.mean(self.latencies)) if self.latencies else None,
                   "max_inference_ms": max(self.latencies, default=None),
                   "protocol": "provisional NDJSON v1; not vendor-compatible",
                   "prediction_validated_for_device": False, "hardware_commands_sent": 0,
                   "usage": "SYNTHETIC_DEMO",
                   "model": "synthetic training; actual lightning prediction unverified"}
        summary.update(self.model_info)
        self.raw.close(); self.status.close()
        (self.directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def receive(args, schema, ready=None):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    if args.bundle is not None:
        manifest = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
        for name in ("lstm_10.keras", "scaler.npz"):
            if hashlib.sha256((args.bundle / name).read_bytes()).hexdigest() != manifest["artifacts"][name]:
                raise ValueError("Bundle artifact checksum mismatch")
        model_path, scaler_path = args.bundle / "lstm_10.keras", args.bundle / "scaler.npz"
    else:
        model_path, scaler_path = args.model_dir / "week56_lstm.keras", args.model_dir / "scaler_train_only.npz"
    model = tf.keras.models.load_model(model_path, compile=False)
    coefficients = np.load(scaler_path, allow_pickle=False)
    features = [s[0] for s in schema]
    if list(coefficients["features"]) != features or model.input_shape != (None, 30, 10):
        raise ValueError("Feature order or model shape mismatch")
    scale, offset = coefficients["scale"], coefficients["offset"]
    if scale.shape != (10,) or offset.shape != (10,) or not np.isfinite(scale).all() or not np.isfinite(offset).all():
        raise ValueError("Invalid saved normalization coefficients")
    # Compile/warm up inference before opening the stream; don't consume data.
    @tf.function(input_signature=[tf.TensorSpec((1, 30, 10), tf.float32)])
    def predict(x):
        return model(x, training=False)
    predict(np.zeros((1, 30, 10), dtype=np.float32))
    journal = Journal(args.output_dir, {
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "model_path": str(model_path.resolve()), "feature_names": features,
        "decision_threshold": args.threshold,
        "model_training_origin": manifest["data_origin"] if args.bundle is not None else "synthetic"})
    controller = Controller(args.threshold, args.trip_count)
    completed, failures = False, 0
    if ready is not None:
        ready.set()
    try:
        while failures < args.max_retries:
            window = StreamWindow(schema)
            try:
                with socket.create_connection((args.host, args.port), timeout=3.) as client:
                    client.settimeout(args.timeout)
                    buffer = b""
                    last_tick_wall = time.monotonic()
                    while True:
                        chunk = client.recv(8192)
                        if time.monotonic() - last_tick_wall > args.timeout:
                            raise TimeoutError("No completed tick within watchdog interval")
                        if not chunk:
                            raise ConnectionError("Stream closed before end frame")
                        buffer += chunk
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            if not line:
                                continue
                            journal.raw.write(json.dumps({"received_utc": datetime.now(timezone.utc).isoformat(),
                                                          "raw": line.decode("utf-8", errors="replace")}) + "\n")
                            journal.raw.flush()
                            frame = parse_frame(line, window.schema)
                            if frame["type"] == "end":
                                completed = True
                                return
                            if frame["type"] == "sample":
                                window.sample(frame)
                                continue
                            t = frame["t_s"]
                            last_tick_wall = time.monotonic()
                            x, fault, detail = window.tick(t)
                            if x is None:
                                state, warning, opened = controller.step(fault=fault)
                                # Warmup is an unready state, not a risk warning.
                                warning = False if fault == "WARMUP" and not opened else warning
                                journal.record(t, None, state, warning, opened, detail=detail)
                            else:
                                start = time.perf_counter()
                                scaled = x.copy(); scaled *= scale; scaled += offset
                                try:
                                    p = float(predict(scaled[None]).numpy()[0, 0])
                                    if not math.isfinite(p) or not 0 <= p <= 1:
                                        raise RuntimeError("Invalid model output")
                                except Exception as error:
                                    state, warning, opened = controller.step(fault="MODEL_FAULT")
                                    journal.record(t, None, state, warning, opened, detail=str(error))
                                    raise
                                ms = (time.perf_counter() - start) * 1000
                                state, warning, opened = controller.step(p)
                                journal.record(t, p, state, warning, opened, ms)
                        if len(buffer) > 4096:
                            raise ValueError("Unterminated/oversized frame")
            except (OSError, ValueError, ConnectionError) as error:
                failures += 1
                state, warning, opened = controller.step(fault="CONNECTION_FAULT")
                journal.record("", None, state, warning, opened, detail=str(error))
                time.sleep(.2)
        raise RuntimeError("Connection retry limit reached; see status.csv")
    finally:
        journal.close(completed)


def send(args, schema, ready=None, receiver_ready=None, errors=None):
    try:
        records = np.load(args.model_dir / "sensor_measurements.npz")
        start, stop = args.start, args.start + args.duration
        events = []
        initial = []
        for name, period, *_ in schema:
            times, values = records[name + "_time_s"], records[name + "_value"]
            if stop > times[-1] + 1:
                raise ValueError("Demo duration exceeds saved synthetic measurements")
            index = np.searchsorted(times, start, side="right") - 1
            if index < 0:
                raise ValueError("No initial observation for sensor")
            initial.append({"type": "sample", "t_s": float(times[index]), "sensor": name, "value": float(values[index])})
            end = np.searchsorted(times, stop, side="left")
            for t, value in zip(times[index + 1:end], values[index + 1:end]):
                events.append({"type": "sample", "t_s": float(t), "sensor": name, "value": float(value)})
        for tick in range(start, stop):
            events.append({"type": "tick", "t_s": tick})
        events.sort(key=lambda e: (round(e["t_s"], 8), e["type"] == "tick"))
        # Sample frames are sent before the tick closing the same timestamp.
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((args.host, args.port)); server.listen(1); server.settimeout(45.)
            if ready is not None:
                ready.set()
            if receiver_ready is not None and not receiver_ready.wait(40.):
                raise RuntimeError("Receiver initialization timeout")
            client, _ = server.accept()
            with client:
                for frame in initial:
                    client.sendall((json.dumps(frame) + "\n").encode())
                origin = time.monotonic()
                disconnected = False
                for frame in events:
                    relative = frame["t_s"] - start
                    if args.disconnect_at is not None and not disconnected and relative >= args.disconnect_at:
                        client.close()
                        client, _ = server.accept()
                        disconnected = True
                        # Send historical last-known values to restart the buffer.
                        for name, *_ in schema:
                            times, values = records[name + "_time_s"], records[name + "_value"]
                            index = np.searchsorted(times, frame["t_s"], side="left") - 1
                            snap = {"type": "sample", "t_s": float(times[index]), "sensor": name, "value": float(values[index])}
                            client.sendall((json.dumps(snap) + "\n").encode())
                    if (frame["type"] == "sample" and frame["sensor"] == args.drop_sensor
                            and args.drop_start <= relative < args.drop_end):
                        continue
                    delay = relative / args.speed - (time.monotonic() - origin)
                    if delay > 0:
                        time.sleep(delay)
                    client.sendall((json.dumps(frame) + "\n").encode())
                client.sendall(b'{"type":"end"}\n')
                client.close()
    except Exception as error:
        if errors is not None:
            errors.append(error)
        else:
            raise
    finally:
        if ready is not None:
            ready.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["demo", "send", "receive"], nargs="?", default="demo")
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent / "results_week56")
    parser.add_argument("--bundle", type=Path, help="Validated ten-input bundle; model-dir still supplies synthetic sensor records")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4001)
    parser.add_argument("--duration", type=int, default=160)
    parser.add_argument("--start", type=int, default=10200)
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--threshold", type=float, help="Demo override; otherwise uses bundle validation policy or legacy 0.5")
    parser.add_argument("--trip-count", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=3.)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--drop-sensor")
    parser.add_argument("--drop-start", type=float, default=70.)
    parser.add_argument("--drop-end", type=float, default=76.)
    parser.add_argument("--disconnect-at", type=float)
    args = parser.parse_args()
    bundle_metadata = None
    if args.bundle is not None:
        bundle_metadata = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
    if args.threshold is None:
        args.threshold = bundle_metadata["threshold"] if bundle_metadata is not None else .5
    if (not math.isfinite(args.speed) or args.speed <= 0 or args.duration < 30 or args.start < 0
            or not 1 <= args.port <= 65535 or not math.isfinite(args.threshold) or not 0 < args.threshold <= 1.000001 or args.trip_count < 1
            or not math.isfinite(args.timeout) or args.timeout <= 0 or args.max_retries < 1):
        parser.error("Invalid duration/start/speed/port/threshold/trip-count/timeout/retry setting")
    metadata = json.loads((args.model_dir / "metrics.json").read_text(encoding="utf-8"))
    if bundle_metadata is not None:
        if (bundle_metadata.get("features") != metadata["features"]
                or bundle_metadata.get("sensor_schema") != metadata["sensor_schema"]
                or bundle_metadata.get("lookback_seconds") != 30
                or bundle_metadata.get("forecast_seconds") != 10):
            parser.error("Bundle feature/schema/time contract mismatch")
    schema = metadata["sensor_schema"]
    if args.drop_sensor is not None and args.drop_sensor not in metadata["features"]:
        parser.error("Unknown drop-sensor")
    if args.drop_sensor is not None and not 0 <= args.drop_start < args.drop_end <= args.duration:
        parser.error("Invalid drop interval")
    if args.disconnect_at is not None and not 0 < args.disconnect_at < args.duration:
        parser.error("disconnect-at must be inside the demo")
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.output_dir = Path(__file__).resolve().parent / "realtime_runs" / stamp
    print("SYNTHETIC TCP DEMO | provisional protocol | simulated outputs only", flush=True)
    print(f"PRIMARY PROJECT MODEL: 10-sensor LSTM | input=(30, 10) | forecast=10 seconds", flush=True)
    print("Inputs: " + ", ".join(metadata["features"]), flush=True)
    print(f"Model: {args.bundle or args.model_dir} | threshold={args.threshold:.6f}", flush=True)
    if args.mode == "send":
        send(args, schema)
    elif args.mode == "receive":
        receive(args, schema)
    else:
        ready, receiver_ready, errors = threading.Event(), threading.Event(), []
        thread = threading.Thread(target=send, args=(args, schema, ready, receiver_ready, errors), daemon=True)
        thread.start()
        if not ready.wait(5.) or errors:
            raise RuntimeError(f"Sender startup failed: {errors}")
        receive(args, schema, receiver_ready)
        thread.join(timeout=5.)
        if errors or thread.is_alive():
            raise RuntimeError(f"Sender did not complete: {errors}")
    if args.mode != "send":
        print(f"Logs: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
