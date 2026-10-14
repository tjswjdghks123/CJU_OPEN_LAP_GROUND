"""Invented teaching patterns only; NOT physical lightning precursor evidence."""
import csv
import json
from pathlib import Path
import numpy as np
from device_server import RAW_FEATURES, PERIODS

def episode(seed, kind, seconds=180, start=0):
    rng = np.random.default_rng(seed)
    event = int(rng.integers(115, 140)) if kind == 'synthetic_precursor' else None
    count = 1000
    base = np.array([.8, .85, .0014, 34., 35., 33., 63., 23., 220.])
    base[:2] += rng.uniform(-.12, .12, 2)
    base[3:6] += rng.uniform(-3, 3, 3)
    base[6:8] += rng.uniform(-4, 4, 2)
    temperature = base[3:6].copy()
    for t in range(seconds):
        x = base.copy()
        x[:2] += rng.normal(0, .025, 2)
        x[2] += rng.normal(0, .00003)
        x[8] += rng.normal(0, 1.3)
        if t % 60 == 0:
            temperature = base[3:6] + rng.normal(0, .4, 3)
            if kind == 'thermal_only': temperature += [18, 15, 12]
        x[3:6] = temperature
        if kind == 'leakage_only': x[2] += .0011
        if kind == 'voltage_step' and 65 <= t < 135: x[:2] += 1.6
        if event is not None and event - 35 <= t < event:
            # Deliberately invented co-ramp, not a scientifically established signal.
            ramp = (t - event + 35) / 35
            x[:2] += ramp * np.array([2.1, 1.9])
            x[2] += ramp * .0016
        if t == event: count += 1
        frame = dict(zip(RAW_FEATURES, x.tolist() + [count]))
        frame['time_s'] = start + t
        frame['age_seconds'] = {k: float(t % p) if p >= 1 else 0. for k,p in zip(RAW_FEATURES, PERIODS)}
        yield frame, (start + event if event is not None else None)

def main():
    root = Path(__file__).resolve().parent
    out = root / 'showcase_training_data'
    out.mkdir(exist_ok=True)
    kinds = ['normal', 'thermal_only', 'leakage_only', 'synthetic_precursor', 'voltage_step', 'synthetic_precursor']
    events = []
    with (out/'frames.jsonl').open('w', encoding='utf-8') as f:
        for i in range(72):
            for frame, event in episode(12000+i, kinds[i % len(kinds)], start=i*1000):
                f.write(json.dumps(frame) + '\n')
            if event is not None: events.append((event, f'invented-{i}'))
    with (out/'events.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['time_s','event_id']); w.writerows(events)
    (out/'manifest.json').write_text(json.dumps({'raw_features':RAW_FEATURES,'data_origin':'synthetic',
        'seed':12000,'episodes':72,'pattern':'invented 35-second joint ground-voltage/leakage ramp',
        'physical_precursor_validated':False,'event_labels':'invented events; never measured lightning',
        'slow_sensors':'temperatures held 60s; humidity/ambient held throughout each episode'},indent=2),encoding='utf-8')
    print(out)
if __name__ == '__main__': main()
