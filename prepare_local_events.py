"""Filter an independent geolocated event catalog; never infer CG from H/L."""
import argparse
import csv
import math
from pathlib import Path
from sensor10_pipeline import new_directory, write_json


def distance_km(lat1, lon1, lat2, lon2):
    for lat, lon in ((lat1, lon1), (lat2, lon2)):
        if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError("Invalid latitude/longitude")
    a, b = math.radians(lat1), math.radians(lat2)
    delta_lat, delta_lon = b - a, math.radians(lon2 - lon1)
    h = math.sin(delta_lat / 2) ** 2 + math.cos(a) * math.cos(b) * math.sin(delta_lon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(min(1., max(0., h))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", required=True); p.add_argument("--latitude", type=float, required=True)
    p.add_argument("--longitude", type=float, required=True); p.add_argument("--output", required=True)
    args = p.parse_args(); distance_km(args.latitude, args.longitude, args.latitude, args.longitude)
    selected = {}; ids = set(); total = 0; selected_count = 0
    with Path(args.catalog).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["time_s", "latitude", "longitude", "kind", "event_id"]:
            raise ValueError("Catalog needs time_s,latitude,longitude,kind,event_id")
        for r in reader:
            total += 1; t = float(r["time_s"])
            if not math.isfinite(t) or t < 0 or r["kind"] not in ("CG", "IC", "CC", "CA"):
                raise ValueError("Invalid event time/kind; unknown H/L types cannot be used as CG labels")
            if not r["event_id"] or r["event_id"] in ids:
                raise ValueError("Nonempty unique source event IDs required")
            ids.add(r["event_id"])
            d = distance_km(args.latitude, args.longitude, float(r["latitude"]), float(r["longitude"]))
            if r["kind"] == "CG" and d <= 10:
                selected.setdefault(t, []).append(r["event_id"]); selected_count += 1
    out = new_directory(args.output)
    with (out / "events.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["time_s", "event_id"])
        # Simultaneous source events have the same binary target; retain their IDs in metadata.
        w.writerows((t, f"time-group-{i}") for i, t in enumerate(sorted(selected)))
    write_json(out / "catalog_scope.json", {"site_latitude": args.latitude, "site_longitude": args.longitude,
        "event_kind": "CG", "event_radius_km": 10, "source_rows": total, "selected_source_events": selected_count,
        "time_groups": selected, "event_catalog_completeness_verified": False,
        "note": "Filtering cannot verify detection efficiency, timestamp accuracy or continuous observation coverage."})


if __name__ == "__main__": main()
