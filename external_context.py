"""Join optional context by verified site, UTC observation AND availability times.

No arbitrary probability averaging, no foreign-site device imputation.
"""
import json
import math
from pathlib import Path
import numpy as np
from prepare_local_events import distance_km
from sensor10_pipeline import digest

WEATHER_FIELDS = ["pressure_hPa", "air_temperature_C", "relative_humidity_percent", "wind_speed_m_s",
                  "wind_gust_m_s", "precipitation_mm", "solar_radiation_W_m2", "lightning_count_5min"]


def available(record, asof):
    if not isinstance(record, dict): return False
    observed, known = record.get("time_s"), record.get("available_time_s")
    return (not isinstance(observed, bool) and not isinstance(known, bool)
            and isinstance(observed, (int, float)) and isinstance(known, (int, float))
            and math.isfinite(observed) and math.isfinite(known) and observed <= known <= asof)


class ExternalContext:
    def __init__(self, document=None, weather_model_dir=None):
        self.document = document or {}
        if not isinstance(self.document, dict): raise ValueError("External context must be an object")
        self.model = self.manifest = None
        if weather_model_dir is not None:
            import joblib
            directory = Path(weather_model_dir)
            self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if self.manifest.get("transform_version") != "external-weather8-causal-v1":
                raise ValueError("Unexpected external weather model")
            file = directory / self.manifest["selected_model"]
            if digest(file) != self.manifest["sha256"]: raise ValueError("External weather model checksum mismatch")
            self.model = joblib.load(file)

    def assess(self, frame, site_lat=None, site_lon=None):
        result = {"weather": {"state": "NOT_CONFIGURED"}, "nearby_lightning": {"state": "NOT_CONFIGURED"},
                  "atmospheric_electric_field": {"state": "NOT_CONFIGURED"},
                  "fusion_risk_score": None, "fusion_model_state": "JOINT_DEVICE_CONTEXT_MODEL_NOT_TRAINED",
                  "automatic_control_allowed": False}
        if site_lat is None or site_lon is None:
            result["join_state"] = "SITE_NOT_CONFIGURED"; return result
        try: distance_km(site_lat, site_lon, site_lat, site_lon)
        except (TypeError, ValueError):
            result["join_state"] = "INVALID_SITE"; return result
        if frame.get("time_basis") != "UTC_UNIX_SECONDS":
            result["join_state"] = "DEVICE_UTC_NOT_VERIFIED"; return result
        asof = frame.get("time_s")
        if not isinstance(asof, (int, float)) or not math.isfinite(asof):
            result["join_state"] = "INVALID_TIME"; return result
        result["join_state"] = "SCOPED_CONTEXT_ONLY"
        weather = self.document.get("weather")
        if weather:
            try:
                if weather.get("time_basis") != "UTC_UNIX_SECONDS": raise ValueError("SOURCE_UTC_NOT_VERIFIED")
                if distance_km(site_lat, site_lon, weather["latitude"], weather["longitude"]) > 5:
                    raise ValueError("WEATHER_SITE_TOO_FAR")
                eligible = sorted((r for r in weather["records"] if available(r, asof)), key=lambda r: r["time_s"])
                if len(eligible) < 30: raise ValueError("NOT_ENOUGH_AVAILABLE_HISTORY")
                history = eligible[-30:]; times = np.array([r["time_s"] for r in history])
                if not (np.diff(times) == 300).all(): raise ValueError("WEATHER_TIME_GAP")
                if not 0 <= asof - times[-1] <= 600: raise ValueError("STALE_WEATHER")
                values = np.array([[r[k] for k in WEATHER_FIELDS] for r in history], dtype=np.float32)
                if (not np.isfinite(values).all() or not ((values[:, 0] >= 500) & (values[:, 0] <= 1100)).all()
                        or not ((values[:, 2] >= 0) & (values[:, 2] <= 100)).all()
                        or not ((values[:, 1] >= -60) & (values[:, 1] <= 80)).all()
                        or (values[:, 3:6] < 0).any() or (values[:, 3:5] > 100).any() or (values[:, 7] < 0).any()):
                    raise ValueError("INVALID_WEATHER_VALUES")
                values[:, 7] = np.log1p(values[:, 7])
                result["weather"] = {"state": "AVAILABLE_RESEARCH_CONTEXT", "source_id": weather.get("source_id"),
                    "forecast_start_time_s": float(times[-1]), "forecast_end_time_s": float(times[-1] + 1800),
                    "event_kind": "unclassified_sensor_activity", "score_is_calibrated_probability": False,
                    "research_score": None, "validated_for_device": False}
                if self.model is not None:
                    from external_weather_model import engineer
                    features = engineer(values[None], np.array([int(times[-1] // 60)], dtype="datetime64[m]"))
                    score = float(self.model.predict_proba(features)[0, 1])
                    if not math.isfinite(score) or not 0 <= score <= 1: raise ValueError("INVALID_MODEL_SCORE")
                    result["weather"].update(research_score=score, validation_threshold=self.manifest["threshold"])
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                result["weather"] = {"state": "UNAVAILABLE", "reason": str(error), "research_score": None}
        lightning = self.document.get("lightning")
        if lightning:
            try:
                west, south, east, north = lightning["coverage_bbox"]
                if not west <= site_lon <= east or not south <= site_lat <= north: raise ValueError("OUTSIDE_SOURCE_COVERAGE")
                if lightning.get("coverage_verified") is not True: raise ValueError("COVERAGE_NOT_VERIFIED")
                if not available(lightning, asof) or lightning["time_s"] < asof - 60:
                    raise ValueError("FEED_NOT_AVAILABLE_OR_STALE")
                if lightning["observed_from_s"] > asof - 1800: raise ValueError("INCOMPLETE_HISTORY")
                nearby = []
                for r in lightning["events"]:
                    if not available(r, asof) or not asof - 1800 < r["time_s"] <= asof: continue
                    d = distance_km(site_lat, site_lon, r["latitude"], r["longitude"])
                    if d <= 40: nearby.append((r, d))
                result["nearby_lightning"] = {"state": "AVAILABLE_OBSERVATIONS", "source_id": lightning.get("source_id"),
                    "CG_events_10km_30min": sum(r.get("kind") == "CG" and d <= 10 for r, d in nearby),
                    "all_events_40km_30min": len(nearby),
                    "nearest_observed_distance_km": min((d for _, d in nearby), default=None),
                    "observation_is_forecast": False}
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                result["nearby_lightning"] = {"state": "UNAVAILABLE", "reason": str(error)}
        field = self.document.get("electric_field")
        if field:
            try:
                if field.get("unit") != "V/m": raise ValueError("ATMOSPHERIC_FIELD_UNIT_REQUIRED_NOT_GROUND_VOLTAGE")
                if not available(field, asof) or not 0 <= asof - field["time_s"] <= 60: raise ValueError("FIELD_STALE_OR_UNAVAILABLE")
                if distance_km(site_lat, site_lon, field["latitude"], field["longitude"]) > 5: raise ValueError("FIELD_SITE_TOO_FAR")
                if isinstance(field["value"], bool) or not math.isfinite(field["value"]): raise ValueError("INVALID_FIELD_VALUE")
                result["atmospheric_electric_field"] = {"state": "AVAILABLE_MEASUREMENT", "value": field["value"],
                    "unit": "V/m", "lightning_probability": None}
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                result["atmospheric_electric_field"] = {"state": "UNAVAILABLE", "reason": str(error)}
        return result
