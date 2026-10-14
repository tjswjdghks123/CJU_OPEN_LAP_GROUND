import copy
import math
import unittest
from device_server import DeviceProcessor, sample_frames


class DeviceTests(unittest.TestCase):
    def frame(self, t=0):
        frame = next(sample_frames("normal")); frame["time_s"] = t
        return frame

    def test_exact_two_milliampere_and_ground_boundaries(self):
        f = self.frame(); f["leakage_current"] = .001999
        self.assertFalse(DeviceProcessor().process(f)["equipment_flags"]["leakage_current"])
        f["leakage_current"] = .002; f["ground_potential_a"] = 2
        r = DeviceProcessor().process(f)
        self.assertTrue(r["equipment_flags"]["leakage_current"])
        self.assertTrue(r["equipment_flags"]["ground_potential_a"])
        self.assertFalse(r["equipment_flags"]["ground_potential_b"])

    def test_independent_thermal_channels_are_not_lightning(self):
        f = self.frame(); f["element_temperature_b"] = 45
        r = DeviceProcessor().process(f)
        self.assertTrue(r["thermal_channels"]["b"]["above_45_C"])
        self.assertFalse(r["thermal_channels"]["a"]["above_45_C"])
        self.assertFalse(r["temperature_is_lightning_evidence"])
        self.assertFalse(r["simulation_conversion_requested"])
        self.assertIsNone(r["risk_score"])

    def test_counter_baseline_difference_transform_and_reset(self):
        p = DeviceProcessor(); f = self.frame()
        self.assertIsNone(p.process(f)["new_lightning_count"])
        f["time_s"] = 1; f["lightning_count_total"] += 2
        r = p.process(f); self.assertEqual(r["new_lightning_count"], 2)
        self.assertAlmostEqual(r["lightning_activity_log1p"], math.log1p(2))
        f["time_s"] = 2
        self.assertEqual(p.process(f)["new_lightning_count"], 0)
        f["time_s"] = 3; f["lightning_count_total"] = 0
        r = p.process(f); self.assertEqual(r["data_state"], "COUNTER_RESET")
        self.assertIsNone(r["new_lightning_count"]); self.assertFalse(r["model_ready"])

    def test_environment_and_voltage_have_separate_meaning(self):
        f = self.frame(); f["relative_humidity"] = 70; f["ambient_temperature"] = 25
        r = DeviceProcessor().process(f)
        self.assertTrue(all(r["environment_flags"].values()))
        self.assertTrue(r["any_user_criterion_exceeded"])
        self.assertEqual(r["equipment_state"], "BELOW_CRITERIA")
        self.assertIsNone(r["input_voltage_limit"])
        self.assertEqual(r["input_voltage_V"], 220)

    def test_voltage_zero_is_unverified_target_even_after_detection(self):
        p = DeviceProcessor(); f = self.frame(); p.process(f)
        f["time_s"] = 1; f["lightning_count_total"] += 1
        r = p.process(f)
        self.assertTrue(r["simulation_conversion_requested"])
        self.assertEqual(r["simulation_target_output_voltage_V"], 0)
        self.assertIsNone(r["measured_output_voltage_V"])
        self.assertFalse(r["output_voltage_verified_zero"])
        self.assertIsNone(r["hardware_command"])

    def test_stale_missing_nan_and_gap_block_predictions(self):
        for mutate in (lambda f: f.pop("leakage_current"),
                       lambda f: f.update(time_s=float("nan")),
                       lambda f: f["age_seconds"].update(ground_potential_a=1)):
            p = DeviceProcessor(); f = self.frame(); mutate(f)
            r = p.process(f); self.assertEqual(r["data_state"], "SENSOR_FAULT")
            self.assertIsNone(r["risk_score"])
        p = DeviceProcessor(); p.process(self.frame())
        self.assertEqual(p.process(self.frame(2))["data_state"], "DATA_GAP")

    def test_time_step_has_thirty_count_transformed_rows(self):
        p = DeviceProcessor()
        for f in sample_frames("normal", 30): r = p.process(f)
        self.assertFalse(r["model_ready"])
        r = p.process(self.frame(30))
        self.assertTrue(r["model_ready"])
        self.assertEqual(len(p.rows), 30)
        self.assertEqual(len(r["model_input"]), 10)
        self.assertEqual(r["model_features"][-1], "lightning_activity_log1p")

    def test_legacy_binary_model_is_not_accepted(self):
        from pathlib import Path
        from unittest.mock import patch
        from train_device_count import CountPredictor
        # Reject the old schema without distributing obsolete model weights.
        with patch.object(Path, "read_text", return_value='{"transform_version":"legacy-HL"}'):
            with self.assertRaisesRegex(ValueError, "legacy H/L"):
                CountPredictor(Path("unused-legacy-fixture"))


if __name__ == "__main__": unittest.main()
