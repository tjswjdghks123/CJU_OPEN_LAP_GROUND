import unittest
import numpy as np
from manual_sensor_server import InputSession
from device_server import RAW_FEATURES

class FakePredictor:
    manifest={'threshold':.8}
    def predict_many(self, windows):
        self.windows=windows.copy()
        return np.full(len(windows),.4)

def frame(**overrides):
    f=dict(zip(RAW_FEATURES,[.8,.85,.0014,34,35,33,63,23,220,1000]))
    f.update(overrides)
    return f

class ManualTests(unittest.TestCase):
    def setUp(self):
        self.model=FakePredictor()
        self.session=InputSession(self.model)

    def test_warmup_and_exact_window(self):
        r=self.session.append([frame() for _ in range(31)])
        self.assertIsNone(r[29]['score'])
        self.assertEqual(r[30]['score'],.4)
        self.assertEqual(self.model.windows.shape,(1,30,10))
        self.assertTrue(np.all(self.model.windows[:,:,9]==0))

    def test_invalid_batch_is_atomic(self):
        self.session.append([frame()])
        with self.assertRaises(ValueError):
            self.session.append([frame(),frame(leakage_current=float('nan'))])
        self.assertEqual(self.session.processor.last_time,0)

    def test_duplicate_timestamp_rejected(self):
        self.session.append([frame(time_s=0)])
        with self.assertRaises(ValueError):
            self.session.append([frame(time_s=0)])

    def test_gap_and_counter_reset_rewarm(self):
        self.session.append([frame() for _ in range(31)])
        r=self.session.append([frame(time_s=40)])[0]
        self.assertEqual(r['data_state'],'DATA_GAP')
        self.assertIsNone(r['score'])
        r=self.session.append([frame(lightning_count_total=0)])[0]
        self.assertEqual(r['data_state'],'COUNTER_RESET')
        self.assertEqual(r['buffer'],0)

    def test_manual_change_reaches_model_and_count_is_delta(self):
        self.session.append([frame() for _ in range(31)])
        r=self.session.append([frame(ground_potential_a=3,lightning_count_total=1002)])[0]
        self.assertEqual(self.model.windows[0,-1,0],3)
        self.assertAlmostEqual(self.model.windows[0,-1,9],np.log1p(2),places=6)
        self.assertEqual(r['new_count'],2)
        self.assertTrue(r['equipment']['ground_potential_a'])

if __name__=='__main__': unittest.main()
