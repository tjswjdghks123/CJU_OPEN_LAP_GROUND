import unittest
import numpy as np
from prepare_showcase import episode
from device_server import DeviceProcessor, RAW_FEATURES

class ShowcaseTests(unittest.TestCase):
    def test_precursor_event_count_is_not_available_early(self):
        frames=list(episode(23,'synthetic_precursor'))
        event=frames[0][1]
        self.assertTrue(all(f['lightning_count_total']==1000 for f,e in frames if f['time_s']<event))
        self.assertEqual(frames[event][0]['lightning_count_total'],1001)

    def test_ten_channels_and_measurement_periods(self):
        for kind in ('normal','thermal_only','leakage_only','voltage_step','synthetic_precursor'):
            processor=DeviceProcessor(); previous=None
            for f,event in episode(22,kind):
                self.assertTrue(all(k in f for k in RAW_FEATURES))
                state=processor.process(f)
                self.assertNotEqual(state['data_state'],'SENSOR_FAULT')
                if previous:
                    self.assertEqual(f['relative_humidity'],previous['relative_humidity'])
                    self.assertEqual(f['ambient_temperature'],previous['ambient_temperature'])
                    if f['time_s']%60:
                        self.assertEqual([f[k] for k in RAW_FEATURES[3:6]],[previous[k] for k in RAW_FEATURES[3:6]])
                previous=f
            self.assertEqual(np.stack(processor.rows).shape,(30,10))

    def test_negative_controls_have_no_events(self):
        for kind in ('normal','thermal_only','leakage_only','voltage_step'):
            self.assertTrue(all(event is None for f,event in episode(31,kind)))

    def test_synthetic_labels_and_no_hardware(self):
        from train_device_count import CountPredictor
        from demo_showcase import ROOT,run_cases
        model=CountPredictor(ROOT/'showcase_bundle');cases=run_cases(model)
        self.assertEqual(model.manifest['data_origin'],'synthetic')
        for c in cases:
            for i,r in enumerate(c['records']):
                self.assertIsNone(r['command'])
                if i<30:self.assertIsNone(r['score'])
                else:self.assertTrue(0<=r['score']<=1)
                self.assertEqual(r['label'],int(c['event'] is not None and i<c['event']<=i+10))

    def test_batch_and_stream_scores_match(self):
        from train_device_count import CountPredictor
        from demo_showcase import ROOT
        model=CountPredictor(ROOT/'showcase_bundle')
        processor=DeviceProcessor(); windows=[]; scores=[]
        for f,event in episode(91004,'synthetic_precursor'):
            state=processor.process(f)
            if state['model_ready']:
                rows=np.stack(processor.rows);windows.append(rows);scores.append(model.predict(rows)['risk_score'])
        batch=np.asarray(windows)*model.scale+model.offset
        expected=np.round(model.model(batch,training=False).numpy().ravel().astype(float),4)
        np.testing.assert_allclose(scores,expected,atol=.0001001,rtol=0)

if __name__=='__main__':unittest.main()
