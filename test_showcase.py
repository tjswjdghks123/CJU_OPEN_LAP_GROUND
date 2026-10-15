import unittest
import numpy as np
from prepare_showcase import episode
from device_server import DeviceProcessor, RAW_FEATURES

class ShowcaseTests(unittest.TestCase):
    def test_seed_repeatability_and_variation(self):
        a=list(episode(200004,'synthetic_precursor'))
        self.assertEqual(a,list(episode(200004,'synthetic_precursor')))
        self.assertNotEqual(a,list(episode(200014,'synthetic_precursor')))

    def test_counter_baseline_cannot_change_model_inputs(self):
        inputs=[]
        for baseline in (0,1000,1000000):
            p=DeviceProcessor();rows=[]
            for f,event in episode(200004,'synthetic_precursor',initial_count=baseline):
                r=p.process(f)
                if r.get('model_input') is not None:rows.append(r['model_input'])
            inputs.append(rows)
        np.testing.assert_array_equal(inputs[0],inputs[1])
        np.testing.assert_array_equal(inputs[1],inputs[2])

    def test_event_metric_does_not_count_early_or_post_event_alerts(self):
        from seed_evaluation import summarize
        cases=[{'event':100,'records':[
            {'time':80,'score':.99,'label':0},
            {'time':95,'score':.1,'label':1},
            {'time':100,'score':.99,'label':0}]}]
        m=summarize(cases,.9)
        self.assertEqual((m['event_hits'],m['event_misses'],m['fp'],m['fn']),(0,1,2,1))
        cases[0]['records'][1]['score']=.99
        m=summarize(cases,.9)
        self.assertEqual(m['event_hits'],1);self.assertEqual(m['lead_seconds'],[5])

    def test_false_episode_and_window_metrics_have_different_denominators(self):
        from seed_evaluation import summarize
        cases=[{'event':None,'records':[{'score':.99,'label':0},{'score':.99,'label':0}]},
               {'event':None,'records':[{'score':.1,'label':0},{'score':None,'label':0}]}]
        m=summarize(cases,.9)
        self.assertEqual(m['false_alert_episodes'],1)
        self.assertEqual(m['false_episode_rate'],.5)
        self.assertEqual(m['window_fpr'],2/3)
        self.assertIsNone(m['event_recall'])
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
        np.testing.assert_allclose(model.predict_many(windows),expected,atol=.0001001,rtol=0)

if __name__=='__main__':unittest.main()
