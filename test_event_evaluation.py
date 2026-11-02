import unittest
from event_evaluation import evaluate, parse_csv

def records(start=0,end=100,alarms=()):
    return [{'time_s':t,'score':1. if t in alarms else 0.} for t in range(start,end+1)]
def event(t,kind='virtual'):
    return {'time_s':t,'kind':kind,'source':'test'}

class EvaluationTests(unittest.TestCase):
    def test_boundary_and_current_time_exclusion(self):
        g=evaluate(records(alarms=[40,60]),[event(50),event(60)],.9)['groups']['virtual']
        self.assertEqual((g['hit'],g['miss']),(1,1))
        self.assertEqual(g['events'][0]['lead_seconds'],10)

    def test_incomplete_and_future_excluded(self):
        g=evaluate(records(30,50,[35]),[event(36),event(60)],.9)['groups']['virtual']
        self.assertEqual(g['excluded'],2)
        self.assertIsNone(g['recall'])

    def test_false_alert_requires_complete_catalog_and_future(self):
        rs=records(0,100,[20,21,95])
        self.assertIsNone(evaluate(rs,[],.9)['groups']['virtual']['false_alert_rate'])
        g=evaluate(rs,[],.9,{'virtual':{'complete':True,'start':0,'end':100}})['groups']['virtual']
        self.assertEqual(g['negative_windows'],91)
        self.assertEqual(g['false_alert_windows'],2)
        self.assertEqual(g['false_alert_intervals'],[{'start':20,'end':21,'windows':2}])

    def test_separate_sources_and_overlapping_events(self):
        g=evaluate(records(alarms=[40]),[event(45),event(46),event(80,'observed')],.9)['groups']
        self.assertEqual(g['virtual']['hit'],2)
        self.assertEqual(g['observed']['miss'],1)

    def test_csv_duplicates_nonfinite_and_quoted_source(self):
        self.assertEqual(parse_csv('time_s,kind,source\n5,observed,"station, A"\n')[0]['source'],'station, A')
        for data in ['time_s,kind,source\nnan,virtual,a','time_s,kind,source\n5,virtual,a\n5,virtual,b']:
            with self.assertRaises(ValueError):parse_csv(data)

    def test_missing_score_and_invalid_coverage(self):
        rs=records();rs[42]['score']=None
        self.assertEqual(evaluate(rs,[event(50)],.9)['groups']['virtual']['excluded'],1)
        with self.assertRaises(ValueError):evaluate(rs,[],.9,{'virtual':{'complete':True,'start':0,'end':101}})

if __name__=='__main__':unittest.main()
