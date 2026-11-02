import unittest
import numpy as np
from improve_showcase import generate, dataset, SPLITS, KINDS, summarize

class ImprovementTests(unittest.TestCase):
    def test_disjoint_episode_seeds(self):
        sets=[{base+g*20+j for g in range(n) for j in range(len(KINDS))} for base,n in SPLITS.values()]
        self.assertTrue(all(not sets[i]&sets[j] for i in range(3) for j in range(i)))

    def test_deterministic_labels_and_counters(self):
        for j,kind in enumerate(KINDS):
            f,m=generate(410000+j,kind)
            self.assertEqual((f,m),generate(410000+j,kind))
            e=m['event']
            self.assertEqual(len(f),180)
            if e is not None:
                self.assertTrue(all(r['lightning_count_total']==1000 for r in f[:e]))
                self.assertEqual(f[e]['lightning_count_total'],1001)
            else:self.assertTrue(all(r['lightning_count_total']==1000 for r in f))

    def test_windows_labels_and_negative_cases(self):
        x,y,c=dataset(710000,1)
        self.assertEqual(x.shape,(len(KINDS)*150,30,10))
        self.assertEqual(y.sum(),30)
        self.assertTrue(np.isfinite(x).all())
        for item in c:
            labels=y[item['begin']:item['end']]
            if item['event'] is None:self.assertEqual(labels.sum(),0)
            else:
                times=np.arange(30,180)
                self.assertTrue(np.array_equal(times[labels==1],np.arange(item['event']-10,item['event'])))
                self.assertTrue((x[item['begin']:item['end']][labels==1,:,-1]==0).all())

    def test_metrics_do_not_count_after_event(self):
        x,y,c=dataset(710000,1)
        scores=np.zeros(len(y))
        for item in c:
            if item['event'] is not None:scores[item['begin']+item['event']-30]=1
        r=summarize(y,scores,c,.5)
        self.assertEqual(sum(g['hits'] for g in r['by_kind'].values()),0)

if __name__=='__main__':unittest.main()
