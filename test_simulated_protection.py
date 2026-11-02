import unittest
from simulated_protection import SimulatedProtection,replay,fault_examples


class ProtectionTests(unittest.TestCase):
    def test_requires_consecutive_scores_and_latches(self):
        c=SimulatedProtection(.9)
        states=[c.step(t,s,acknowledgement=t==6)['state'] for t,s in enumerate([.95,.1,.95,.95,.95,.1,.1,.1])]
        self.assertEqual(states,['WARNING','NORMAL','WARNING','WARNING','REQUESTED','REQUESTED','PROTECTED_SIM','PROTECTED_SIM'])
        self.assertTrue(all(e['hardware_command'] is None for e in c.events))

    def test_timeout_wins_over_late_ack(self):
        c=SimulatedProtection(.9)
        for t in range(6): c.step(t,.99,acknowledgement=t==5)
        self.assertEqual(c.state,'TIMEOUT')
        self.assertEqual(c.step(6,.99,acknowledgement=True)['state'],'TIMEOUT')

    def test_input_fault_and_gap_latch(self):
        fixtures=fault_examples(.9)
        self.assertEqual(fixtures['sensor_missing']['final_state'],'DATA_FAULT')
        self.assertEqual(fixtures['time_gap']['final_state'],'DATA_FAULT')
        self.assertEqual(fixtures['ack_timeout']['final_state'],'TIMEOUT')
        self.assertEqual(fixtures['mock_ack']['final_state'],'PROTECTED_SIM')

    def test_labels_cannot_trigger_protection(self):
        rows=[{'time':t,'score':.1,'event':3,'label':1} for t in range(8)]
        self.assertIsNone(replay(rows,.9)['request_time'])
        self.assertEqual(replay(rows,.9)['final_state'],'NORMAL')

    def test_invalid_score_and_warmup(self):
        c=SimulatedProtection(.9)
        self.assertEqual(c.step(0,None)['state'],'WARMUP')
        self.assertEqual(c.step(1,float('nan'))['state'],'DATA_FAULT')


if __name__=='__main__':unittest.main()
