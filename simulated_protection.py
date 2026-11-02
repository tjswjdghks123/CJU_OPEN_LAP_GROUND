"""Deterministic software-only request/response simulation. No I/O or hardware commands."""
import math


class SimulatedProtection:
    def __init__(self, threshold, consecutive=3, timeout=3):
        self.threshold=threshold
        self.consecutive=consecutive
        self.timeout=timeout
        self.state='WARMUP'
        self.streak=0
        self.request_time=None
        self.last_time=None
        self.events=[]

    def step(self, time_s, score, sensor_ok=True, acknowledgement=False):
        previous=self.state
        bad_time=not isinstance(time_s,(int,float)) or not math.isfinite(time_s)
        gap=self.last_time is not None and not bad_time and time_s != self.last_time+1
        if not bad_time:
            self.last_time=time_s
        valid_score=score is None or (isinstance(score,(int,float)) and math.isfinite(score) and 0<=score<=1)
        if bad_time or gap or not sensor_ok or not valid_score:
            self.state='DATA_FAULT';self.streak=0
        elif self.state in ('DATA_FAULT','TIMEOUT','PROTECTED_SIM'):
            pass  # Latched; a new replay creates a new simulator. Never auto-resume after a fault.
        elif self.state=='REQUESTED':
            elapsed=time_s-self.request_time
            if elapsed >= self.timeout:
                self.state='TIMEOUT'
            elif acknowledgement and elapsed>0:
                self.state='PROTECTED_SIM'
        elif score is None:
            self.state='WARMUP';self.streak=0
        elif score >= self.threshold:
            self.streak+=1;self.state='WARNING'
            if self.streak >= self.consecutive:
                self.state='REQUESTED';self.request_time=time_s
        else:
            self.state='NORMAL';self.streak=0
        if self.state != previous:
            self.events.append({'time':time_s if not bad_time else None,'from':previous,'to':self.state,
                                'simulation_only':True,'hardware_command':None})
        return {'state':self.state,'streak':self.streak,'request_time':self.request_time,
                'simulation_only':True,'hardware_command':None,'physical_protection_verified':False}


def replay(records, threshold, ack_delay=1):
    """Mock transport acknowledges one second after a request; never consults event labels."""
    controller=SimulatedProtection(threshold)
    states=[]
    for row in records:
        ack=ack_delay is not None and controller.request_time is not None and row['time']-controller.request_time>=ack_delay
        result=controller.step(row['time'],row['score'],row.get('sensor_ok',True),ack)
        states.append(result['state'])
    return {'states':states,'events':controller.events,'final_state':controller.state,
            'request_time':controller.request_time,'ack_delay':ack_delay,
            'consecutive':controller.consecutive,'timeout_seconds':controller.timeout,
            'simulation_only':True,'hardware_commands_sent':0,'physical_protection_verified':False}


def fault_examples(threshold):
    # Explicit controller fixtures; scores here are injected, not LSTM predictions.
    base=[{'time':t,'score':min(1.,threshold+.001)} for t in range(7)]
    missing=[dict(r) for r in base];missing[1]['sensor_ok']=False
    gap=[dict(r) for r in base if r['time']!=1]
    return {name:{'score_source':'injected_controller_fixture',**replay(rows,threshold,delay)}
            for name,rows,delay in [('mock_ack',base,1),('ack_timeout',base,None),
                                    ('sensor_missing',missing,1),('time_gap',gap,1)]}
