"""Local ten-channel synthetic LSTM teaching demonstration. No hardware control."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import webbrowser
import numpy as np
from device_server import DeviceProcessor, RAW_FEATURES
from prepare_showcase import episode
from train_device_count import CountPredictor

ROOT=Path(__file__).resolve().parent
LABELS=['대지전위 a','대지전위 b','누설전류','소자온도 a','소자온도 b','소자온도 c','상대습도','외기온도','입력전압','낙뢰 누적횟수']
CASES={'normal':'정상','thermal_only':'과열만 발생','leakage_only':'누설만 발생','voltage_step':'전위 상승만 발생','synthetic_precursor':'가상 전조 패턴'}

def run_cases(predictor, seed_base=91000, pattern_scale=1., noise_scale=1., kinds=None):
    results=[]
    for j,(kind,label) in enumerate(CASES.items()):
        if kinds is not None and kind not in kinds: continue
        processor=DeviceProcessor(); records=[]; windows=[]; anchors=[]
        for frame,event in episode(seed_base+j,kind,pattern_scale=pattern_scale,noise_scale=noise_scale):
            state=processor.process(frame)
            if state['data_state']=='SENSOR_FAULT': raise ValueError(state['detail'])
            if state['model_ready']:
                windows.append(np.stack(processor.rows)); anchors.append(len(records))
            score=state['risk_score']
            alert=score is not None and score>=predictor.manifest['threshold']
            records.append({'time':frame['time_s'],'values':[frame[k] for k in RAW_FEATURES],
                'score':score,'alert':alert,'criterion':state['any_user_criterion_exceeded'],
                'equipment':state['equipment_state'],'new_count':state['new_lightning_count'],
                'event':event,'label':int(event is not None and frame['time_s']<event<=frame['time_s']+10),
                'command':None})
        for at,score in zip(anchors,predictor.predict_many(np.asarray(windows))):
            records[at]['score']=float(score)
            records[at]['alert']=bool(score>=predictor.manifest['threshold'])
        valid=[r for r in records if r['score'] is not None]
        warnings=[r['time'] for r in valid if r['alert']]
        before=[r['time'] for r in valid if r['alert'] and r['label']==1]
        results.append({'kind':kind,'name':label,'event':event,'seed':seed_base+j,'records':records,
            'summary':{'max_score':max(r['score'] for r in valid),'alert_rows':len(warnings),
                'positive_window_hits':sum(r['alert'] and r['label']==1 for r in valid),
                'false_alert_rows':sum(r['alert'] and r['label']==0 for r in valid),
                'lead_seconds':event-min(before) if before else None}})
    return results

def plots(out,cases,bundle):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic'; plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained')
    for c in cases:
        axes[0].plot([r['time'] for r in c['records']], [np.nan if r['score'] is None else r['score'] for r in c['records']],label=c['name'])
    axes[0].axhline(bundle['threshold'],color='black',ls='--',label=f"경보 기준 {bundle['threshold']:.4f}")
    precursor=cases[-1]; ev=precursor['event']
    axes[0].axvspan(ev-10,ev,color='orange',alpha=.15,label='가상 사건 직전 10초')
    axes[0].axvline(ev,color='red',ls=':'); axes[0].set(ylim=(-.03,1.06),ylabel='Sigmoid 시험 점수',title='10개 센서 기반 LSTM 시연 — 모두 합성 데이터')
    axes[0].legend(ncol=3,fontsize=8); axes[0].grid(alpha=.2)
    axes[1].plot([r['time'] for r in precursor['records']],[r['values'][0] for r in precursor['records']],label='대지전위 a (V AC)')
    axes[1].plot([r['time'] for r in precursor['records']],[r['values'][2]*1000 for r in precursor['records']],label='누설전류 (mA)')
    axes[1].axvline(ev,color='red',ls=':',label='가상 사건'); axes[1].set(xlabel='시연 시간 (초)',ylabel='각 표시 단위의 값',title='의도적으로 만든 전위·누설 동반 상승 패턴 — 실제 전조 근거 아님')
    axes[1].legend(); axes[1].grid(alpha=.2)
    fig.savefig(out/'prediction_demo.png',dpi=160); plt.close(fig)
    history=json.loads((ROOT/'showcase_bundle/history.json').read_text())
    fig,ax=plt.subplots(figsize=(8,3.6),layout='constrained')
    for key,name in [('loss','학습'),('val_loss','검증')]: ax.plot(range(1,len(history[key])+1),history[key],label=name)
    ax.set(xlabel='Epoch',ylabel='Binary cross entropy',title='합성 데이터 LSTM 학습 손실 — 실제 낙뢰 예측 성능 아님'); ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'training_loss.png',dpi=160);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--no-open',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--seed-count',type=int,default=30,help='Number of new seed groups per evaluation condition (1-100)')
    args=p.parse_args(); bundle=ROOT/'showcase_bundle'
    if not 1<=args.seed_count<=100:p.error('--seed-count must be 1-100')
    if not (bundle/'manifest.json').is_file(): raise SystemExit('Missing showcase_bundle. Run build_showcase.cmd first.')
    print('Loading 10-input LSTM | SYNTHETIC DEMONSTRATION ONLY | no hardware commands',flush=True)
    predictor=CountPredictor(bundle)
    if predictor.manifest['data_origin']!='synthetic': raise ValueError('This showcase requires its explicitly synthetic bundle')
    cases=run_cases(predictor)
    from seed_evaluation import evaluate,report,payload,encode_cases
    print(f'Evaluating {args.seed_count} fresh seeds per condition; frozen model and threshold...',flush=True)
    bank=evaluate(predictor,run_cases,args.seed_count)
    out=args.output or ROOT/'showcase_runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True,exist_ok=False)
    summary={'synthetic_only':True,'real_lightning_prediction_validated':False,'hardware_commands_sent':0,
        'threshold':predictor.manifest['threshold'],'cases':[{k:v for k,v in c.items() if k!='records'} for c in cases],
        'heldout_synthetic_evaluation':json.loads((bundle/'evaluation.json').read_text())['test']}
    summary['multi_seed_evaluation']={p['id']:p['metrics'] for p in bank['profiles']}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['case','time_s',*RAW_FEATURES,'score','alert','invented_label_next_10s'])
        for c in cases:
            for r in c['records']:w.writerow([c['kind'],r['time'],*r['values'],r['score'],r['alert'],r['label']])
    plots(out,cases,predictor.manifest);report(out,bank)
    data={'cases':encode_cases(cases),'labels':LABELS,'threshold':predictor.manifest['threshold'],'bank':payload(bank)}
    template=(ROOT/'showcase_dashboard.html').read_text(encoding='utf-8')
    (out/'dashboard.html').write_text(template.replace('__PAYLOAD__',json.dumps(data,ensure_ascii=False,allow_nan=False)),encoding='utf-8')
    for c in cases: print(json.dumps({'case':c['kind'],**c['summary']}),flush=True)
    for p in bank['profiles']:print(json.dumps({'evaluation_profile':p['id'],**p['metrics']}),flush=True)
    print('Dashboard: '+str(out/'dashboard.html'),flush=True)
    if not args.no_open:webbrowser.open((out/'dashboard.html').as_uri())
if __name__=='__main__':main()
