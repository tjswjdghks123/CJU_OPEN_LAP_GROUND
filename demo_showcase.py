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

def run_cases(predictor):
    results=[]
    for j,(kind,label) in enumerate(CASES.items()):
        processor=DeviceProcessor(); records=[]
        for frame,event in episode(91000+j,kind):
            state=processor.process(frame)
            if state['data_state']=='SENSOR_FAULT': raise ValueError(state['detail'])
            if state['model_ready']: state.update(predictor.predict(np.stack(processor.rows)))
            score=state['risk_score']
            alert=score is not None and score>=predictor.manifest['threshold']
            records.append({'time':frame['time_s'],'values':[frame[k] for k in RAW_FEATURES],
                'score':score,'alert':alert,'criterion':state['any_user_criterion_exceeded'],
                'equipment':state['equipment_state'],'new_count':state['new_lightning_count'],
                'event':event,'label':int(event is not None and frame['time_s']<event<=frame['time_s']+10),
                'command':None})
        valid=[r for r in records if r['score'] is not None]
        warnings=[r['time'] for r in valid if r['alert']]
        before=[t for t in warnings if event is not None and t<event]
        results.append({'kind':kind,'name':label,'event':event,'records':records,
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

def dashboard(out,cases,manifest):
    payload=json.dumps({'cases':cases,'labels':LABELS,'threshold':manifest['threshold']},ensure_ascii=False,allow_nan=False)
    html='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>5주차 LSTM 시연</title>
<style>*{box-sizing:border-box}body{font:16px 'Malgun Gothic',sans-serif;background:#f2f5fa;color:#18253c;margin:0;padding:24px}main{max-width:1150px;margin:auto}h1{font-size:26px;margin:0 0 10px}p{line-height:1.6}.notice{background:#fff3d5;padding:12px 18px;border-radius:8px}section{background:white;padding:20px;margin:16px 0;border-radius:12px}.controls{display:flex;flex-wrap:wrap;gap:10px}button,select{font:inherit;padding:9px;border:1px solid #b9c6d8;border-radius:6px;cursor:pointer}.status{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.status div{background:#eff4fb;padding:14px;border-radius:8px}.status strong{display:block;font-size:24px;margin-top:8px}svg{width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{text-align:left;border-bottom:1px solid #dbe3ee;padding:9px;font-size:14px}.alert{color:#ba2632}.good{color:#087860}small{color:#52627b}#details{display:grid;grid-template-columns:1fr 1fr;gap:20px}a{color:#205db0}@media(max-width:700px){.status{grid-template-columns:1fr 1fr}#details{grid-template-columns:1fr}}</style>
<main><h1>5주차 센서 10개 기반 LSTM 시연</h1><p class="notice"><b>합성 데이터 기반 시험 모델</b> · 점수는 실제 낙뢰 발생 확률이 아닙니다. 실제 센서 연결 및 장비 제어 없음.</p>
<section class="controls"><select id="case"></select><button id="play">일시정지</button><button id="restart">처음부터</button><label>속도 <select id="speed"><option value="1">1배</option><option value="5" selected>5배</option><option value="15">15배</option></select></label><span id="time"></span></section>
<section class="status"><div>LSTM 시험 점수<strong id="score"></strong></div><div>가상 전조 경보<strong id="alarm"></strong></div><div>장비 기준 판단<strong id="equipment"></strong></div><div>실제 장비 명령<strong>전송 없음</strong></div></section>
<section><b>최근 30초 분석 → 다음 10초 가상 사건 점수</b><p><small>처음 30초는 입력을 모읍니다. 주황 영역은 가상 사건 직전 10초입니다. 누적 횟수는 사건이 발생한 뒤 증가합니다.</small></p><svg id="chart" viewBox="0 0 1000 260" role="img" aria-label="시간별 LSTM 점수"></svg></section>
<section id="details"><div><b>현재 센서 입력 10개</b><table><thead><tr><th>센서</th><th>값</th><th>기준</th></tr></thead><tbody id="sensors"></tbody></table></div><div><b>선택한 시나리오</b><p id="explain"></p><p id="outcome"></p><p><small>가상 전조는 전위·누설이 35초 동안 함께 상승하도록 사람이 정한 패턴입니다. 이를 맞히는 능력은 실제 낙뢰 예측력의 증거가 아닙니다. 외기온도·습도는 시나리오 안에서 유지하고 소자온도는 60초마다 갱신합니다.</small></p><p><a href="prediction_demo.png">전체 시나리오 그래프</a> · <a href="training_loss.png">학습 손실 그래프</a> · <a href="summary.json">시험 결과</a></p></div></section></main>
<script>const data=PAYLOAD;let selected=0,index=0,playing=true,last=performance.now();const $=x=>document.getElementById(x);data.cases.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=c.name;$('case').append(o)});
const units=['V AC','V AC','mA','℃','℃','℃','%','℃','V','회'],limits=[2,2,2,45,45,45,70,25,null,null];
function render(){let c=data.cases[selected],r=c.records[index];$('time').textContent=`${index} / ${c.records.length-1}초`;$('score').textContent=r.score===null?'입력 수집 중':r.score.toFixed(4);$('alarm').textContent=r.score===null?'대기':r.alert?'시험 경보':'낮은 점수';$('alarm').className=r.alert?'alert':'good';$('equipment').textContent=r.equipment==='CRITERION_EXCEEDED'?'기준 초과':'기준 미만';$('sensors').innerHTML=data.labels.map((name,j)=>{let v=r.values[j]*(j===2?1000:1);return `<tr><td>${name}</td><td>${v.toFixed(j===9?0:2)} ${units[j]}</td><td>${limits[j]===null?'—':limits[j]+(v>=limits[j]?' ↑':'')}</td></tr>`}).join('');
let descriptions={normal:'정상 범위의 잡음이 포함된 가상 입력입니다.',thermal_only:'소자온도만 높습니다. 장비 이상 표시와 전조 점수를 구분합니다.',leakage_only:'누설전류만 높습니다. 장비 기준 초과가 곧 낙뢰라는 뜻은 아닙니다.',voltage_step:'대지전위만 갑자기 높아집니다. 함께 상승하는 가상 전조와 비교합니다.',synthetic_precursor:`${c.event}초에 가상 사건이 발생합니다. 그 전 35초 동안 전위와 누설을 함께 올렸습니다.`};$('explain').textContent=descriptions[c.kind];$('outcome').textContent='전체 재생 후 요약: 최대 시험 점수 '+c.summary.max_score.toFixed(4)+', 경보 '+c.summary.alert_rows+'초, 정답 구간 밖 경보 '+c.summary.false_alert_rows+'초'+(c.summary.lead_seconds===null?'':', 최초 사전 경보 '+c.summary.lead_seconds+'초 전')+'.';
let x=t=>55+t/179*920,y=s=>220-s*190;let points=c.records.slice(0,index+1).filter(q=>q.score!==null).map(q=>x(q.time)+','+y(q.score)).join(' ');let svg='';for(let v of [0,.25,.5,.75,1])svg+=`<line x1="55" y1="${y(v)}" x2="975" y2="${y(v)}" stroke="#e2e8f0"/><text x="8" y="${y(v)+5}" font-size="13">${v}</text>`;for(let t of [0,30,60,90,120,150,179])svg+=`<text x="${x(t)-8}" y="248" font-size="13">${t}</text>`;if(c.event!==null)svg+=`<rect x="${x(c.event-10)}" y="25" width="${x(10)-x(0)}" height="195" fill="#ffcf80" opacity=".35"/><line x1="${x(c.event)}" y1="25" x2="${x(c.event)}" y2="220" stroke="#d03232"/>`;svg+=`<line x1="55" y1="${y(data.threshold)}" x2="975" y2="${y(data.threshold)}" stroke="#ab4256" stroke-dasharray="6 4"/><text x="60" y="18" font-size="13">试验 threshold THRESHOLD</text><polyline points="${points}" fill="none" stroke="#216ac3" stroke-width="3"/><line x1="${x(index)}" y1="25" x2="${x(index)}" y2="220" stroke="#71839e" stroke-dasharray="3 4"/>`;$('chart').innerHTML=svg.replace('试验 threshold THRESHOLD','시험 경보 기준 '+data.threshold.toFixed(4));}
$('case').onchange=e=>{selected=+e.target.value;index=0;render()};$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'일시정지':'재생'};$('restart').onclick=()=>{index=0;render()};setInterval(()=>{let now=performance.now(),elapsed=(now-last)/1000;last=now;if(playing){index=Math.min(179,index+elapsed*+$('speed').value);let i=index;index=Math.floor(index);render();index=i}},50);render();</script></html>'''
    # Keep the fractional playback clock separate from array indexing.
    html=html.replace('최근 30초 분석 → 다음 10초 가상 사건 점수', '최근 30초 분석 → 다음 10초 가상 사건 점수 · 저장된 추론 결과 재생')
    (out/'dashboard.html').write_text(html.replace('PAYLOAD',payload),encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--no-open',action='store_true');p.add_argument('--output',type=Path)
    args=p.parse_args(); bundle=ROOT/'showcase_bundle'
    if not (bundle/'manifest.json').is_file(): raise SystemExit('Missing showcase_bundle. Run build_showcase.cmd first.')
    print('Loading 10-input LSTM | SYNTHETIC DEMONSTRATION ONLY | no hardware commands',flush=True)
    predictor=CountPredictor(bundle)
    if predictor.manifest['data_origin']!='synthetic': raise ValueError('This showcase requires its explicitly synthetic bundle')
    cases=run_cases(predictor)
    out=args.output or ROOT/'showcase_runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True,exist_ok=False)
    summary={'synthetic_only':True,'real_lightning_prediction_validated':False,'hardware_commands_sent':0,
        'threshold':predictor.manifest['threshold'],'cases':[{k:v for k,v in c.items() if k!='records'} for c in cases],
        'heldout_synthetic_evaluation':json.loads((bundle/'evaluation.json').read_text())['test']}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['case','time_s',*RAW_FEATURES,'score','alert','invented_label_next_10s'])
        for c in cases:
            for r in c['records']:w.writerow([c['kind'],r['time'],*r['values'],r['score'],r['alert'],r['label']])
    plots(out,cases,predictor.manifest);dashboard(out,cases,predictor.manifest)
    for c in cases: print(json.dumps({'case':c['kind'],**c['summary']}),flush=True)
    print('Dashboard: '+str(out/'dashboard.html'),flush=True)
    if not args.no_open:webbrowser.open((out/'dashboard.html').as_uri())
if __name__=='__main__':main()
