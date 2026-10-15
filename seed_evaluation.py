"""Frozen synthetic-model evaluation. Seeds vary inputs, never model selection."""
import csv
import json
import numpy as np

PROFILES = [
    {'id':'standard','name':'기존 합성 규칙 · 난수 변화','pattern_scale':1.,'noise_scale':1.,'kinds':None},
    {'id':'weak','name':'약한 가상 전조 · 잡음 증가','pattern_scale':.6,'noise_scale':2.,'kinds':['synthetic_precursor']},
]

def summarize(cases, threshold):
    tp=fp=fn=tn=0; events=hits=negative=false_episodes=0; leads=[]
    for c in cases:
        valid=[r for r in c['records'] if r['score'] is not None]
        for r in valid:
            pred=r['score']>=threshold
            tp+=int(pred and r['label']==1);fp+=int(pred and r['label']==0)
            fn+=int(not pred and r['label']==1);tn+=int(not pred and r['label']==0)
        if c['event'] is not None:
            events+=1
            inside=[r['time'] for r in valid if r['label']==1 and r['score']>=threshold]
            if inside: hits+=1;leads.append(c['event']-min(inside))
        else:
            negative+=1;false_episodes+=int(any(r['score']>=threshold for r in valid))
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'window_recall':tp/(tp+fn) if tp+fn else None,
        'window_precision':tp/(tp+fp) if tp+fp else None,'window_fpr':fp/(fp+tn) if fp+tn else None,
        'events':events,'event_hits':hits,'event_misses':events-hits,
        'event_recall':hits/events if events else None,
        'no_event_episodes':negative,'false_alert_episodes':false_episodes,
        'false_episode_rate':false_episodes/negative if negative else None,
        'lead_seconds':leads,'lead_median':float(np.median(leads)) if leads else None}

def evaluate(predictor, run_cases, count=30, start=200000):
    if not 1<=count<=100: raise ValueError('seed count must be between 1 and 100')
    if start<200000: raise ValueError('Use reserved evaluation seeds >= 200000')
    result={'synthetic_only':True,'real_lightning_prediction_validated':False,
        'model_refitted':False,'threshold_retuned':False,'threshold':predictor.manifest['threshold'],
        'seed_start':start,'seed_count':count,'seed_stride':10,'profiles':[]}
    result['model_artifacts_sha256']=predictor.manifest['artifacts']
    for profile in PROFILES:
        trials=[]
        for i in range(count):
            base=start+i*10
            cases=run_cases(predictor,base,profile['pattern_scale'],profile['noise_scale'],profile['kinds'])
            trials.append({'base_seed':base,'cases':cases,'metrics':summarize(cases,result['threshold'])})
            if (i+1)%5==0 or i+1==count:
                print(f"  {profile['id']}: {i+1}/{count} seed groups evaluated",flush=True)
        result['profiles'].append({**profile,'trials':trials,
            'metrics':summarize([c for t in trials for c in t['cases']],result['threshold'])})
    return result

def report(out,bank):
    small={k:v for k,v in bank.items() if k!='profiles'}
    small['profiles']=[{k:v for k,v in p.items() if k!='trials'}|{
        'trials':[{'base_seed':t['base_seed'],'metrics':t['metrics'],
            'cases':[{k:v for k,v in c.items() if k!='records'} for c in t['cases']]} for t in p['trials']]} for p in bank['profiles']]
    (out/'seed_evaluation.json').write_text(json.dumps(small,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'seed_evaluation.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['profile','seed','case','max_score','event','hits_in_next_10s','false_alert_rows','lead_seconds'])
        for p in bank['profiles']:
            for t in p['trials']:
                for c in t['cases']:
                    s=c['summary'];w.writerow([p['id'],c['seed'],c['kind'],s['max_score'],c['event'],s['positive_window_hits'],s['false_alert_rows'],s['lead_seconds']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors=['#216ac3','#b75216']
    for p,color in zip(bank['profiles'],colors):
        for t in p['trials']:
            c=next(c for c in t['cases'] if c['event'] is not None)
            axes[0,0].plot([r['time']-c['event'] for r in c['records']],
                [np.nan if r['score'] is None else r['score'] for r in c['records']],color=color,alpha=.28,lw=.9)
        axes[0,0].plot([],[],color=color,label=p['name'])
    axes[0,0].axhline(bank['threshold'],color='black',ls='--',label='고정 경보 기준')
    axes[0,0].axvspan(-10,0,color='orange',alpha=.15);axes[0,0].axvline(0,color='red',ls=':')
    axes[0,0].set(xlim=(-45,15),ylim=(-.02,1.05),xlabel='가상 사건 기준 시간 (초)',ylabel='시험 점수',title='새 난수별 점수 곡선')
    axes[0,0].legend(fontsize=8)
    names=['기존 합성 규칙','약한 패턴·잡음 2배']
    hits=[p['metrics']['event_hits'] for p in bank['profiles']];misses=[p['metrics']['event_misses'] for p in bank['profiles']]
    axes[0,1].bar(names,hits,label='직전 10초 안 경보',color='#216ac3')
    axes[0,1].bar(names,misses,bottom=hits,label='놓친 가상 사건',color='#df8968')
    for i,(h,m) in enumerate(zip(hits,misses)):axes[0,1].text(i,h+m+.3,f'{h}/{h+m} 감지',ha='center')
    axes[0,1].set(ylim=(0,bank['seed_count']*1.2),ylabel='가상 사건 수',title='가상 사건 감지와 누락');axes[0,1].legend(fontsize=8)
    standard=bank['profiles'][0]
    negatives=[c for t in standard['trials'] for c in t['cases'] if c['event'] is None]
    kinds=list(dict.fromkeys(c['kind'] for c in negatives)); labels=[];rates=[]
    for kind in kinds:
        group=[c for c in negatives if c['kind']==kind];labels.append(group[0]['name'])
        rates.append(100*sum(c['summary']['alert_rows']>0 for c in group)/len(group))
    axes[1,0].bar(labels,rates,color='#7694ad');axes[1,0].set(ylim=(0,105),ylabel='사건 없는 사례 중 경보 비율 (%)',title='기존 합성 규칙의 사례별 오경보')
    for i,v in enumerate(rates):axes[1,0].text(i,v+2,f'{v:.1f}%',ha='center')
    for p,color,name in zip(bank['profiles'],colors,names):
        leads=p['metrics']['lead_seconds'];axes[1,1].hist(leads,bins=np.arange(.5,11.5,1),alpha=.6,color=color,label=f'{name} ({len(leads)}건)')
    axes[1,1].set(xlabel='첫 유효 경보의 사건 전 시간 (초)',ylabel='감지한 가상 사건 수',title='직전 10초 안 경고 시간 분포',xticks=range(1,11));axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(axis='y',alpha=.15)
    fig.suptitle(f"난수 {bank['seed_count']}개 반복 평가 · 합성 데이터만 사용 · 모델/임계값 고정\n약한 패턴은 원래 진폭의 60% · 실제 낙뢰 예측 성능 아님",fontsize=13)
    fig.savefig(out/'seed_evaluation.png',dpi=160);plt.close(fig)

def encode_cases(cases):
    # Compact payload: evaluation histories are replayed locally without a server.
    return [{k:v for k,v in c.items() if k!='records'}|{'records':[
        [r['time'],r['values'],r['score'],int(r['alert']),int(r['criterion']),r['equipment'],r['new_count'],r['label']]
        for r in c['records']]} for c in cases]

def payload(bank):
    return {k:v for k,v in bank.items() if k!='profiles'}|{'profiles':[
        {k:v for k,v in p.items() if k!='trials'}|{'trials':[
            {'base_seed':t['base_seed'],'metrics':t['metrics'],'cases':encode_cases(t['cases'])} for t in p['trials']]} for p in bank['profiles']]}
