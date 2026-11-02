"""Predeclared synthetic-only experiment. Test seeds never tune weights or threshold."""
import argparse
import json
from pathlib import Path
import hashlib
import numpy as np
from device_server import DeviceProcessor, RAW_FEATURES, MODEL_FEATURES, TRANSFORM_VERSION
from prepare_showcase import episode
from sensor10_pipeline import tf_setup, choose_threshold, binary_metrics, digest

KINDS=['normal','thermal_only','leakage_only','voltage_step','spike','recovery','standard','weak','slow']
NAMES=['정상','과열만','누설만','전위만','순간 동반 튐','상승 후 회복','기본 가상 패턴','약한 가상 패턴','느린 가상 패턴']
SPLITS={'train':(410000,24),'validation':(510000,10),'test':(610000,30)}

def generate(seed,kind):
    rng=np.random.default_rng(seed+9000000)
    event=int(rng.integers(115,140)) if kind in ('standard','weak','slow') else None
    duration=int(rng.integers(50,66)) if kind=='slow' else int(rng.integers(28,44))
    amplitude=float(rng.uniform(.45,.75)) if kind=='weak' else float(rng.uniform(.85,1.15))
    noise=float(rng.uniform(.8,2.))
    if kind=='standard':duration=35;amplitude=1.;noise=1.
    start=event-duration if event is not None else int(rng.integers(65,85))
    negative_peak=float(rng.uniform(.45,1.15))
    original=kind if kind in KINDS[:4] else 'normal'
    frames=[]
    for frame,_ in episode(seed,original,noise_scale=noise):
        t=frame['time_s'];ramp=0.
        if event is not None and start<=t<event:ramp=(t-start)/duration*amplitude
        if kind=='spike' and start<=t<start+3:ramp=negative_peak
        # Negative recovery shares the positive co-ramp form until it turns down.
        # This intentional ambiguity can cause genuine synthetic false alarms.
        if kind=='recovery' and start<=t<start+40:
            ramp=negative_peak*((t-start)/20 if t<start+20 else (start+40-t)/20)
        for key,delta in zip(RAW_FEATURES[:3],[2.1,1.9,.0016]):frame[key]+=ramp*delta
        if event is not None and t>=event:frame['lightning_count_total']+=1
        frames.append(frame)
    return frames,{'seed':seed,'kind':kind,'event':event,'start':start,'duration':duration,'amplitude':amplitude,'noise':noise}

def dataset(base,n):
    windows=[];labels=[];cases=[]
    for group in range(n):
        for j,kind in enumerate(KINDS):
            frames,meta=generate(base+group*20+j,kind)
            p=DeviceProcessor();begin=len(windows)
            for f in frames:
                state=p.process(f)
                assert state['data_state']!='SENSOR_FAULT'
                if state['model_ready']:
                    windows.append(np.stack(p.rows))
                    labels.append(int(meta['event'] is not None and f['time_s']<meta['event']<=f['time_s']+10))
            cases.append({**meta,'begin':begin,'end':len(windows),'values':[[f[k] for k in RAW_FEATURES] for f in frames]})
    return np.asarray(windows,dtype=np.float32),np.asarray(labels,dtype=np.int32),cases

def summarize(y,scores,cases,threshold):
    overall=binary_metrics(y,scores,threshold);by_kind={};details=[]
    for c in cases:
        s=scores[c['begin']:c['end']];a=np.arange(30,180);e=c['event']
        hits=a[(s>=threshold)&((a<e)&(a>=e-10) if e is not None else np.zeros(len(a),bool))]
        row={'seed':c['seed'],'kind':c['kind'],'event':e,'hit':bool(len(hits)) if e is not None else None,
             'lead':int(e-hits[0]) if len(hits) else None,'alert_windows':int(np.sum(s>=threshold)),
             'false_windows':int(np.sum((s>=threshold)&(y[c['begin']:c['end']]==0)))}
        details.append(row)
    for kind in KINDS:
        rows=[r for r in details if r['kind']==kind];events=[r for r in rows if r['event'] is not None]
        leads=[r['lead'] for r in events if r['hit']]
        by_kind[kind]={'cases':len(rows),'events':len(events),'hits':sum(r['hit'] for r in events),
                       'misses':sum(not r['hit'] for r in events),
                       'false_alert_cases':sum(r['alert_windows']>0 for r in rows) if not events else None,
                       'median_lead':float(np.median(leads)) if leads else None}
    return {'threshold':threshold,'windows':overall,'by_kind':by_kind,'cases':details}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--epochs',type=int,default=20)
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parent
    oldroot=root/'showcase_bundle'
    oldmanifest=json.loads((oldroot/'manifest.json').read_text(encoding='utf-8'))
    oldhash=digest(oldroot/'count_model.keras')
    policy={'synthetic_only':True,'real_lightning_validated':False,'splits':SPLITS,'kinds':KINDS,
            'epochs_max':args.epochs,'selection':'validation F1 maximum subject to window FPR <= 1%; rounded to 4 decimals',
            'checkpoint':'lowest validation loss; patience 5','test_used_for_selection':False,
            'negative_ambiguity':'recovery can share an initial rise with positive episodes; do not promise perfect discrimination',
            'old_model_sha256':oldhash,'script_sha256':digest(__file__)}
    def write(name,value): (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    write('experiment_policy.json',policy)
    print('Generating train/validation ONLY. Test is generated after model and threshold freeze.',flush=True)
    xt,yt,ct=dataset(*SPLITS['train']);xv,yv,cv=dataset(*SPLITS['validation'])
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler().fit(xt.reshape(-1,10))
    scale=scaler.scale_.astype(np.float32);offset=scaler.min_.astype(np.float32)
    xt=xt*scale+offset;xv=xv*scale+offset
    tf=tf_setup()
    model=tf.keras.Sequential([tf.keras.Input((30,10)),tf.keras.layers.LSTM(64,return_sequences=True),
        tf.keras.layers.LSTM(32),tf.keras.layers.Dense(16,activation='relu'),tf.keras.layers.Dense(1,activation='sigmoid')])
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC(curve='PR',name='pr_auc')])
    def ds(x,y,shuffle=False):
        d=tf.data.Dataset.from_tensor_slices((x,y))
        if shuffle:d=d.shuffle(len(x),seed=42)
        opt=tf.data.Options();opt.threading.private_threadpool_size=2
        return d.batch(128).with_options(opt).prefetch(1)
    hist=model.fit(ds(xt,yt,True),validation_data=ds(xv,yv),epochs=args.epochs,verbose=2,
        class_weight={0:1.,1:8.},callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)])
    def predict(m,x):return np.round(np.concatenate([m(x[i:i+256],training=False).numpy().ravel() for i in range(0,len(x),256)]).astype(float),4)
    vs=predict(model,xv);threshold=choose_threshold(yv,vs,.01)
    if binary_metrics(yv,vs,threshold)['tp']==0:threshold=1.0001
    model.save(out/'count_model.keras')
    np.savez(out/'scaler.npz',scale=scale,offset=offset,features=np.array(MODEL_FEATURES))
    write('history.json',hist.history)
    manifest={**oldmanifest,'threshold':threshold,'source_manifest':policy,'target':'invented_event_in_next_10s',
              'artifacts':{name:digest(out/name) for name in ('count_model.keras','scaler.npz')}}
    manifest.pop('frames_sha256',None);manifest.pop('events_sha256',None)
    write('manifest.json',manifest)
    write('validation.json',summarize(yv,vs,cv,threshold))
    print('Model and threshold frozen. Generating untouched test seed groups.',flush=True)
    x,y,cases=dataset(*SPLITS['test'])
    improved=predict(model,x*scale+offset)
    oldmodel=tf.keras.models.load_model(oldroot/'count_model.keras',compile=False)
    with np.load(oldroot/'scaler.npz') as z:old=predict(oldmodel,x*z['scale']+z['offset'])
    results={'policy':policy,'old':summarize(y,old,cases,oldmanifest['threshold']),
             'new':summarize(y,improved,cases,threshold)}
    # Threshold-only baseline comparison isolates some threshold-selection effects.
    with np.load(oldroot/'scaler.npz') as z:
        rawval=(xv-offset)/scale
        oldval=predict(oldmodel,rawval*z['scale']+z['offset'])
    retuned=choose_threshold(yv,oldval,.01)
    results['old_rethresholded']=summarize(y,old,cases,retuned)
    results['policy']['old_rethresholded_selection']='new validation only; weights unchanged'
    write('comparison.json',results)
    np.savez_compressed(out/'test_predictions.npz',labels=y,old=old,new=improved)
    write('test_cases.json',cases)
    assert digest(oldroot/'count_model.keras')==oldhash
    print(json.dumps({m:{'threshold':results[m]['threshold'],'windows':results[m]['windows'],'by_kind':results[m]['by_kind']} for m in ['old','new','old_rethresholded']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
