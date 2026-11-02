"""Reproduce the frozen CIRAD benchmark and build a portable presentation report.

Usage: python build_real_observation_report.py --source PATH --output PATH
Requires the previously collected external_weather_inputs and external_weather_bundle.
Never trains or adjusts thresholds on the held-out observations.
"""
import argparse
import csv
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, prediction):
    y = np.asarray(y, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    tp = int((y & prediction).sum())
    fn = int((y & ~prediction).sum())
    fp = int((~y & prediction).sum())
    tn = int((~y & ~prediction).sum())
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, windows=len(y),
                recall=tp/(tp+fn) if tp+fn else None,
                precision=tp/(tp+fp) if tp+fp else None,
                false_positive_rate=fp/(fp+tn) if fp+tn else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.source.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('weather_benchmark', root/'external_weather_model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import joblib
    bundle = root/'external_weather_bundle'
    manifest = json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    evaluation = json.loads((bundle/'evaluation.json').read_text(encoding='utf-8'))
    choice = json.loads((bundle/'frozen_validation_choice.json').read_text(encoding='utf-8'))
    name = choice['selected']
    threshold = choice['thresholds'][name]
    assert threshold == manifest['threshold']
    assert sha(bundle/manifest['selected_model']) == manifest['sha256']
    model = joblib.load(bundle/manifest['selected_model'])
    all_y, all_score, all_quiet, all_baseline = [], [], [], []
    stations, examples = [], {}
    onset_hits = []
    onset_total = 0
    hashes = {}
    for source in evaluation['sources']:
        if not source['role'].startswith('previously unscored'):
            continue
        sid = source['station']
        raw_path = root/'external_weather_inputs'/f'{sid}.csv'
        assert sha(raw_path) == source['sha256'], sid
        times, raw, events, valid = station = module.read_station(raw_path)
        start = int(np.searchsorted(times, np.datetime64('2025-04-13T00:00')))
        x, y, anchors, baseline, quiet = module.make_examples(station, start, len(times))
        pred_path = bundle/f'{sid}_predictions.csv'
        with pred_path.open(encoding='utf-8-sig') as stream:
            saved = list(csv.DictReader(stream))
        assert len(saved) == len(y)
        assert [r['time_as_provided'] for r in saved] == times[anchors].astype(str).tolist()
        np.testing.assert_array_equal(y, [int(r['label']) for r in saved])
        np.testing.assert_array_equal(quiet, [bool(int(r['quiet_history'])) for r in saved])
        score = model.predict_proba(x)[:, 1]
        np.testing.assert_allclose(score, [float(r[name]) for r in saved], rtol=1e-7, atol=1e-9)
        predicted = score >= threshold
        result = metrics(y, predicted)
        for key in ('tp', 'fn', 'fp', 'tn'):
            assert result[key] == evaluation['holdout'][sid][name]['all_windows'][key]
        hashes[sid] = dict(raw=sha(raw_path), predictions=sha(pred_path))
        stations.append(dict(station=sid, first=str(times[anchors[0]]), last=str(times[anchors[-1]]), **result))
        all_y.extend(y.tolist()); all_score.extend(score.tolist())
        all_quiet.extend(quiet.tolist()); all_baseline.extend(baseline.tolist())
        # Reproducible first chronological example of each outcome, not best-looking examples.
        for label, mask in [('hit', (y == 1) & predicted), ('miss', (y == 1) & ~predicted),
                            ('false_alarm', (y == 0) & predicted)]:
            if label in examples or not mask.any():
                continue
            k = int(np.flatnonzero(mask)[0]); anchor = anchors[k]
            take = (times[anchors] >= times[anchor]-np.timedelta64(60, 'm')) & (times[anchors] <= times[anchor]+np.timedelta64(60, 'm'))
            raw_take = (times >= times[anchor]-np.timedelta64(60, 'm')) & (times <= times[anchor]+np.timedelta64(60, 'm'))
            examples[label] = dict(station=sid, anchor=str(times[anchor]), score=float(score[k]), label=int(y[k]),
                quiet_history=bool(quiet[k]), times=times[anchors[take]].astype(str).tolist(),
                scores=score[take].tolist(), labels=y[take].tolist(),
                observation_times=times[raw_take].astype(str).tolist(),
                observation_counts=np.expm1(raw[raw_take, 7]).round().astype(int).tolist())
        # Onset means first recorded activity after 150 quiet minutes, not a verified first CG strike.
        lookup = {int(a): i for i, a in enumerate(anchors)}
        for event in np.flatnonzero(events):
            if event < 30 or not valid[event-30:event+1].all() or events[event-30:event].any():
                continue
            if not (np.diff(times[event-30:event+1]) == np.timedelta64(5, 'm')).all():
                continue
            prior = list(range(event-6, event))
            if not all(a in lookup for a in prior):
                continue
            onset_total += 1
            alerted = [a for a in prior if predicted[lookup[a]]]
            if alerted:
                onset_hits.append(int((times[event]-times[min(alerted)])/np.timedelta64(1, 'm')))
        print('Verified', sid, len(y), flush=True)
    y = np.array(all_y); scores = np.array(all_score); quiet = np.array(all_quiet, bool)
    summary = dict(source_doi='10.18167/DVN1/S7DNJG', license='CC BY 4.0',
        model=name, model_sha256=manifest['sha256'], threshold=threshold,
        target='Same-station ATMOS41 activity in next six 5-minute records; not classified CG',
        history_minutes=150, forecast_minutes=30, raw_features=manifest['raw_features'],
        engineered_features=106, trained_stations=['TA00804','TA00805'],
        training_before='2025-02-17', validation_before='2025-04-13',
        timing='Source timestamp labels; nominal 5-minute records, no sub-minute event time or sensor latency audit',
        model_refitted=False, threshold_retuned=False, reused_previous_holdout=True,
        verification='Raw SHA256, timestamps, rebuilt future labels, quiet flags, fresh model predictions and confusion counts matched',
        all_windows=metrics(y, scores >= threshold), quiet_history_windows=metrics(y[quiet], scores[quiet] >= threshold),
        persistence=metrics(y, np.array(all_baseline) >= .5), stations=stations, hashes=hashes,
        quiet_onsets=dict(eligible=onset_total, hits=len(onset_hits), misses=onset_total-len(onset_hits),
            nominal_lead_minutes=onset_hits, definition='Activity after 150 minutes without recorded activity; six preceding valid scored windows required'),
        example_selection='First qualifying timestamp in sorted station file order; illustrative, not random or representative',
        device_validation=False, verified_cloud_to_ground=False, calibrated_probability=False)
    for filename, obj in [('summary.json', summary), ('examples.json', examples)]:
        (out/filename).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (out/'SOURCE.txt').write_text(
        'Nenwala, Victor Hugo; Foulna Tcheobe, Carmel; Lavarenne, Jérémy\n'
        'Weather Station Data from Northern Cameroon (January 2024 - June 2025)\n'
        'CIRAD Dataverse; https://doi.org/10.18167/DVN1/S7DNJG\n'
        'CC BY 4.0 https://creativecommons.org/licenses/by/4.0/\n'
        'Derived evaluation and plots; observations transformed into causal features and future activity labels. No endorsement implied.\n',encoding='utf-8')
    render(out, summary, examples)
    print(json.dumps(summary['all_windows']), flush=True)
    print('Quiet onsets:', summary['quiet_onsets'], flush=True)


def render(out, summary, examples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    labels = {'hit':'경보 성공 시점', 'miss':'놓친 시점', 'false_alarm':'오경보 시점'}
    for kind, example in examples.items():
        fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True, layout='constrained')
        t = np.array(example['times'], dtype='datetime64[m]')
        anchor = np.datetime64(example['anchor'])
        axes[0].plot(t, example['scores'], label='외부 모델 점수', color='#2266bd')
        axes[0].axhline(summary['threshold'], color='#b42e49', ls='--', label='고정 경보 기준')
        axes[0].axvspan(anchor, anchor+np.timedelta64(30,'m'), alpha=.17, color='orange', label='선택 시점 이후 30분')
        axes[0].axvline(anchor, color='black', ls=':'); axes[0].set(ylim=(-.03,1.05), ylabel='점수 (확률 보정 없음)', title=f"{labels[kind]} · {example['station']} · {example['anchor']}")
        axes[0].legend(loc='upper left', fontsize=8)
        axes[1].bar(np.array(example['observation_times'],dtype='datetime64[m]'), example['observation_counts'], width=4/1440, color='#d47a22')
        axes[1].set(ylabel='센서 활동 횟수', xlabel='원자료 기록 시각 · 5분 간격 (지면 낙뢰 위치 정답 아님)')
        axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.savefig(out/f'{kind}.png', dpi=140); plt.close(fig)
    def pct(value):
        return '해당 없음' if value is None else f'{value*100:.2f}%'
    rows = ''
    for name, key in [('외부 모델 · 전체','all_windows'),('외부 모델 · 이전 150분 무감지','quiet_history_windows'),('최근 30분 활동 지속 가정','persistence')]:
        m=summary[key]
        rows += f"<tr><td>{name}</td><td>{m['windows']:,}</td><td>{pct(m['recall'])}</td><td>{pct(m['precision'])}</td><td>{pct(m['false_positive_rate'])}</td><td>{m['tp']} / {m['fn']} / {m['fp']} / {m['tn']}</td></tr>"
    strows=''.join(f"<tr><td>{s['station']}</td><td>{s['first']} ~ {s['last']}</td><td>{s['windows']:,}</td></tr>" for s in summary['stations'])
    cards=''
    for i,(kind,e) in enumerate(examples.items()):
        cards+=f'<article id="{kind}" '+('hidden' if i else '')+f'><h2>{labels[kind]}</h2><p>선택 시점 {html.escape(e["anchor"])} · 점수 {e["score"]:.4f} · 다음 30분 활동 정답 {e["label"]} · 이전 150분 무감지: {"예" if e["quiet_history"] else "아니오"}</p><img src="{kind}.png" alt="예측 점수와 실제 센서 활동 기록 비교"></article>'
    onset=summary['quiet_onsets']
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>외부 실측 평가</title>
<style>body{{font:17px 'Malgun Gothic',sans-serif;background:#f2f5fa;color:#18253c;margin:0;padding:24px}}main{{max-width:1150px;margin:auto}}section{{background:white;border-radius:12px;margin:18px 0;padding:24px}}p,li{{line-height:1.8}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}.scroll{{overflow:auto}}img{{width:100%}}select{{font:inherit;padding:10px}}.notice{{background:#fff1d3}}a{{color:#205db0}}</style><main>
<h1>외부 실측 자료로 다음 30분의 센서 활동 예측</h1>
<section class="notice"><b>실제 관측 자료 · 기존 10개 센서 LSTM과 별도 실험</b><p>카메룬 CIRAD/TAHMO 관측소의 ATMOS41 기록을 사용합니다. 정답은 센서의 낙뢰 활동 기록이며, 지면 낙뢰(CG)의 발생 위치·시각을 독립적으로 확인한 자료가 아닙니다. 한국 장비 검증 및 실제 보호 제어는 하지 않았습니다. 점수는 보정된 실제 발생 확률이 아닙니다.</p></section>
<section><h2>무엇을 입력하고 예측했나</h2><p>기압·기온·습도·풍속·돌풍·강수·일사량·과거 낙뢰 활동의 8개 관측 → 최근 150분(30개 기록)의 106개 특징 → HistGradientBoosting → 다음 30분(6개 기록)의 활동 유무.</p><p>TA00804·TA00805의 2025-02-17 이전 기록으로 학습하고, 그 이후 2025-04-13 이전 기록으로 모델과 기준을 선택했습니다. 다른 5곳의 2025-04-13 이후 기간에서 평가했습니다. 아래 결과는 기존 시험 자료의 재현 확인이며 새로운 독립 시험은 아닙니다. 모델·경보 기준을 재조정하지 않았습니다.</p><p>고정 경보 기준: {summary['threshold']:.6f}. 표의 단위는 겹치는 5분 간격 예측 창이며 독립적인 낙뢰 건수가 아닙니다.</p></section>
<section><h2>실제 기록과 비교한 평가</h2><div class="scroll"><table><tr><th>조건</th><th>평가 창</th><th>재현율</th><th>정밀도</th><th>음성 창 오경보율</th><th>TP / FN / FP / TN</th></tr>{rows}</table></div><p>재현율은 활동이 이어진 창을 얼마나 잡았는지, 정밀도는 경보한 창 중 실제 활동이 뒤따른 비율입니다. 음성 창 오경보율은 활동이 없던 창 중 경보한 비율이며, 전체 경보 중 오경보 비율과 다릅니다.</p><p><b>핵심 한계:</b> 이전 150분 동안 활동이 없었던 조건에서 재현율은 {pct(summary['quiet_history_windows']['recall'])}입니다. 첫 활동을 미리 경고하는 능력은 약하며, 전체 결과만으로 첫 낙뢰 예측에 성공했다고 말할 수 없습니다.</p><p>150분 무감지 뒤 시작된 활동 구간의 첫 기록을 별도로 평가하면 {onset['eligible']}개 중 {onset['hits']}개를 직전 30분 안에 경보했습니다. 이는 검증된 첫 지면 낙뢰 건수가 아닙니다. 선행시간은 5분 기록 시각 차이이며 정확한 방전 시각·통신 지연을 반영하지 않습니다.</p></section>
<section><h2>성공·실패 사례 비교</h2><select id="case">{''.join(f'<option value="{k}">{v}</option>' for k,v in labels.items() if k in examples)}</select><p>주황 구간은 선택 시점 이후 30분, 아래 막대는 실제 센서 활동 기록입니다. 선택 시점 자체의 활동은 미래 정답에서 제외합니다. 성공 시점에서도 이미 활동이 시작돼 있을 수 있으므로 첫 활동의 사전 예측 성공과는 다릅니다. 각 유형에서 관측소 파일 순서상 처음 발견한 시점을 보여줍니다. 대표성 있는 무작위 사례나 최고 성능 사례는 아닙니다. 감지 기록의 구름/지면 구분과 정확한 분 단위 시각은 확인되지 않았습니다.</p>{cards}</section>
<section><h2>자료 범위와 검증</h2><table><tr><th>시험 관측소</th><th>사용한 예측 시각 범위</th><th>창 수</th></tr>{strows}</table><p>원자료 SHA256, 기록 시각, 미래 정답, 무감지 조건, 저장 모델을 다시 실행한 점수와 평가표의 일치를 확인했습니다.</p><p><a href="https://doi.org/10.18167/DVN1/S7DNJG">원자료 CIRAD DOI · CC BY 4.0</a> · <a href="https://www.nature.com/articles/s41612-019-0098-0">방법 검토에 참고한 Mostajabi 등 연구</a> (해당 논문의 모델·성능을 재현했다는 의미는 아닙니다.)</p><p><a href="summary.json">평가·검증 근거 JSON</a> · <a href="examples.json">사례 그래프 수치 JSON</a> · <a href="발표설명.txt">실측 평가 발표 대본</a></p></section></main><script>document.getElementById('case').onchange=function(){{document.querySelectorAll('article').forEach(e=>e.hidden=e.id!==this.value)}};</script></html>'''
    page=page.replace('원자료 CIRAD DOI · CC BY 4.0</a>', '원자료 CIRAD DOI · CC BY 4.0</a> · <a href="SOURCE.txt">저자·자료명·가공 내역</a>')
    (out/'index.html').write_text(page,encoding='utf-8')
    speech=f'''외부 실측 평가 추가 발표 대본

회사 센서의 실제 기록을 확보하지 못해, 공개된 실제 관측 자료를 이용한 별도 예측 실험을 추가했습니다.
카메룬 관측소의 기상 및 낙뢰 활동 8개 변수를 사용합니다. 기존 장비 센서 10개를 대신 측정한 자료는 아닙니다.
최근 150분 기록에서 만든 특징으로 다음 30분 안에 센서 활동이 기록되는지 예측합니다. 모델은 LSTM과 별도의 부스팅 모델입니다.
두 관측소에서 학습과 검증을 하고 다른 다섯 관측소의 이후 기간을 평가했습니다. 이번에는 저장된 모델과 기존 시험 결과를 원자료로 재현 확인했습니다.
전체 {summary['all_windows']['windows']:,}개 예측 창에서 재현율은 {pct(summary['all_windows']['recall'])}, 정밀도는 {pct(summary['all_windows']['precision'])}입니다. 이 숫자는 독립적인 낙뢰 사건 수가 아닙니다.
과거 150분 동안 활동이 없었던 창에서는 재현율이 {pct(summary['quiet_history_windows']['recall'])}로 낮았습니다. 따라서 첫 낙뢰를 잘 예측하는 모델이라고 말할 수 없습니다.
화면에서 경보 성공 시점, 놓친 시점, 오경보 시점을 차례대로 선택하겠습니다. 위는 모델 점수와 기준이고, 아래는 센서가 실제 기록한 활동입니다. 주황 영역은 선택 시점 이후 30분입니다.
관측 센서가 구름 번개와 지면 낙뢰를 구분한 정답은 아니므로, 실제 장비 주변에 낙뢰가 떨어질 확률로 해석하면 안 됩니다.
이번 추가 작업의 성과는 합성 패턴 시험에서 나아가 실제 공개 관측에서 예측 결과와 한계를 확인했다는 것입니다. 회사 장비 적용에는 같은 장소의 센서 10개와 독립적인 낙뢰 기록으로 검증해야 합니다.
'''
    (out/'발표설명.txt').write_text(speech,encoding='utf-8')


if __name__ == '__main__':
    main()
