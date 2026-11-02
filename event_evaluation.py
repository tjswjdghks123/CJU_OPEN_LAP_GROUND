"""Match frozen scores with separately supplied events; never creates labels from scores."""
import csv
import io
import math


def integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or value != int(value):
        raise ValueError(name + '는 0 이상의 정수 초여야 합니다.')
    return int(value)


def validate_events(events):
    if not isinstance(events, list) or len(events) > 2000:
        raise ValueError('사건은 최대 2000개입니다.')
    result = []
    seen = set()
    for e in events:
        if not isinstance(e, dict):
            raise ValueError('사건 형식 오류')
        t = integer(e.get('time_s'), '사건 시각')
        kind = e.get('kind')
        source = e.get('source')
        if kind not in ('virtual', 'observed') or not isinstance(source, str) or not source.strip() or len(source) > 200:
            raise ValueError('kind는 virtual/observed, source는 1~200자 출처여야 합니다.')
        if (kind, t) in seen:
            raise ValueError('같은 종류의 같은 초 사건은 한 번만 기록하세요.')
        seen.add((kind, t))
        result.append({'time_s': t, 'kind': kind, 'source': source.strip()})
    return sorted(result, key=lambda e: (e['time_s'], e['kind']))


def parse_csv(text):
    if not isinstance(text, str) or len(text) > 500000:
        raise ValueError('사건 CSV는 500KB 이하입니다.')
    reader = csv.DictReader(io.StringIO(text.lstrip('\ufeff')))
    if reader.fieldnames != ['time_s', 'kind', 'source']:
        raise ValueError('사건 CSV 열은 time_s,kind,source 순서입니다.')
    events = []
    for row in reader:
        if None in row or any(v is None for v in row.values()):
            raise ValueError('CSV 열 수가 맞지 않습니다.')
        events.append({**row, 'time_s': float(row['time_s'])})
        if len(events) > 2000:
            raise ValueError('사건은 최대 2000개입니다.')
    return validate_events(events)


def evaluate(records, events, threshold, coverage=None):
    events = validate_events(events)
    scores = {r['time_s']: r['score'] for r in records if r['score'] is not None}
    last = max((r['time_s'] for r in records), default=-1)
    coverage = coverage or {}
    output = {'horizon_seconds': 10, 'threshold': threshold, 'real_lightning_validated': False,
              'input_origin': 'user_supplied_unverified', 'groups': {}}
    for kind in ('virtual', 'observed'):
        selected = [e for e in events if e['kind'] == kind]
        rows = []
        for e in selected:
            t = e['time_s']
            window = list(range(t - 10, t))
            hits = [s for s in window if scores.get(s) is not None and scores[s] >= threshold]
            complete = all(s in scores for s in window)
            status = 'pending' if t > last else 'insufficient_history' if not complete else 'hit' if hits else 'miss'
            rows.append({**e, 'status': status, 'first_alert_s': min(hits) if hits else None,
                         'lead_seconds': t - min(hits) if hits else None, 'full_window': complete})
        eligible = [r for r in rows if r['status'] in ('hit', 'miss')]
        hits = sum(r['status'] == 'hit' for r in eligible)
        group = {'events': rows, 'hit': hits, 'miss': len(eligible)-hits,
                 'excluded': len(rows)-len(eligible), 'recall': hits / len(eligible) if eligible else None,
                 'negative_windows': None, 'false_alert_windows': None, 'false_alert_rate': None,
                 'false_alert_intervals': [], 'coverage': None}
        bounds = coverage.get(kind)
        if bounds is not None:
            if not isinstance(bounds, dict) or bounds.get('complete') is not True:
                raise ValueError('무사건 구간 평가에는 사건 기록 완전성 확인이 필요합니다.')
            start, end = integer(bounds.get('start'), '시작'), integer(bounds.get('end'), '끝')
            if start > end or end > last:
                raise ValueError('평가 끝은 현재 입력 시각 이하여야 하고 시작 이후여야 합니다.')
            # A prediction at s targets (s, s+10]; current-time events cannot count as prediction.
            times = [e['time_s'] for e in selected]
            negatives = [s for s in scores if start <= s and s+10 <= end and not any(s < t <= s+10 for t in times)]
            false = sorted(s for s in negatives if scores[s] >= threshold)
            intervals = []
            for s in false:
                if intervals and s == intervals[-1]['end']+1:
                    intervals[-1]['end'] = s
                    intervals[-1]['windows'] += 1
                else:
                    intervals.append({'start': s, 'end': s, 'windows': 1})
            group.update(negative_windows=len(negatives), false_alert_windows=len(false),
                         false_alert_rate=len(false)/len(negatives) if negatives else None,
                         false_alert_intervals=intervals, coverage=bounds)
        output['groups'][kind] = group
    return output
