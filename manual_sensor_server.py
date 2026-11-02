"""Loopback-only manual input laboratory. No hardware or external connections."""
import argparse
import copy
import json
from pathlib import Path
import secrets
from http.server import BaseHTTPRequestHandler, HTTPServer
import webbrowser
import numpy as np
from device_server import DeviceProcessor, RAW_FEATURES, RANGES
from event_evaluation import evaluate, parse_csv

ROOT = Path(__file__).resolve().parent


class InputSession:
    def __init__(self, predictor):
        self.predictor = predictor
        self.processor = DeviceProcessor()
        self.records = []

    def append(self, frames):
        if not isinstance(frames, list) or not 1 <= len(frames) <= 2000:
            raise ValueError('1~2000개의 입력 행이 필요합니다.')
        candidate = copy.deepcopy(self.processor)
        results = []
        for i, raw in enumerate(frames):
            if not isinstance(raw, dict):
                raise ValueError(f'{i+1}행: 객체가 필요합니다.')
            frame = dict(raw)
            frame.setdefault('time_s', 0 if candidate.last_time is None else candidate.last_time + 1)
            # Manually supplied complete rows are assumed fresh, not device telemetry.
            frame['age_seconds'] = {k: 0. for k in RAW_FEATURES}
            result = candidate.process(frame)
            if result['data_state'] == 'SENSOR_FAULT':
                raise ValueError(f'{i+1}행: {result["detail"]}')
            results.append({'time_s': frame['time_s'], 'values': [frame[k] for k in RAW_FEATURES],
                            'data_state': result['data_state'], 'buffer': len(candidate.rows),
                            'score': None, 'equipment': result['equipment_flags'],
                            'environment': result['environment_flags'],
                            'new_count': result['new_lightning_count'],
                            '_window': np.stack(candidate.rows) if result['model_ready'] else None})
        ready = [r for r in results if r['_window'] is not None]
        if ready:
            scores = self.predictor.predict_many(np.stack([r['_window'] for r in ready]))
            for r, score in zip(ready, scores):
                r['score'] = float(score)
        for r in results:
            r.pop('_window')
            r['alert'] = r['score'] is not None and r['score'] >= self.predictor.manifest['threshold']
        self.processor = candidate
        self.records.extend(results)
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8769)
    parser.add_argument('--no-open', action='store_true')
    args = parser.parse_args()
    from train_device_count import CountPredictor
    predictor = CountPredictor(ROOT / 'showcase_bundle')
    sessions = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, payload, mime='application/json; charset=utf-8'):
            data = payload.encode('utf-8') if isinstance(payload, str) else json.dumps(payload, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != '/':
                return self.send(404, {'error': 'Not found'})
            if len(sessions) >= 32:
                return self.send(429, {'error': '서버를 재시작해 세션을 비워 주세요.'})
            token = secrets.token_urlsafe(32)
            sessions[token] = InputSession(predictor)
            config = {'token': token, 'features': RAW_FEATURES, 'ranges': RANGES + [[0, 1000000000]],
                      'threshold': predictor.manifest['threshold']}
            page = (ROOT / 'manual_sensor.html').read_text(encoding='utf-8')
            page = page.replace('__EVENT_JS__', (ROOT / 'manual_events.js').read_text(encoding='utf-8'))
            self.send(200, page.replace('__CONFIG__', json.dumps(config)), 'text/html; charset=utf-8')

        def do_POST(self):
            origin = self.headers.get('Origin')
            expected = f'http://127.0.0.1:{self.server.server_port}'
            if origin not in (None, expected):
                return self.send(403, {'error': 'Local page only'})
            session = sessions.get(self.headers.get('X-Session'))
            if session is None:
                return self.send(403, {'error': '페이지를 새로 열어 주세요.'})
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 2000000:
                    raise ValueError('요청 크기가 너무 크거나 비어 있습니다.')
                payload = json.loads(self.rfile.read(size))
                if self.path == '/reset':
                    session.processor = DeviceProcessor()
                    session.records = []
                    return self.send(200, {'ok': True})
                if self.path == '/evaluate':
                    return self.send(200, evaluate(session.records, payload['events'], predictor.manifest['threshold'], payload.get('coverage')))
                if self.path == '/events-csv':
                    return self.send(200, {'events': parse_csv(payload['csv'])})
                if self.path != '/append':
                    return self.send(404, {'error': 'Not found'})
                self.send(200, {'records': session.append(payload['frames'])})
            except (ValueError, KeyError, TypeError, OverflowError) as error:
                self.send(400, {'error': str(error)})

    server = HTTPServer(('127.0.0.1', args.port), Handler)
    url = f'http://127.0.0.1:{server.server_port}/'
    print(f'Manual sensor laboratory: {url}', flush=True)
    print('Synthetic-trained model; no actual lightning probability or hardware commands. Ctrl+C to stop.', flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
