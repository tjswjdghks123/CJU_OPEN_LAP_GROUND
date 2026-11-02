"""Open saved five-week showcase without rerunning the evaluation."""
from pathlib import Path
import webbrowser

def main():
    page=Path(__file__).resolve().parent/'week5_simple'/'index.html'
    if not page.is_file():
        raise SystemExit('Missing week5_simple/index.html. Recalculate using ground.py --showcase.')
    print('5주차 시연 보완판: 저장된 LSTM 결과 재생 (재학습/실측 예측 아님)',flush=True)
    print(str(page),flush=True)
    webbrowser.open(page.as_uri())
