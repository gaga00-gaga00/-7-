# 음성 잡음 제거 챌린지 (신호및시스템 2026-2, 7조)

전통적 신호 처리만으로 16 kHz 음성의 잡음을 제거한다. 순위 = 히든 테스트셋 파일별 **PESQ-WB 평균** (보조: SNR).
팀 저장소: https://github.com/gaga00-gaga00/-7- (public, 조교가 clone해서 실행)

## 절대 규칙
- **딥러닝·머신러닝 금지.** 학습된 모델, 학습 가중치, 데이터로 fit하는 회귀/분류기(GMM, NMF 사전학습 등) 사용 금지. 파라미터를 train으로 *튜닝*하는 것은 허용.
- **Clean 음성은 평가에만.** `nr/`와 `denoise.py`는 noisy 입력만 읽는다. clean은 `evaluate.py`에서만 연다.
- **제출 입출력 계약:** `input/`의 모든 WAV → `output/`에 같은 이름(하위 폴더 유지), 같은 sr·채널·샘플 수·포맷. 절대 경로 금지 (`Path(__file__)` 기준).
- **실행 의존성은 `requirements.txt`만:** numpy, scipy, soundfile. pesq·pandas·pytest는 `requirements-dev.txt` (pesq는 Windows pip 빌드가 실패할 수 있음).
- 히든 테스트셋은 새로운 화자·잡음 → validation 점수에 과적합하지 말 것.

## 환경 / 명령
conda 환경 `denoise` (Python 3.12, conda-forge, pesq 0.0.4 — 조교 노트북과 동일). 반드시 활성화된 환경에서 실행:
`PYTHONIOENCODING=utf-8 conda run --no-capture-output -n denoise python <script>`
(env의 python.exe를 직접 부르면 libsndfile DLL 로드 실패. 콘솔이 cp949라 PYTHONIOENCODING 필요.)

| 목적 | 명령 |
|---|---|
| 테스트 | `python -m pytest -q` |
| 빠른 튜닝 (train 200개 층화) | `python evaluate.py --split train --subset quick --config configs/X.json --tag T` |
| 검증 | `python evaluate.py --config configs/X.json --tag T` |
| 파라미터 하나 바꿔 보기 | `python evaluate.py ... --set lsa_gain_floor=0.1` |
| 두 실행 비교 | `python compare.py <runA> <runB> [--method-a M --method-b M]` |
| 제출 시뮬레이션 | `python tools/check_submission.py --python "py -3.13"` |
| 제출본 검증 결과 갱신 | `python evaluate.py --out results/submission --tag submission` |

## 구조
- `nr/` 알고리즘: `config.py`(Config 데이터클래스, 모든 파라미터), `stft.py`, `noise.py`(잡음 PSD 추정기), `methods.py`(METHODS 레지스트리, `enhance`)
- `configs/submission.json` 제출 버전 메소드·파라미터 (denoise.py 기본값). 실험용 config는 `configs/`에 추가.
- `results/experiments.csv` 모든 실행 로그(커밋함). `results/runs/` 실행별 상세(무시). `results/submission/` 제출 버전 validation 결과(커밋함).
- `splits/train_quick.txt` 고정 튜닝 서브셋.
- `docs/ai_usage/` 회차별 AI 활용 내역서 — **비공개, git 제외(.gitignore)**. 커밋·push 금지. 생성: `python tools/build_ai_usage.py docs/ai_usage/roundN.json` (학교 hwpx 서식 `template.hwpx` 사용).
- `train/`, `validation/` 데이터는 git 제외.

## 실험 프로토콜
1. 새 아이디어는 `nr/`에 메소드/추정기로 추가하고 `METHODS`에 등록, 파라미터는 `Config`에 필드로.
2. train quick에서 튜닝 → validation 전체로 확인. `--note`에 무엇을 바꿨는지 기록.
3. `compare.py`로 잡음 종류·SNR별 손해 보는 그룹이 없는지 확인 (평균만 보지 말 것).
4. `pytest` 통과 후 커밋.

## 제출 체크리스트 (주 1회)
1. `pytest` 통과 → 2. `configs/submission.json` 갱신 → 3. `evaluate.py --out results/submission` →
4. README의 결과 표 갱신 → 5. `tools/check_submission.py` READY → 6. 커밋·push →
7. `docs/ai_usage/`에 해당 회차 AI 활용 내역서(hwpx, 학교 서식) 작성 → 팀장이 런어스 질의응답 비밀글로 링크+내역서 제출.

## 현재 기록 (validation PESQ-WB)
noisy 1.395 · basic 1.449 · spectral subtraction 1.538 · Wiener 1.598 · **MMSE-LSA 1.718 (제출 v1 후보)**
약점: 0 dB / hard 잡음에서 개선 작음, 충격성 잡음(bell, click, clap)에서 오히려 손해 → 시변 잡음 추정(MCRA/IMCRA, minimum statistics)이 다음 과제.
