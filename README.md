# 내 목소리를 구해줘! — 7조 음성 잡음 제거

2026-2 신호및시스템 조별 챌린지. 딥러닝/머신러닝 없이 **전통적 신호 처리**만으로 16 kHz 음성의 잡음을 제거합니다.

## 실행 방법 (제출 코드)

```bash
pip install -r requirements.txt     # numpy, scipy, soundfile
python denoise.py                   # ./input/*.wav  ->  ./output/*.wav
```

- `input/` 폴더의 모든 WAV(하위 폴더 포함)를 처리하여 `output/`에 **같은 파일명**으로 저장합니다.
- 출력은 입력과 **같은 샘플링 주파수, 채널 수, 샘플 수, 샘플 포맷**(예: 16-bit PCM)을 유지합니다. 다채널은 채널별로 처리합니다.
- 경로는 모두 `denoise.py` 위치 기준이므로 어디에 clone해도 동작합니다. 다른 폴더를 쓰려면
  `python denoise.py --input <입력폴더> --output <출력폴더>`.
- 알고리즘과 파라미터는 [`configs/submission.json`](configs/submission.json)에 포함되어 있습니다 (추가 설정 불필요).
- 확인된 환경: Python 3.12 (conda-forge), Python 3.13 (pip 새 venv).

## 알고리즘 (제출 v1)

STFT(Hann 32 ms, hop 8 ms) 영역에서 **MMSE Log-Spectral Amplitude 추정기** (Ephraim & Malah, 1985)를 적용합니다.

1. 잡음 PSD: 프레임 에너지가 가장 낮은 20% 프레임의 평균 파워 (주파수 방향 3-bin 평활)
2. 사후 SNR γ = |Y|² / λ_d, 사전 SNR ξ는 decision-directed 방식 (α = 0.98, 하한 −25 dB)
3. 이득 G = ξ/(1+ξ) · exp(½·E1(ξγ/(1+ξ))), 하한 0.05
4. 잡음 섞인 위상을 그대로 사용해 ISTFT (overlap-add)

## Validation 결과 (164개 파일, PESQ-WB 16 kHz / SNR)

| 방법 | PESQ | SNR (dB) |
|---|---|---|
| 잡음 입력 그대로 | 1.395 | 9.63 |
| 기본 필터 (HP+LP+BS) | 1.449 | 9.07 |
| Spectral Subtraction | 1.538 | 12.57 |
| Wiener (decision-directed) | 1.598 | 13.12 |
| **MMSE-LSA (제출 v1)** | **1.718** | **13.29** |

제출 버전의 파일별·잡음 종류별 결과: [`results/submission/`](results/submission/) ·
전체 실험 기록: [`results/experiments.csv`](results/experiments.csv)

## 개발 / 평가 (팀원용)

```bash
conda env create -f environment.yml && conda activate denoise    # pesq 0.0.4 포함
python -m pytest -q                                              # 입출력 계약 테스트
python evaluate.py --split train --subset quick --tag test       # 빠른 튜닝 (train 200개)
python evaluate.py --tag test                                    # validation 전체
python compare.py <runA> <runB>                                  # 두 실행 파일별 비교
python tools/check_submission.py                                 # 새 venv에서 제출 시뮬레이션
```

데이터(`train/`, `validation/`)는 저장소에 포함하지 않습니다. 배포받은 폴더를 저장소 루트에 두세요.
PESQ는 강의 노트북과 동일하게 정규화·시간 정렬 없이 계산하고 파일별 평균을 냅니다. Clean 음성은 평가에만 사용합니다.

| 경로 | 내용 |
|---|---|
| `denoise.py` | 제출 진입점 |
| `nr/` | 알고리즘 (STFT, 잡음 추정, 이득 함수) |
| `configs/` | 메소드·파라미터 설정 |
| `evaluate.py`, `compare.py` | 평가, 실행 간 비교 |
| `tests/`, `tools/` | 테스트, 제출 시뮬레이션 |
| `results/` | 실험 로그, 제출 버전 결과 |

## 참고문헌
- S. F. Boll, "Suppression of acoustic noise in speech using spectral subtraction," IEEE TASSP, 1979.
- Y. Ephraim, D. Malah, "Speech enhancement using a minimum mean-square error short-time spectral amplitude estimator," IEEE TASSP, 1984.
- Y. Ephraim, D. Malah, "Speech enhancement using a minimum mean-square error log-spectral amplitude estimator," IEEE TASSP, 1985.
