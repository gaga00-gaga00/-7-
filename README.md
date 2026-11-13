# 내 목소리를 구해줘! — 7조 음성 잡음 제거

**2026 신호 및 시스템 1차 챌린지 — 전통적 신호 처리 기반 음성 잡음 제거** (2026-11-17 23:59 종료)

딥러닝/머신러닝 없이 **전통적 신호 처리**만으로 16 kHz 음성의 잡음을 제거합니다.
잡음 제거는 noisy 입력만으로 동작하며, clean 음성은 평가(`evaluate.py`)에서만 사용합니다.

## 제출 정보

<!-- submissions:start -->
팀명: **7조** · 저장소: https://github.com/gaga00-gaga00/-7-

| 회차 | 태그 | 커밋 | 런어스 제출일 | 방법 | Validation PESQ | Validation SNR (dB) | Train PESQ | Train SNR (dB) | 히든 테스트 PESQ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `submission-1` | 태그 참조 | - | 입력 적응형 OM-LSA: SPP 잡음 추적 + LSA 이득 + 배음 재생성, 특징량 기반 설정 혼합, 200 Hz 고역 통과 | 2.114 | 5.85 | 2.080 | 5.63 | - |

평가할 코드 버전 = 해당 회차 **태그**가 가리키는 커밋 (`git checkout <태그>`). Validation 164개·Train 1,296개 전체의 파일별 평균이며, 회차별 상세 결과: [`results/submissions/`](results/submissions/) (train은 `roundN/train/`)
<!-- submissions:end -->

## 실행 방법 (제출 코드)

Python 3.10 이상 (3.12, 3.13에서 확인).

```bash
git clone https://github.com/gaga00-gaga00/-7-.git
cd -7-
git checkout submission-1           # 평가할 회차의 태그
pip install -r requirements.txt     # numpy, scipy, soundfile
python denoise.py                   # ./input/*.wav  ->  ./output/*.wav
```

- `input/` 폴더의 모든 WAV(하위 폴더 포함)를 처리하여 `output/`에 **같은 파일명**으로 하나씩 저장합니다.
- 출력은 입력과 **같은 샘플링 주파수, 채널 수, 샘플 수, 샘플 포맷**(예: 16-bit PCM)을 유지합니다. 다채널은 채널별로 처리합니다.
- 어떤 파일을 처리하지 못하면 그 파일은 원본 그대로 복사해, 입력마다 출력이 반드시 하나 생깁니다.
- 경로는 모두 `denoise.py` 위치 기준이므로 어디에 clone해도 동작합니다. 다른 폴더를 쓰려면
  `python denoise.py --input <입력폴더> --output <출력폴더>`.
- 알고리즘·파라미터·선택 규칙은 모두 저장소에 포함되어 있습니다
  ([`configs/submission.json`](configs/submission.json), [`nr/`](nr/)). 추가 설정이나 외부 자료는 필요 없습니다.

## 알고리즘 (제출 1회차)

STFT(Hann 80 ms, hop 20 ms) 영역에서 칸마다 잡음 파워를 추정하고 이득을 곱하는 **입력 적응형 OM-LSA**입니다.
입력은 noisy WAV 하나뿐이며, 학습된 모델·가중치·사전은 사용하지 않습니다. 모든 규칙과 임계값은 코드([`nr/`](nr/))와
[`configs/submission.json`](configs/submission.json)에 들어 있습니다.

1. **잡음 PSD 추적**: 음성 존재 확률(SPP) 기반 MMSE 잡음 추정 (Gerkmann & Hendriks, 2012). 시간 순방향·역방향으로 추적해 기하평균
2. **이득**: MMSE-LSA (Ephraim & Malah, 1985), decision-directed 사전 SNR (α = 0.96, 하한 −30 dB), OM-LSA 형태 (Cohen & Berdugo, 2001)
3. **배음 재생성**: 1차 처리 결과를 전파 정류해 지워진 배음을 되살리고, 그것으로 사전 SNR을 다시 추정한 2차 이득을 1차 이득과 혼합 (Plapous 등, 2006의 방식)
4. **입력에 따른 설정 혼합** ([`nr/blend.py`](nr/blend.py)): noisy 파형에서 계산한 특징량 두 개(잡음 스펙트럼 평탄도, 300 Hz 이하 저역 비중)에
   시그모이드를 적용한 소속도(0~1)로 설정을 연속적으로 섞음 — 평탄한 잡음은 파일 고정 잡음 스펙트럼 쪽으로, 유색 잡음은 빠른 추적으로,
   저역 잡음이 적으면 더 강한 억제로
5. 완만한 200 Hz 고역 통과(2차 Butterworth, 영위상) 후 출력

파라미터는 Train 1,296개에서 정했고(같은 파일끼리 짝지은 PESQ 차이가 표준오차의 2배 이상일 때만 채택), Validation은 확인용으로만 사용했습니다.

## Validation 결과 (164개 파일, PESQ-WB 16 kHz / SNR)

| 방법 | PESQ | SNR (dB) |
|---|---|---|
| 잡음 입력 그대로 | 1.395 | 9.63 |
| 기본 필터 (HP+LP+BS) | 1.449 | 9.07 |
| Spectral Subtraction | 1.538 | 12.57 |
| Wiener (decision-directed) | 1.598 | 13.12 |
| MMSE-LSA (파일 고정 잡음 추정) | 1.718 | 13.29 |
| + SPP 잡음 추적 | 1.870 | 14.16 |
| + 입력 적응형 설정 혼합 | 1.941 | 11.73 |
| + 배음 재생성 (반파 정류) | 1.998 | 11.96 |
| **제출 1회차** (전파 정류, 80 ms 창, 200 Hz 고역 통과) | **2.114** | **5.85** |

PESQ는 강의 노트북과 동일하게 `pesq==0.0.4`, wideband 모드, 정규화·시간 정렬 없이 계산하고 파일별 단순 평균을 냅니다.
마지막 단계의 고역 통과는 PESQ를 올리지만 clean 음성의 저역 성분도 줄여 SNR(파형 오차 기준)을 낮춥니다. 순위 기준이 PESQ이므로 이를 선택했습니다.
제출 버전의 파일별·잡음 종류별 결과: [`results/submissions/`](results/submissions/) ·
전체 실험 기록: [`results/experiments.csv`](results/experiments.csv) ·
단계별 실험(오라클 진단, 특징량, 범주별 로직의 채택·기각): `results/stage*/`, `results/diagnose_all/`

## 개발 / 평가 (팀원용)

```bash
conda env create -f environment.yml && conda activate denoise    # pesq 0.0.4 포함
python -m pytest -q                                              # 입출력 계약 테스트
python evaluate.py --split train --subset quick --tag test       # 빠른 튜닝 (train 200개)
python evaluate.py --split train --tag test                      # train 전체 1,296개로 후보 확인
python evaluate.py --tag test                                    # validation 전체 164개로 최종 확인
python evaluate.py --split all --tag test                        # train + validation 한 번에
python compare.py <runA> <runB>                                  # 두 실행 파일별 비교
python tools/spectrogram.py <segment_id> --config configs/X.json # 스펙트로그램 + 청취용 WAV
python tools/explore_dataset.py                                  # 데이터 분포 요약
python tools/export_enhanced.py --methods wiener_filter mmse_lsa # split 전체 결과 WAV
python tools/check_submission.py --round N                       # 제출 전 확인 (7개 항목)
```

데이터(`train/`, `validation/`)는 저장소에 포함하지 않습니다. 배포받은 폴더를 저장소 루트에 두세요.
Train으로 알고리즘·파라미터를 정하고(빠른 튜닝 200개 → 전체 1,296개), Validation 164개로 비교·확인합니다.
제출 회차마다 Train 1,296개와 Validation 164개 **전체**의 결과를 저장합니다(`python evaluate.py --round N`).

| 경로 | 내용 |
|---|---|
| `denoise.py` | 제출 진입점 |
| `nr/` | 알고리즘 (STFT, 잡음 추정, 이득 함수, 특징량, 입력 적응형 혼합) |
| `configs/` | 메소드·파라미터 설정 (`submission.json` = 제출 버전) |
| `evaluate.py`, `compare.py` | 평가, 실행 간 비교 |
| `submissions.json` | 회차별 제출 기록 (태그, 커밋, 점수) |
| `tests/`, `tools/` | 테스트, 제출 점검·태그·스펙트로그램·데이터 탐색·내역서 도구 |
| `notebooks/workbench.ipynb` | Colab/Jupyter에서 샘플을 골라 방법별 스펙트로그램·청취 비교, 평가 실행 |
| `results/` | 실험 로그, 회차별 제출 버전 결과 |
| `docs/references.json` | 확인한 참고 문헌 (제목·저자·연도·DOI·참고 내용) |

## 참고문헌
- S. F. Boll, "Suppression of acoustic noise in speech using spectral subtraction," *IEEE Trans. ASSP*, vol. 27, no. 2, pp. 113–120, 1979. doi:10.1109/TASSP.1979.1163209
- Y. Ephraim, D. Malah, "Speech enhancement using a minimum-mean square error short-time spectral amplitude estimator," *IEEE Trans. ASSP*, vol. 32, no. 6, pp. 1109–1121, 1984. doi:10.1109/TASSP.1984.1164453
- Y. Ephraim, D. Malah, "Speech enhancement using a minimum mean-square error log-spectral amplitude estimator," *IEEE Trans. ASSP*, vol. 33, no. 2, pp. 443–445, 1985. doi:10.1109/TASSP.1985.1164550
- I. Cohen, B. Berdugo, "Speech enhancement for non-stationary noise environments," *Signal Processing*, vol. 81, no. 11, pp. 2403–2418, 2001. doi:10.1016/S0165-1684(01)00128-1
- T. Gerkmann, R. C. Hendriks, "Unbiased MMSE-based noise power estimation with low complexity and low tracking delay," *IEEE Trans. ASLP*, vol. 20, no. 4, pp. 1383–1393, 2012. doi:10.1109/TASL.2011.2180896
- C. Plapous, C. Marro, P. Scalart, "Improved signal-to-noise ratio estimation for speech enhancement," *IEEE Trans. ASLP*, vol. 14, no. 6, pp. 2098–2108, 2006. doi:10.1109/TASL.2006.872621
