# 내 목소리를 구해줘! — 7조 음성 잡음 제거

**2026 신호 및 시스템 1차 챌린지 — 전통적 신호 처리 기반 음성 잡음 제거** (2026-11-17 23:59 종료)

딥러닝/머신러닝 없이 **전통적 신호 처리**만으로 16 kHz 음성의 잡음을 제거합니다.
잡음 제거는 noisy 입력만으로 동작하며, clean 음성은 평가(`evaluate.py`)에서만 사용합니다.

## 제출 정보

<!-- submissions:start -->
팀명: **7조** · 저장소: https://github.com/gaga00-gaga00/-7-

| 회차 | 태그 | 커밋 | 런어스 제출일 | 방법 | Validation PESQ | Validation SNR (dB) | 히든 테스트 PESQ |
|---|---|---|---|---|---|---|---|
| 1 | `submission-1` | 태그 참조 | - | MMSE-LSA + decision-directed SNR, 최저 에너지 프레임 잡음 추정 | 1.718 | 13.29 | - |

평가할 코드 버전 = 해당 회차 **태그**가 가리키는 커밋 (`git checkout <태그>`). 회차별 Validation 결과: [`results/submissions/`](results/submissions/)
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
| **MMSE-LSA (제출 1회차)** | **1.718** | **13.29** |

PESQ는 강의 노트북과 동일하게 `pesq==0.0.4`, wideband 모드, 정규화·시간 정렬 없이 계산하고 파일별 단순 평균을 냅니다.
제출 버전의 파일별·잡음 종류별 결과: [`results/submissions/`](results/submissions/) ·
전체 실험 기록: [`results/experiments.csv`](results/experiments.csv)

## 개발 / 평가 (팀원용)

```bash
conda env create -f environment.yml && conda activate denoise    # pesq 0.0.4 포함
python -m pytest -q                                              # 입출력 계약 테스트
python evaluate.py --split train --subset quick --tag test       # 빠른 튜닝 (train 200개)
python evaluate.py --tag test                                    # validation 전체
python compare.py <runA> <runB>                                  # 두 실행 파일별 비교
python tools/spectrogram.py <segment_id> --config configs/X.json # 스펙트로그램 + 청취용 WAV
python tools/check_submission.py --round N                       # 제출 전 확인 (7개 항목)
```

데이터(`train/`, `validation/`)는 저장소에 포함하지 않습니다. 배포받은 폴더를 저장소 루트에 두세요.
Train으로 알고리즘·파라미터를 정하고, Validation으로 비교·확인합니다.

| 경로 | 내용 |
|---|---|
| `denoise.py` | 제출 진입점 |
| `nr/` | 알고리즘 (STFT, 잡음 추정, 이득 함수) |
| `configs/` | 메소드·파라미터 설정 (`submission.json` = 제출 버전) |
| `evaluate.py`, `compare.py` | 평가, 실행 간 비교 |
| `submissions.json` | 회차별 제출 기록 (태그, 커밋, 점수) |
| `tests/`, `tools/` | 테스트, 제출 점검·태그·스펙트로그램·내역서 도구 |
| `results/` | 실험 로그, 회차별 제출 버전 결과 |
| `docs/references.json` | 확인한 참고 문헌 (제목·저자·연도·DOI·참고 내용) |

## 참고문헌
- S. F. Boll, "Suppression of acoustic noise in speech using spectral subtraction," *IEEE Trans. ASSP*, vol. 27, no. 2, pp. 113–120, 1979. doi:10.1109/TASSP.1979.1163209
- Y. Ephraim, D. Malah, "Speech enhancement using a minimum-mean square error short-time spectral amplitude estimator," *IEEE Trans. ASSP*, vol. 32, no. 6, pp. 1109–1121, 1984. doi:10.1109/TASSP.1984.1164453
- Y. Ephraim, D. Malah, "Speech enhancement using a minimum mean-square error log-spectral amplitude estimator," *IEEE Trans. ASSP*, vol. 33, no. 2, pp. 443–445, 1985. doi:10.1109/TASSP.1985.1164550
