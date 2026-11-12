# 음성 잡음 제거 — 2026 신호 및 시스템 1차 챌린지 (7조)

**마감 2026-11-17 23:59.** 전통적 신호 처리만으로 16 kHz 음성의 잡음을 제거한다.
순위 = 히든 테스트셋 파일별 **PESQ-WB 평균** (보조: SNR). Validation 점수는 개발·비교용일 뿐이다.
팀 저장소: https://github.com/gaga00-gaga00/-7- (public, 조교가 clone해서 실행)

## 절대 규칙
- **딥러닝·머신러닝 금지.** 학습된 모델, 학습 가중치, 데이터로 fit하는 회귀/분류기(GMM, NMF 사전학습 등) 사용 금지. 파라미터를 train으로 *튜닝*하는 것은 허용. 입력 신호에 따른 자동 파라미터 조절은 허용(규칙은 코드에 포함).
- **Clean 음성은 평가에만.** `nr/`와 `denoise.py`는 noisy 입력만 읽는다. clean은 `evaluate.py`·`tools/`에서만 연다.
- **제출 입출력 계약:** `input/`의 모든 WAV → `output/`에 같은 이름 하나씩(하위 폴더 유지), 같은 sr·채널·샘플 수·포맷. 처리 실패 시 원본 복사로라도 출력은 반드시 생성. 절대 경로 금지 (`Path(__file__)` 기준).
- **실행 의존성은 `requirements.txt`만:** numpy, scipy, soundfile. pesq·pandas·pytest·matplotlib는 `requirements-dev.txt`.
- Train으로 개발·튜닝, Validation으로 비교·확인. 히든 테스트셋은 새로운 화자·잡음 → validation 점수에 과적합하지 말 것.
- **AI 참고 문헌은 실재·내용을 직접 확인한 것만.** `docs/references.json`의 `checked_by`(원문 확인한 팀원)는 사람이 채운다. AI가 채우지 말 것.

## 환경 / 명령
conda 환경 `denoise` (Python 3.12, conda-forge, pesq 0.0.4 — 조교 노트북과 동일). 반드시 활성화된 환경에서 실행:
`PYTHONIOENCODING=utf-8 conda run --no-capture-output -n denoise python <script>`
(env의 python.exe를 직접 부르면 libsndfile DLL 로드 실패. 콘솔이 cp949라 PYTHONIOENCODING 필요.)

| 목적 | 명령 |
|---|---|
| 테스트 | `python -m pytest -q` |
| 빠른 튜닝 (train 200개 층화) | `python evaluate.py --split train --subset quick --config configs/X.json --tag T` |
| 후보 확인 (train 전체 1,296개) | `python evaluate.py --split train --config configs/X.json --tag T` |
| 검증 (validation 전체 164개) | `python evaluate.py --config configs/X.json --tag T` |
| train + validation 한 번에 (split별 집계) | `python evaluate.py --split all --config configs/X.json --tag T` |
| 파라미터 하나 바꿔 보기 | `python evaluate.py ... --set lsa_gain_floor=0.1` |
| 두 실행 비교 | `python compare.py <runA> <runB> [--method-a M --method-b M]` |
| 스펙트로그램·청취 WAV | `python tools/spectrogram.py --pick worst --n 3` / `<segment_id> --config A.json --compare B.json` / `<id> --methods m1 m2 --added-noise` / `--noisy any.wav [--clean ref.wav]` |
| 데이터 개요 (분할·잡음·SNR 분포) | `python tools/explore_dataset.py` |
| split 전체 결과 WAV 저장 | `python tools/export_enhanced.py [--methods ...] [--split train --subset quick]` → `results/enhanced/` |
| 대화형 청취·비교 (Colab/Jupyter) | `notebooks/workbench.ipynb` (Colab: 저장소 clone + Drive 데이터 링크) |
| N회차 제출 결과 저장 | `python evaluate.py --round N` → validation 164개 `results/submissions/roundN/` + train 1,296개 `roundN/train/` |
| 제출 기록·README 표 | `python tools/release.py prepare N --summary "..."` → 커밋 → `release.py tag N` |
| 제출 전 확인 (공지 7항목) | `python tools/check_submission.py --round N --python "py -3.13"` |
| 참고 문헌 DOI 확인 | `python tools/check_refs.py --write` |
| AI 활용 내역서 | `python tools/build_ai_usage.py docs/ai_usage/roundN.json [--check]` |

## 구조
- `nr/` 알고리즘: `config.py`(Config 데이터클래스, 모든 파라미터, 생성 시 범위·NOLA 검증), `stft.py`, `noise.py`(잡음 PSD 추정기), `methods.py`(METHODS 레지스트리, `enhance`; 강의 베이스라인 + 개별 lowpass/highpass/bandstop + mmse_lsa)
- 강의 배포 노트북(`dataset_exploration.ipynb`, `noise_reduction_baselines.ipynb`, 루트, git 제외)의 코드는 모두 이식됨: 알고리즘 → `nr/`, 평가(manifest 검증, PESQ 오류 기록, 공통 부분집합) → `evaluate.py`, 탐색 → `tools/explore_dataset.py`, show_example/inspect_sample → `tools/spectrogram.py`, export_enhanced → `tools/export_enhanced.py`, 위젯 브라우저 → `notebooks/workbench.ipynb`. 노트북에 알고리즘을 따로 두지 말 것.
- `configs/submission.json` 제출 버전 메소드·파라미터 (denoise.py 기본값). 실험용 config는 `configs/`에 추가.
- `submissions.json` 회차별 제출 기록(태그, 커밋, 제출일, validation·히든 점수) → README `<!-- submissions -->` 표는 `release.py`가 생성. 직접 편집 금지.
- `results/experiments.csv` 모든 실행 로그(커밋함). `results/runs/` 실행별 상세(무시). `results/submissions/roundN/` 회차별 제출 버전 validation 결과(커밋함, `code_fingerprint`로 코드와 대조). `results/figures/` 스펙트로그램(무시).
- `tools/common.py` 팀명·저장소·태그 규칙(`submission-N`)·코드 지문.
- `docs/references.json` 참고 문헌(커밋함). `docs/ai_usage/` 회차별 AI 활용 내역서 — **비공개, git 제외**. 커밋·push 금지. 학교 서식 `template.hwpx`의 Part 1~2·종합 고찰에 공지 요구 항목(Part 3 알고리즘 이해, 4 참고 문헌, 5 개선 결과 표, 6 분석 및 고찰)을 추가해 생성.
- `splits/train_quick.txt` 고정 튜닝 서브셋. `train/`, `validation/` 데이터는 git 제외.

## 실험 프로토콜
1. 새 아이디어는 `nr/`에 메소드/추정기로 추가하고 `METHODS`에 등록, 파라미터는 `Config`에 필드로.
2. 데이터는 전부 쓴다: **train quick(200)으로 빠르게 튜닝 → train 전체(1,296)로 후보 확인 → validation 전체(164)로 최종 확인.** 파라미터는 train에서 고르고 validation은 확인용(validation에 맞춰 고르지 않기). `--note`에 무엇을 바꿨는지 기록.
3. `compare.py`로 잡음 종류·SNR별 손해 보는 그룹이 없는지 확인 (평균만 보지 말 것). `spectrogram.py`로 손해 본 파일을 보고 듣는다.
4. `pytest` 통과 후 커밋.

## 제출 절차 (팀당 주 1회, 최초 제출은 횟수 제외, 막바지 완화는 별도 공지)
1. `pytest` 통과 → `configs/submission.json` 갱신
2. `evaluate.py --round N` (train 1,296 + validation 164 전체, 약 2분) → `release.py prepare N --summary "..."` (README 알고리즘·결과 표도 손으로 갱신)
3. `docs/ai_usage/roundN.json` 작성(최초: 접근 방법 / 이후: 이전 회차 대비 개선 변화, results.rows에 `round{N-1}` vs `roundN`) → `build_ai_usage.py`
4. 커밋 → `release.py tag N` → **사용자 확인 후** `git push origin main` + `git push origin submission-N`
5. `check_submission.py --round N` → READY
6. 팀장이 런어스 → 질의응답 비밀글: 저장소 링크 + 태그 `submission-N`(커밋 해시) + 내역서 hwpx
7. `release.py submitted N` (커밋 해시·제출일 기록), 리더보드 공개 후 `release.py leaderboard N --pesq X`

## 현재 기록 (validation PESQ-WB)
noisy 1.395 · basic 1.449 · spectral subtraction 1.538 · Wiener 1.598 · **MMSE-LSA 1.718 (1회차, 아직 미제출)**
약점: 0 dB / hard 잡음에서 개선 작음, 충격성 잡음(bell, click, clap)·고 SNR 파일에서 오히려 손해(약한 고주파 음성 깎임) → 시변 잡음 추정(MCRA/IMCRA, minimum statistics)이 다음 과제.
