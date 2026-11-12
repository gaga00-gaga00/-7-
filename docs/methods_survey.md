# 전통적 음성 잡음 제거 기법 조사 (ML/DL 제외)

> 7조 · 신호및시스템 2026-2 · 작성 2026-11-12
> 기준점: **MMSE-LSA + decision-directed(DD) + 최저 에너지 프레임 잡음 추정 = validation PESQ-WB 1.718 (제출 v1)**
> 약점: 0 dB / hard 잡음에서 개선 작음, 충격성 잡음(bell, click, clap)에서 오히려 손해.

이 문서는 학습된 모델 없이 **신호 모델 + 통계적 추정 + 신호 분석**만으로 동작하는 기법 17개를 정리한다.
모든 기법은 입력(noisy) 신호 하나만 보고 동작하며, 파라미터는 train으로 튜닝만 한다(규칙 위반 아님).

---

## 0. 공통 표기와 현재 파이프라인

STFT 영역에서 $Y(k,l) = X(k,l) + D(k,l)$ ($k$: 주파수 bin, $l$: 프레임).

| 기호 | 의미 |
|---|---|
| $\lambda_d(k,l)$ | 잡음 PSD $E\|D\|^2$ |
| $\gamma = \|Y\|^2/\lambda_d$ | 사후 SNR (a posteriori) |
| $\xi = \lambda_x/\lambda_d$ | 사전 SNR (a priori) |
| $v = \xi\gamma/(1+\xi)$ | 이득 함수에 공통으로 등장하는 변수 |
| $G(k,l)$ | 스펙트럼 이득, $\hat X = G\,Y$ |

현재 제출본(v1)은 아래 4단계로 이루어져 있고, 조사한 기법들은 모두 이 중 **어느 한 칸을 교체**하는 모듈로 볼 수 있다.

```
 ① 잡음 PSD 추정   →  ② 사전 SNR 추정  →  ③ 이득 함수  →  ④ 후처리
 (최저 에너지 20%)     (DD, α=0.98)        (MMSE-LSA)      (gain floor 0.05)
```

**①이 가장 큰 병목이다.** 현재 추정은 파일 전체에서 고정된 잡음 PSD 하나만 쓰므로, 시간에 따라 변하는 잡음(babble, 거리, 충격음)을 전혀 따라가지 못한다. 0 dB/hard 잡음과 충격성 잡음에서의 손해가 정확히 이 증상이다.

---

## 요약표

| # | 기법 | 교체 단계 | 겨냥하는 약점 | 구현 난이도 | 기대 효과(주관) |
|---|---|---|---|---|---|
| 1 | Minimum Statistics (Martin 2001) | ① 잡음 추정 | 비정상 잡음 | 중 | 상 |
| 2 | MCRA / IMCRA (Cohen 2002·2003) | ① 잡음 추정 | 비정상 잡음, 저 SNR | 중~상 | 상 |
| 3 | MMSE-SPP 잡음 추정 (Gerkmann & Hendriks 2012) | ① 잡음 추정 | 비정상 잡음 | **하** | **상** |
| 4 | OM-LSA (Cohen 2001) | ③ 이득 | 잔여 잡음·뮤지컬 노이즈 | 중 | 상 |
| 5 | TSNR / HRNR (Plapous 2006) | ② 사전 SNR | DD의 1프레임 지연, 고조파 손실 | **하** | 중 |
| 6 | 켑스트럼 평활화 사전 SNR (Breithaupt 2008) | ② 사전 SNR | 뮤지컬 노이즈, 고조파 보존 | 중 | 중~상 |
| 7 | Super-Gaussian / β-order / 가중 추정기 | ③ 이득 | 음성 왜곡 vs 잔여 잡음 균형 | **하** | 중 |
| 8 | 다중대역·과감산 스펙트럼 차감 (Berouti, Kamath–Loizou) | ③ 이득 | 대역별 SNR 편차 | 하 | 하~중 |
| 9 | 청각 마스킹 기반 억제 (Virag 1999) | ③ 이득 | 청감상 잔여 잡음 | 상 | 중 |
| 10 | 반복 Wiener / LPC 칼만 필터 | 대체 파이프라인 | 저 SNR 스펙트럼 포락 | 중~상 | 중 |
| 11 | 신호 부분공간 (KLT, Ephraim & Van Trees 1995) | 대체 파이프라인 | 뮤지컬 노이즈 | 상 | 중 |
| 12 | TF 블록 임계화 (Yu, Mallat, Bacry 2008) | ③+④ | 뮤지컬 노이즈 | 중 | 중~상 |
| 13 | 뮤지컬 노이즈 후처리 (이득 평활화) | ④ 후처리 | 뮤지컬 노이즈 | 하 | 중 |
| 14 | 충격음 억제: HPSS 기반 과도 성분 검출 | ①/④ | **bell·click·clap** | 중 | 상(해당 그룹) |
| 15 | 클릭 검출·AR 보간 (Godsill & Rayner, Vaseghi) | 전처리(시간영역) | **click** | 중 | 중(해당 그룹) |
| 16 | 변조(modulation) 영역 스펙트럼 차감 (Paliwal 2010) | 대체 파이프라인 | 뮤지컬 노이즈 | 상 | 중 |
| 17 | 위상 스펙트럼 보상 (PSC, Wójcicki 2008) | 위상 | 위상 미처리 | 하 | 하~중 |

> "기대 효과"는 문헌 경향과 우리 약점에 근거한 **추정**이며 실측이 아니다. 실제 순위는 train quick → validation으로 확인해야 한다.

---

## A. 잡음 PSD 추정 (단계 ①) — 최우선

### 1. Minimum Statistics (Martin, 2001)

**아이디어.** 음성은 간헐적이므로, 각 주파수 bin에서 1~2초 창 안의 *평활화된 파워의 최솟값*은 거의 항상 잡음 수준을 가리킨다. VAD가 필요 없다.

**로직.**
1. 평활 파워: $P(k,l) = \alpha(k,l)P(k,l-1) + (1-\alpha(k,l))|Y(k,l)|^2$
   - 최적 시변 평활 계수: $\alpha_{opt} = 1 / \big(1 + (P(k,l-1)/\lambda_d(k,l-1) - 1)^2\big)$ — 음성이 있으면 빠르게, 잡음만 있으면 느리게 평활.
2. 길이 $D$ (약 1.5 s) 창에서 $P$의 최솟값 $P_{min}$을 추적. 메모리 절약을 위해 창을 $U$개의 하위 창(각 $V$프레임)으로 나누어 하위 창별 최솟값만 저장.
3. 최솟값은 평균보다 작으므로 편향 보정: $\lambda_d = B_{min}\cdot P_{min}$ ($B_{min}$은 창 길이와 평활 분산에 따라 1.5~3).
4. 잡음 상승을 빨리 따라가도록, 하위 창 최솟값들이 단조 증가하면 바로 갱신.

**장단점.** VAD 없이 비정상 잡음을 추적. 단, 잡음이 커질 때 최대 $D$만큼 지연되고 편향 보정 상수에 민감하다.
**파라미터.** $D$(창 길이), $U\times V$, $B_{min}$, $\alpha_{max}$(≈0.96).

### 2. MCRA / IMCRA (Cohen & Berdugo 2002, Cohen 2003)

**아이디어.** 최소 통계로 *음성 존재 확률* $p(k,l)$을 구하고, 그 확률로 잡음 갱신 속도를 조절한다(음성이 있을 땐 갱신 정지, 없을 땐 빠르게 갱신).

**로직 (MCRA).**
1. 주파수 방향 창 $b$로 평활 후 시간 평활: $S(k,l) = \alpha_s S(k,l-1) + (1-\alpha_s)\sum_i b(i)|Y(k-i,l)|^2$
2. $S_{min}$ 추적(최소 통계와 동일).
3. 비율 $S_r = S/S_{min}$ 이 임계값 $\delta$(≈5)를 넘으면 음성 존재 지시자 $I=1$.
4. $p(k,l) = \alpha_p p(k,l-1) + (1-\alpha_p) I(k,l)$
5. 시변 평활 계수: $\tilde\alpha_d = \alpha_d + (1-\alpha_d)p$
6. 잡음 갱신: $\lambda_d(k,l+1) = \tilde\alpha_d \lambda_d(k,l) + (1-\tilde\alpha_d)|Y(k,l)|^2$

**IMCRA의 개선점.** 평활·최소 추적을 **두 번** 반복한다. 1차 결과로 강한 음성 성분을 걸러낸 뒤(음성이 확실히 없는 bin만 사용) 2차 최소 추적을 하므로 편향이 작고 저 SNR에서도 안정적. 사전 음성 부재 확률 $q$도 함께 계산해 OM-LSA(#4)에 그대로 넘길 수 있다. 편향 보정 $\beta \approx 1.47$.

**장단점.** 문헌에서 비정상 잡음 추적 성능이 좋은 대표 기법. 파라미터가 많고 구현이 길다.

### 3. MMSE 기반 잡음 추정 + 음성 존재 확률 (Gerkmann & Hendriks, 2012) ★ 1순위 추천

**아이디어.** 각 bin에서 "음성 있음/없음" 두 가설의 사후 확률을 *고정된* 사전 SNR로 계산하고, 잡음 파워의 조건부 기댓값으로 갱신한다. 최소값 추적이 없어서 **편향 보정이 필요 없고 지연이 짧다.**

**로직.**
1. 직전 잡음 추정으로 $\gamma = |Y(k,l)|^2/\hat\lambda_d(k,l-1)$.
2. 음성 존재 사후 확률 (사전 확률 0.5, 음성 존재 시 고정 SNR $\xi_{H1}$ = 15 dB):
   $$P(H_1|Y) = \left[1 + (1+\xi_{H1})\exp\!\left(-\frac{\gamma\,\xi_{H1}}{1+\xi_{H1}}\right)\right]^{-1}$$
3. 정체 방지: $\bar P = 0.9\bar P + 0.1 P(H_1|Y)$ 를 유지하다가 $\bar P > 0.99$ 이면 $P(H_1|Y) \le 0.99$ 로 제한(잡음 수준이 갑자기 올라 계속 "음성"으로 판정되는 교착 방지).
4. 잡음 파워 기댓값: $E[|D|^2\,|\,Y] = P(H_0|Y)\,|Y|^2 + P(H_1|Y)\,\hat\lambda_d(k,l-1)$
5. 재귀 평활: $\hat\lambda_d(k,l) = \alpha_n\hat\lambda_d(k,l-1) + (1-\alpha_n)E[|D|^2|Y]$, $\alpha_n \approx 0.8$.
6. 초기값: 첫 몇 프레임(또는 현재의 최저 에너지 프레임 추정)으로 시작.

**장단점.** 코드 20줄 수준, 파라미터 3개($\xi_{H1}$, $\alpha_n$, 정체 임계). 현재 `nr/noise.py`에 추정기 하나를 추가하는 것만으로 MMSE-LSA와 바로 결합 가능. 충격음처럼 아주 짧은 잡음은 여전히 "음성"으로 판정될 수 있어 #14와 병행이 필요.

---

## B. 이득 함수 (단계 ③)

### 4. OM-LSA: Optimally-Modified LSA (Cohen & Berdugo, 2001)

**아이디어.** LSA 이득에 *음성 존재 확률*을 결합한다. 음성이 없는 bin은 일정한 최소 이득 $G_{min}$으로 균일하게 눌러 뮤지컬 노이즈를 줄이고, 음성이 있는 bin은 LSA 이득을 그대로 쓴다.

**로직.**
1. 음성 존재 확률: $p = \left[1 + \frac{q}{1-q}(1+\xi)\exp(-v)\right]^{-1}$ ($q$: 사전 음성 부재 확률)
2. $q$는 사전 SNR $\xi$를 국소(주파수 창 작음)/전역(창 큼)/프레임 단위로 평활한 세 값으로부터 휴리스틱하게 계산 (IMCRA가 제공하는 $q$를 써도 됨).
3. 최종 이득 (로그 진폭 기댓값을 두 가설로 섞음):
   $$G = G_{H1}^{\,p}\cdot G_{min}^{\,1-p},\qquad G_{H1} = \frac{\xi}{1+\xi}\exp\!\left(\tfrac12 E_1(v)\right)$$
4. $G_{min}$ ≈ −25 dB 내외.

**장단점.** 현재 `lsa_gain` 함수를 그대로 재사용할 수 있다. 잔여 잡음이 "고르게" 남아 PESQ에 유리한 경향. $q$ 추정 로직이 복잡하면 고정 $q$(예: 0.3~0.5)부터 시작해도 된다.

### 7. Super-Gaussian 사전분포 / β-order / 지각 가중 추정기

현재 MMSE-LSA는 음성 DFT 계수를 **가우시안**으로 가정한다. 실제 음성은 더 뾰족한(super-Gaussian) 분포를 가진다는 관찰과, 비용 함수를 바꾸면 청감상 더 좋다는 연구에서 나온 이득 함수 계열이다. 모두 닫힌 형태라 `scipy.special`(gamma, hyp1f1, i0e, i1e)로 바로 구현된다.

- **β-order MMSE (You, Koh, Rahardja 2005):** $\hat A^\beta$의 MMSE 추정.
  $$G = \frac{\sqrt v}{\gamma}\left[\Gamma\!\left(\tfrac\beta2+1\right)\,{}_1F_1\!\left(-\tfrac\beta2;1;-v\right)\right]^{1/\beta}$$
  $\beta=1$이면 MMSE-STSA, $\beta\to0$이면 LSA. $\beta$를 (또는 SNR에 따라 적응적으로) 튜닝해 억제 강도를 연속적으로 조절.
- **가중 유클리드 비용 (Loizou 2005):** 비용 $(A-\hat A)^2A^p$. $p<0$이면 약한 스펙트럼 성분(골짜기)의 오차를 더 크게 봐서 저에너지 음성 성분을 덜 깎는다.
  $$G = \frac{\sqrt v}{\gamma}\cdot\frac{\Gamma\!\left(\frac{p+3}{2}\right){}_1F_1\!\left(-\frac{p+1}{2};1;-v\right)}{\Gamma\!\left(\frac p2+1\right){}_1F_1\!\left(-\frac p2;1;-v\right)}$$
- **Super-Gaussian MAP 진폭 추정 (Lotter & Vary 2005):** 진폭 사전분포를 $p(a)\propto a^\nu e^{-\mu a}$ 로 두면
  $$G = u + \sqrt{u^2 + \frac{\nu}{2\gamma}},\qquad u = \frac12 - \frac{\mu}{4\sqrt{\gamma\xi}}$$
  문헌 값 $\mu\approx1.74,\ \nu\approx0.126$. 계산이 매우 가볍다.
- (참고) Martin(2005)의 Laplace/Gamma 사전분포 MMSE 추정기, Erkelens(2007) 일반화 감마 사전분포.

**장단점.** 단계 ①·②를 건드리지 않고 이득 함수만 교체하므로 실험이 쉽다. 단, 잡음 추정이 틀리면 어떤 이득 함수도 효과가 제한된다 → #3 이후에 시도.

### 8. 과감산·다중대역 스펙트럼 차감 (Berouti 1979, Kamath & Loizou 2002)

**로직.**
- Berouti: $|\hat X|^2 = \max(|Y|^2 - \alpha\lambda_d,\ \beta\lambda_d)$, 과감산 계수를 구간 SNR에 따라 변경 $\alpha = \alpha_0 - \tfrac{3}{20}\mathrm{SNR_{dB}}$ ($-5\le\mathrm{SNR}\le20$, $\alpha_0=4$). 바닥 $\beta$는 0.002~0.02.
- 다중대역(MBSS): 스펙트럼을 4~8개 대역으로 나눠 대역별 SNR로 $\alpha_i$를 정하고, 대역 가중치 $\delta_i$ (1 kHz 이하 1, 중간 2.5, 고대역 1.5)를 곱한다. 유색 잡음(저역에 몰린 차량 잡음 등)에서 대역별로 다른 강도로 차감.

**장단점.** 간단하지만 우리 실험에서 SS(1.538)는 LSA(1.718)보다 한참 낮다. 단독 후보보다는 "**대역/구간 SNR에 따라 억제 강도를 바꾼다**"는 아이디어를 LSA 이득 바닥(`lsa_gain_floor`)이나 #7의 β에 이식하는 용도가 유용하다.

### 9. 청각 마스킹 기반 억제 (Virag 1999; Tsoukalas 1997; Hu & Loizou 2004)

**아이디어.** 사람 귀에 들리지 않는(마스킹되는) 잔여 잡음은 굳이 지울 필요가 없다. 마스킹 임계값이 높은 곳은 약하게, 낮은 곳은 강하게 억제해 음성 왜곡을 줄인다. PESQ도 청각 모델 기반 지표라 방향이 일치한다.

**로직.**
1. 1차 추정(예: 현재 LSA 출력)으로 깨끗한 음성 스펙트럼 근사.
2. Bark 대역 에너지 → 확산 함수(spreading function) 컨볼루션 → 음조성(tonality, 스펙트럼 평탄도로 계산)에 따른 오프셋 차감 → 절대 가청 임계와 비교해 마스킹 임계 $T(k,l)$.
3. 차감/이득 파라미터를 $T$에 따라 보간: $T$가 최소일 때 $\alpha=\alpha_{max}$(≈6), 최대일 때 $\alpha=\alpha_{min}$(≈1). 바닥 $\beta$도 같은 방식.
4. 이 파라미터로 일반화 스펙트럼 차감 또는 이득을 계산.

**장단점.** 청감 품질 향상 보고가 많지만 MPEG 심리음향 모델 구현이 길고, 파라미터가 많다. 중후반 단계 후보.

---

## C. 사전 SNR 추정 개선 (단계 ②)

### 5. TSNR (Two-Step Noise Reduction) / HRNR (Harmonic Regeneration) (Plapous, Marro, Scalart 2006) ★ 2순위 추천

**문제.** DD 방식 $\hat\xi(l) = \alpha\frac{|\hat X(l-1)|^2}{\lambda_d} + (1-\alpha)\max(\gamma-1,0)$ 는 $\alpha=0.98$일 때 사실상 **이전 프레임**을 따라가므로, 음성 시작부에서 늦게 올라오고(앞 자음 손상) 끝부분에서 늦게 내려온다(잔향 같은 잔여 잡음).

**TSNR 로직.**
1. 1단계: 기존 DD로 $\xi_{DD}$ → 이득 $G_{DD}$.
2. 2단계: 현재 프레임 결과로 다시 사전 SNR을 추정 $\xi_{TSNR} = |G_{DD}(k,l)Y(k,l)|^2/\lambda_d$ (DD에서 $\alpha=1$을 현재 프레임에 적용한 것과 같음).
3. 최종 이득 $G_{TSNR} = \xi_{TSNR}/(1+\xi_{TSNR})$ (또는 LSA 이득에 $\xi_{TSNR}$ 대입).

**HRNR 로직 (TSNR 위에 추가).**
1. TSNR 출력을 시간 영역으로 복원 $\hat s(t)$.
2. 비선형 함수(반파 정류 $\max(\hat s,0)$ 등)를 적용 → 잃어버린 고조파가 재생성된 신호 $s_{harmo}(t)$, 그 스펙트럼 $S_{harmo}$.
3. $\xi_{HRNR} = \frac{\rho|G_{TSNR}Y|^2 + (1-\rho)|S_{harmo}|^2}{\lambda_d}$, $\rho\approx0.5$(튜닝).
4. 이 $\xi$로 최종 이득 계산.

**장단점.** 기존 코드에 몇 줄 추가로 구현 가능. 0 dB에서 고조파 사이 골짜기가 깎이는 문제를 완화.

### 6. 켑스트럼 영역 평활화 사전 SNR (Breithaupt, Gerkmann, Martin 2007·2008)

**아이디어.** 뮤지컬 노이즈는 사전 SNR이 시간-주파수에서 무작위로 튀는 데서 생긴다. 켑스트럼 영역에서 보면 **음성 정보는 소수의 계수(스펙트럼 포락=저켑스트럼, 피치=피치 켑스트럼 피크)** 에 몰려 있고 나머지는 잡음 요동이다. 그러니 음성 계수는 거의 평활하지 않고 나머지만 강하게 평활한다.

**로직.**
1. ML 추정 $\hat\lambda_x^{ML} = \max(|Y|^2 - \lambda_d,\ \xi_{min}\lambda_d)$ (또는 $|G Y|^2$).
2. 켑스트럼 $c(q,l) = \mathrm{IDFT}\{\log\hat\lambda_x^{ML}(k,l)\}$.
3. 피치 검출: 2.5~20 ms(16 kHz에서 $q$=40~320) 구간에서 켑스트럼 최댓값 → 피치 quefrency $q_0$.
4. quefrency별 평활 계수: 저 quefrency(포락, 약 $q<$ 1 ms)와 $q_0$ 주변은 $\alpha\approx0\sim0.2$(거의 평활 안 함), 나머지는 $\alpha\approx0.97$.
   $\bar c(q,l) = \alpha(q)\bar c(q,l-1) + (1-\alpha(q))c(q,l)$
5. 역변환 $\hat\lambda_x = \exp(\mathrm{DFT}\{\bar c\})$, 로그 평활에 따른 편향 보정 상수 곱함.
6. $\xi = \hat\lambda_x/\lambda_d$ 를 LSA 이득에 투입.

**장단점.** DD보다 음성 시작부 반응이 빠르면서도 뮤지컬 노이즈가 적다는 보고. 피치 검출이 필요하고 파라미터 튜닝이 조금 까다롭다.

---

## D. 다른 신호 모델 기반 파이프라인

### 10. 반복 Wiener / LPC 기반 칼만 필터 (Lim & Oppenheim 1978; Paliwal & Basu 1987; Hansen & Clements 1991)

**아이디어.** 음성은 성도(vocal tract)를 전극점(AR) 모델로 잘 표현된다: $x(n) = \sum_{i=1}^p a_i x(n-i) + g\,e(n)$. 이 모델로 깨끗한 음성 스펙트럼을 *포락 형태로 제한*해서 추정하면 저 SNR에서도 스펙트럼 모양이 무너지지 않는다. (프레임별 LPC 분석은 입력 신호 자체의 분석이지 학습이 아니므로 규칙 위반 아님.)

**반복 Wiener 로직.**
1. 프레임(20~32 ms)에서 noisy(첫 반복) 또는 직전 추정 음성으로 LPC 계수 $\{a_i\}$, 이득 $g$ 계산 ($p$=12~16).
2. 음성 PSD $P_x(\omega) = g^2/|A(e^{j\omega})|^2$.
3. Wiener 필터 $H(\omega) = P_x/(P_x + \lambda_d)$ 적용 → 새 음성 추정.
4. 2~3회 반복. 반복이 많아지면 포르만트가 지나치게 뾰족해지므로 Hansen–Clements의 제약(LSP 계수의 프레임 간 평활, 포르만트 대역폭 하한) 적용.

**칼만 필터 로직.** AR 계수로 상태공간 모델(상태 = 최근 $p$개 샘플)을 만들고, 관측 잡음 분산 = 잡음 추정으로 표본 단위 칼만 필터링. 유색 잡음은 잡음도 AR 모델로 상태에 추가(augmented state).

**장단점.** 백색·정상 잡음에 강하지만, 잡음 추정이 나쁘면 LPC 추정이 무너진다. 우리 파이프라인에서는 "LPC 포락으로 사전 SNR을 보정"하는 식의 부분 결합이 현실적.

### 11. 신호 부분공간 (KLT) 방법 (Ephraim & Van Trees 1995; Hu & Loizou 2003)

**아이디어.** 음성 프레임 벡터의 공분산은 랭크가 낮다(소수 고유벡터에 에너지 집중). 잡음은 모든 방향에 퍼져 있다. 고유분해로 신호 부분공간만 남기고 나머지를 버린다.

**로직.**
1. 길이 $K$(≈40 샘플) 벡터로 신호를 자르고, 수백 ms 구간에서 공분산 $R_y$ 추정.
2. 백색 잡음 가정: $R_x = R_y - \sigma^2 I$, 고유분해 $R_x = U\Lambda_xU^T$, 양수 고유값의 개수 = 신호 부분공간 차원 $M$.
3. TDC(시간영역 제약) 추정기: $H = U\,\mathrm{diag}(g_1..g_M,0..0)\,U^T$, $g_k = \frac{\lambda_{x,k}}{\lambda_{x,k} + \mu\sigma^2}$.
4. $\mu$는 프레임 SNR에 따라: $\mu = \mu_0 - \mathrm{SNR_{dB}}/s$ ($\mu_0\approx4.2,\ s\approx6.25$).
5. 유색 잡음: 잡음 공분산 $R_d$로 사전 백색화하거나 $R_d^{-1}R_y$의 일반화 고유분해 사용(Hu & Loizou).
6. 겹침-더하기로 재합성.

**장단점.** 뮤지컬 노이즈가 적다. 계산량이 크고(프레임마다 고유분해), 잡음 공분산 추정이 필요해 결국 #1~3이 선행되어야 한다.

### 12. 시간-주파수 블록 임계화 (Yu, Mallat, Bacry 2008)

**아이디어.** bin 하나씩 판단하면 무작위 요동이 그대로 이득에 실려 뮤지컬 노이즈가 된다. 시간-주파수 평면을 **블록**으로 묶어 블록 평균 SNR로 같은 감쇠를 주면 요동이 평균화된다. 블록 크기는 **SURE(Stein의 불편 위험 추정)** 로 해석적으로 고른다(학습 아님).

**로직.**
1. 주파수 대역별로 후보 블록 크기(예: 시간 1~8 프레임 × 주파수 1~16 bin) 목록 준비.
2. 블록 $B$에서 평균 사후 SNR $\bar\gamma_B = \frac1{|B|}\sum_{(k,l)\in B}|Y|^2/\lambda_d$.
3. 감쇠 $a_B = \max\!\left(1 - \frac{\lambda}{\bar\gamma_B},\ 0\right)$ (임계 $\lambda$는 블록 크기에 따라 약 4~5).
4. 각 대역·구간에서 후보 블록 크기별 SURE를 계산해 최소인 크기를 선택.
5. 선택된 블록 이득을 적용해 재합성.

**장단점.** 청감상 잔여 잡음이 매끄럽다. 우리 LSA와 결합하려면 "블록 평균 SNR을 LSA의 $\gamma$ 대신 쓰기" 같은 변형도 가능.

### 16. 변조(modulation) 영역 스펙트럼 차감 (Paliwal, Wójcicki, Schwerin 2010)

**아이디어.** 각 주파수 bin의 진폭 궤적 $|Y(k,l)|$을 *시간 신호*로 보고 한 번 더 STFT를 한다(변조 스펙트럼). 음성은 약 2~16 Hz 변조에 에너지가 몰려 있고, 정상 잡음은 0 Hz 근처에 몰려 있어 이 영역에서 분리가 더 잘 된다.

**로직.**
1. 음향 STFT(32 ms 프레임, 짧은 hop) → 진폭 $|Y(k,l)|$, 위상은 보관.
2. 각 $k$에 대해 $|Y(k,\cdot)|$ 를 변조 프레임(≈256 ms, hop ≈32 ms)으로 다시 STFT → $Z(k,m,\eta)$.
3. 변조 영역에서 스펙트럼 차감(잡음 변조 스펙트럼은 음성 부재 구간에서 추정).
4. 변조 역 STFT로 $|\hat X(k,l)|$ 복원 → 원래 위상과 합쳐 음향 역 STFT.

**장단점.** 일반 SS보다 PESQ가 높고 뮤지컬 노이즈가 적다는 보고. 계산량·지연이 크고, 결합할 때 이득 함수 대신 별도 파이프라인이 된다.

---

## E. 충격성·과도 잡음 대응 (bell, click, clap) — 우리 약점 직격

### 14. HPSS(median filtering) 기반 과도 성분 검출·억제 (FitzGerald 2010 응용)

**아이디어.** 스펙트로그램에서 지속음(음성 모음의 고조파, bell의 여운)은 **가로줄**, 클릭·박수 같은 충격음은 **세로줄**(짧고 광대역)이다. 시간 방향 median filter는 세로줄을, 주파수 방향 median filter는 가로줄을 제거한다.

**로직.**
1. $H(k,l) = \mathrm{median}_{l'}\{|Y(k,l')|\}$ (시간 방향, 길이 ≈ 17 프레임) — 지속 성분
2. $P(k,l) = \mathrm{median}_{k'}\{|Y(k',l)|\}$ (주파수 방향, 길이 ≈ 17 bin) — 과도 성분
3. 과도 마스크 $M_P = P^2/(H^2+P^2)$.
4. **음성 자음(파열음 /p/, /t/ 등)도 세로줄**이므로 그대로 지우면 안 된다. 다음 조건을 함께 사용:
   - 광대역성: 해당 프레임에서 $M_P$가 큰 bin이 전체 대역(특히 4 kHz 이상)에 걸쳐 있음
   - 짧은 지속: 1~3 프레임 이내의 에너지 급상승 후 급하강
   - 주변 프레임에 음성 고조파(유성음)가 없음
5. 충격 프레임으로 판정되면 (a) 해당 bin의 잡음 PSD를 일시적으로 $\lambda_d \leftarrow \max(\lambda_d, P^2)$ 로 올려 LSA가 억제하게 하거나, (b) 직접 이득을 $G_{min}$ 쪽으로 낮춘다.
6. bell처럼 고조파 지속음인 경우는 오히려 $H$ 성분 쪽에 잡히므로, #3(빠른 잡음 추적)과 결합해서 다룬다.

**장단점.** 현재 약점 그룹을 직접 겨냥. 자음 손상 위험이 있어 `compare.py`로 다른 그룹이 손해 보지 않는지 꼭 확인해야 함.

### 15. 클릭 검출 + AR 보간 (Vaseghi & Rayner 1990; Godsill & Rayner 1998)

**아이디어.** 레코드 복원에서 쓰는 고전 기법. 음성은 짧은 구간에서 AR 모델로 잘 예측되지만 클릭은 예측 불가능한 큰 오차를 만든다.

**로직.**
1. 블록(≈20~50 ms)마다 AR($p$≈20~40) 계수 추정 후 예측 오차 $e(n) = x(n) - \sum a_i x(n-i)$.
2. $|e(n)| > K\cdot\hat\sigma_e$ ($\hat\sigma_e$는 강건한 추정, 예: MAD; $K$≈4~6)인 구간을 클릭으로 표시, 앞뒤로 몇 샘플 확장.
3. 표시된 샘플을 **LSAR 보간**: 주변 샘플이 주어졌을 때 AR 예측 오차 에너지를 최소화하는 결측 샘플 값을 최소제곱으로 풀어 대체.
4. 이후 기존 STFT 잡음 제거 수행.

**장단점.** 수 ms 이하의 날카로운 클릭에 매우 효과적이고 음성 왜곡이 적다. 박수(수십 ms)처럼 긴 충격음엔 보간 구간이 길어져 한계 → #14와 역할 분담.

---

## F. 후처리·위상

### 13. 뮤지컬 노이즈 후처리 (이득 평활화)

**아이디어.** 이득 $G(k,l)$의 고립된 튐(음성 없는 곳에서 갑자기 큰 이득)이 "삐용삐용"하는 뮤지컬 노이즈다. 이득 자체를 후처리한다.

**로직 (대표 변형들).**
- **2D median / 평균 평활:** $G$에 3×3 median filter. 고립 피크 제거.
- **SNR 의존 평활 (Esch & Vary 2009 계열):** 프레임 평균 사전 SNR이 낮으면(잡음 구간) 주파수 방향으로 넓은 창으로 이득을 평활, 높으면(음성 구간) 평활하지 않음. 창 폭을 SNR에 대해 연속적으로 변화.
- **이득 하한 적응:** 잡음만 있는 프레임에서는 $G_{min}$을 조금 높여 잔여 잡음을 "고르게" 남김(comfort noise 효과). PESQ는 완전 무음보다 고른 잔여 잡음을 덜 감점하는 경향이 있다.
- **시간 방향 이득 평활(attack/release):** 이득이 떨어질 때는 천천히, 올라갈 때는 빠르게.

**장단점.** 매우 간단하고 어떤 이득 함수 뒤에도 붙일 수 있다. 과하면 음성 경계가 뭉개지므로 평활 강도를 SNR로 조절하는 것이 핵심.

### 17. 위상 스펙트럼 보상 (PSC: Wójcicki, Stark, Paliwal 2008) 및 위상 인지 처리

**아이디어.** 대부분의 기법은 진폭만 고치고 noisy 위상을 그대로 쓴다. PSC는 복소 스펙트럼에 *반대칭* 보상 항을 더해 위상을 바꾸고, 역 STFT 시 저 SNR 성분끼리 일부 상쇄되도록 만든다.

**로직.**
1. 보상 함수 $\Lambda(k) = \lambda\,\psi(k)\,|\hat D(k)|$, $\psi(k) = +1\ (0<k<N/2),\ -1\ (N/2<k<N)$, 0과 N/2에서는 0. 문헌 $\lambda \approx 3.74$.
2. 보상 스펙트럼 $Y_\Lambda = Y + \Lambda$, 위상 $\angle Y_\Lambda$.
3. 진폭은 기존 LSA 추정 $|\hat X|$를 쓰고 위상만 $\angle Y_\Lambda$로 교체해 재합성 (원 논문은 $|Y|$와 결합 후 실수부만 취함).

**장단점.** 구현 10줄. 효과는 진폭 개선보다 작다는 것이 일반적인 평가. 시간이 남을 때 시도.
(참고: Gerkmann·Krawczyk의 위상 인지 진폭 추정기, 고조파 모델 기반 위상 재구성(PBE)도 있으나 피치 추정이 정확해야 함.)

---

## G. 기타 참고 (우선순위 낮음 / 제외)

- **웨이블릿 임계화 (Donoho 1995, SURE/VisuShrink):** 백색 잡음에는 고전적이지만 음성에서는 STFT 기반 LSA보다 낮은 결과가 일반적. 비교용으로만 고려.
- **피치 동기 comb filter:** 유성음 고조파 사이를 깎는 필터. 피치 추정 오류 시 큰 왜곡 → HRNR(#5)이 더 안전한 대안.
- **제외(규칙 위반 가능성):** 사전 학습 NMF/사전(dictionary) 학습, GMM/HMM 기반 음성 모델, 모든 DNN. 입력 파일 하나에 비지도 NMF를 fit하는 것도 "데이터로 fit"에 해당할 수 있는 회색지대이므로 쓰지 않는다.

---

## H. 제안 로드맵 (현재 1.718에서 출발)

우리 약점(비정상·충격성 잡음, 0 dB)과 구현 비용을 함께 고려한 순서. 각 단계는 `nr/`에 추정기/메소드로 추가하고 `Config` 필드로 파라미터를 노출한 뒤, train quick → validation → `compare.py` 그룹별 확인 순서로 진행한다.

1. **잡음 추정 교체: MMSE-SPP(#3)** — `nr/noise.py`에 추정기 추가, `noise_estimator` 필드로 선택. 최저 에너지 추정은 초기값으로만 사용. → 비정상 잡음·0 dB 그룹 개선 기대.
   - 대안 비교: Minimum Statistics(#1), IMCRA(#2). 세 개를 같은 LSA 위에서 비교.
2. **사전 SNR 개선: TSNR(#5)** → 효과 있으면 HRNR 추가. 이어서 켑스트럼 평활화(#6)와 비교.
3. **이득 함수: OM-LSA(#4)** — IMCRA를 채택했다면 $q$를 그대로 연결. 그 다음 β-order / Lotter–Vary MAP(#7)로 억제 강도 미세 조정.
4. **충격음 대응: 클릭 AR 보간(#15) + HPSS 과도 검출(#14)** — bell/click/clap 그룹 점수만 따로 보고, 다른 그룹 손해가 없을 때만 채택.
5. **후처리: SNR 의존 이득 평활(#13)** 또는 블록 임계화(#12)로 뮤지컬 노이즈 정리.
6. 여력이 있으면 마스킹 기반(#9), 부분공간(#11), 변조 영역(#16), PSC(#17) 실험.

**과적합 주의:** 히든 테스트셋은 새로운 화자·잡음이므로, 특정 잡음 종류에만 맞춘 규칙(예: 특정 주파수 대역 고정 제거)보다 **잡음 종류와 무관하게 동작하는 추정기 개선(1~3단계)** 이 일반화에 더 안전하다.

---

## 참고문헌

1. Y. Ephraim, D. Malah, "Speech enhancement using a minimum mean-square error short-time spectral amplitude estimator," *IEEE Trans. ASSP*, 1984.
2. Y. Ephraim, D. Malah, "Speech enhancement using a minimum mean-square error log-spectral amplitude estimator," *IEEE Trans. ASSP*, 1985.
3. R. Martin, "Noise power spectral density estimation based on optimal smoothing and minimum statistics," *IEEE Trans. SAP*, 2001.
4. I. Cohen, B. Berdugo, "Speech enhancement for non-stationary noise environments," *Signal Processing*, 2001. (OM-LSA)
5. I. Cohen, B. Berdugo, "Noise estimation by minima controlled recursive averaging for robust speech enhancement," *IEEE SP Letters*, 2002. (MCRA)
6. I. Cohen, "Noise spectrum estimation in adverse environments: Improved minima controlled recursive averaging," *IEEE Trans. SAP*, 2003. (IMCRA)
7. T. Gerkmann, R. C. Hendriks, "Unbiased MMSE-based noise power estimation with low complexity and low tracking delay," *IEEE Trans. ASLP*, 2012.
8. C. Plapous, C. Marro, P. Scalart, "Improved signal-to-noise ratio estimation for speech enhancement," *IEEE Trans. ASLP*, 2006. (TSNR/HRNR)
9. C. Breithaupt, T. Gerkmann, R. Martin, "A novel a priori SNR estimation approach based on selective cepstro-temporal smoothing," *ICASSP*, 2008.
10. C. H. You, S. N. Koh, S. Rahardja, "β-order MMSE spectral amplitude estimation for speech enhancement," *IEEE Trans. SAP*, 2005.
11. P. C. Loizou, "Speech enhancement based on perceptually motivated Bayesian estimators of the magnitude spectrum," *IEEE Trans. SAP*, 2005.
12. T. Lotter, P. Vary, "Speech enhancement by MAP spectral amplitude estimation using a super-Gaussian speech model," *EURASIP JASP*, 2005.
13. R. Martin, "Speech enhancement based on minimum mean-square error estimation and supergaussian priors," *IEEE Trans. SAP*, 2005.
14. M. Berouti, R. Schwartz, J. Makhoul, "Enhancement of speech corrupted by acoustic noise," *ICASSP*, 1979.
15. S. Kamath, P. Loizou, "A multi-band spectral subtraction method for enhancing speech corrupted by colored noise," *ICASSP*, 2002.
16. N. Virag, "Single channel speech enhancement based on masking properties of the human auditory system," *IEEE Trans. SAP*, 1999.
17. J. S. Lim, A. V. Oppenheim, "All-pole modeling of degraded speech," *IEEE Trans. ASSP*, 1978.
18. K. K. Paliwal, A. Basu, "A speech enhancement method based on Kalman filtering," *ICASSP*, 1987.
19. J. H. L. Hansen, M. A. Clements, "Constrained iterative speech enhancement with application to speech recognition," *IEEE Trans. SP*, 1991.
20. Y. Ephraim, H. L. Van Trees, "A signal subspace approach for speech enhancement," *IEEE Trans. SAP*, 1995.
21. Y. Hu, P. C. Loizou, "A generalized subspace approach for enhancing speech corrupted by colored noise," *IEEE Trans. SAP*, 2003.
22. G. Yu, S. Mallat, E. Bacry, "Audio denoising by time-frequency block thresholding," *IEEE Trans. SP*, 2008.
23. T. Esch, P. Vary, "Efficient musical noise suppression for speech enhancement systems," *ICASSP*, 2009.
24. D. FitzGerald, "Harmonic/percussive separation using median filtering," *DAFx*, 2010.
25. S. V. Vaseghi, P. J. W. Rayner, "Detection and suppression of impulsive noise in speech communication systems," *IEE Proc.*, 1990.
26. S. J. Godsill, P. J. W. Rayner, *Digital Audio Restoration*, Springer, 1998.
27. K. Paliwal, K. Wójcicki, B. Schwerin, "Single-channel speech enhancement using spectral subtraction in the short-time modulation domain," *Speech Communication*, 2010.
28. K. Wójcicki, M. Milacic, A. Stark, J. Lyons, K. Paliwal, "Exploiting conjugate symmetry of the short-time Fourier spectrum for speech enhancement," *IEEE SP Letters*, 2008. (PSC)
29. P. C. Loizou, *Speech Enhancement: Theory and Practice*, 2nd ed., CRC Press, 2013. (전반 참고서)
