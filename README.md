# Bayesian Corrosion Rate Estimation

배관 두께 감육률(corrosion rate)을 베이지안 방법으로 추정하는 프로젝트.

12×13 = 156개 측정 포인트, 두께는 1로 노말라이즈, 최대 감육 ~20%.

## 파일 구조

| 파일 | 설명 |
|---|---|
| `test1.py` | **Empirical Bayes** — OLS + 풀링 표준오차 → 공식으로 사후분포 계산 (numpy only) |
| `test2.py` | **Full Bayes (MCMC)** — 계층적 베이지안 모델 (PyMC + numpyro) |
| `ba_1.py` | Bernoulli grid approximation 데모 (BernGrid 함수) |
| `ba_2.py` | 동전 던지기 베이지안 업데이트 시각화 (binomial likelihood) |

## 핵심 아이디어

1. 각 포인트별 OLS 선형회귀로 감육 기울기(beta) 추정
2. 잔차 SSE 풀링 → 표준오차 계산
3. 전체 beta 분포를 Prior로 사용 → 포인트별 사후분포 계산
4. **Order Statistic**: 156개 사후분포에서 min을 샘플링 → 최대 감육 포인트의 불확실성 추정

## 실행

```bash
# Empirical Bayes (빠름, numpy만 필요)
python test1.py

# MCMC 버전 (PyMC + numpyro 필요)
python test2.py
```

## 의존성

- numpy, matplotlib, scipy
- pymc, arviz, numpyro (test2.py만)
