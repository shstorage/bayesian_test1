import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 설정: "cpu" 또는 "gpu"
# ============================================================
DEVICE = "cuda"  # "cpu" or "cuda"

import os
os.environ["JAX_PLATFORMS"] = DEVICE

import jax
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats

N_CHAINS = 4 if DEVICE == "cpu" else 1
print(f"[설정] device={DEVICE}, chains={N_CHAINS}, jax devices={jax.devices()}")

import platform
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 가상 데이터 생성 (12×13 = 156개 포인트)
# ============================================================
np.random.seed(42)
n_rows, n_cols = 12, 13
n_points = n_rows * n_cols  # 156

# 측정 시기 (모든 포인트 동일: 1, 3, 5)
times = np.array([1.0, 3.0, 5.0])
n_meas = len(times)

# 두께는 1로 노말라이즈, 최대 감육 ~20%
true_mu_beta = -0.02
true_sigma_beta = 0.008
noise_sd = 0.01

true_betas = np.zeros(n_points)
data_list = []

for i in range(n_points):
    true_beta_i = np.random.normal(true_mu_beta, true_sigma_beta)
    true_betas[i] = true_beta_i
    thickness = 1.0 + true_beta_i * times + np.random.normal(0, noise_sd, n_meas)
    for t, val in zip(times, thickness):
        data_list.append({'point_id': i, 'time': t, 'thickness': val})

df = pd.DataFrame(data_list)

# 비교용 OLS
ols_betas = np.zeros(n_points)
for i in range(n_points):
    sub = df[df.point_id == i]
    slope, _, _, _, _ = stats.linregress(sub.time.values, sub.thickness.values)
    ols_betas[i] = slope

print(f"데이터 포인트: {n_points}개, 측정시기: {times}")
print(f"t=5 기준 실제 두께 범위: {1.0 + true_betas.min()*5:.3f} ~ {1.0 + true_betas.max()*5:.3f}")

# ============================================================
# 2. 계층적 베이지안 모델 (MCMC)
# ============================================================
point_idx = df.point_id.values.astype(int)

with pm.Model() as model:
    # Hyperprior: 전체 그리드 공통
    mu_beta = pm.Normal("mu_beta", mu=-0.02, sigma=0.05)
    sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.05)

    # Prior: 개별 포인트
    beta_i = pm.Normal("beta_i", mu=mu_beta, sigma=sigma_beta, shape=n_points)
    alpha_i = pm.Normal("alpha_i", mu=1.0, sigma=0.1, shape=n_points)

    # 측정 노이즈
    eps = pm.HalfNormal("eps", sigma=0.05)

    # 선형 회귀
    mu_hat = alpha_i[point_idx] + beta_i[point_idx] * df.time.values

    # 관측값
    y_obs = pm.Normal("y_obs", mu=mu_hat, sigma=eps, observed=df.thickness.values)

    # MCMC
    trace = pm.sample(1000, tune=1000, target_accept=0.9, nuts_sampler="numpyro", chains=N_CHAINS)

# ============================================================
# 3. 사후분포에서 Order Statistic (min) 추출
# ============================================================
# beta_i 사후 샘플: (chain, draw, point) → (전체 샘플, point)
beta_samples = trace.posterior['beta_i'].values  # (chain, draw, 156)
n_chains, n_draws, _ = beta_samples.shape
beta_samples_flat = beta_samples.reshape(-1, n_points)  # (chain*draw, 156)

# 각 MCMC 샘플에서 156개 중 min → 최대 감육의 사후분포
min_per_sample = beta_samples_flat.min(axis=1)

# 포인트별 사후 평균
post_means = beta_samples_flat.mean(axis=0)
post_stds = beta_samples_flat.std(axis=0)

# ============================================================
# 4. 시각화
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) OLS vs MCMC 사후 평균 (shrinkage)
ax = axes[0, 0]
ax.scatter(ols_betas, post_means, alpha=0.5, s=15, label='포인트별')
lim = [min(ols_betas.min(), post_means.min()) - 0.005,
       max(ols_betas.max(), post_means.max()) + 0.005]
ax.plot(lim, lim, 'r--', label='y=x (shrinkage 없음)')
ax.axhline(np.mean(ols_betas), color='gray', linestyle=':', alpha=0.5,
           label=f'전체 평균={np.mean(ols_betas):.3f}')
ax.set_xlabel("OLS 기울기")
ax.set_ylabel("MCMC 사후 평균")
ax.set_title("(a) Shrinkage 효과 (MCMC)")
ax.legend(fontsize=8)

# (b) 가장 감육 심한 포인트의 사후분포
worst_idx = np.argmin(ols_betas)
ax = axes[0, 1]
worst_samples = beta_samples_flat[:, worst_idx]
ax.hist(worst_samples, bins=60, density=True, alpha=0.7, color='steelblue',
        label='MCMC 사후분포')
ax.axvline(ols_betas[worst_idx], color='green', linestyle='--',
           label=f'OLS={ols_betas[worst_idx]:.4f}')
ax.axvline(true_betas[worst_idx], color='red', linestyle='--',
           label=f'실제={true_betas[worst_idx]:.4f}')
ax.axvline(post_means[worst_idx], color='blue', linestyle=':',
           label=f'사후평균={post_means[worst_idx]:.4f}')
ax.set_title(f"(b) 포인트 {worst_idx} (OLS 최대감육) 사후분포")
ax.set_xlabel("기울기 (beta)")
ax.legend(fontsize=8)

# (c) 최대 감육 기울기의 분포 (Order Statistic)
ax = axes[1, 0]
ax.hist(min_per_sample, bins=80, density=True, alpha=0.7, color='steelblue',
        label='min(156개) 분포')
ax.axvline(true_betas.min(), color='red', linestyle='--',
           label=f'실제 min={true_betas.min():.4f}')
ax.axvline(np.mean(min_per_sample), color='blue', linestyle=':',
           label=f'추정 평균={np.mean(min_per_sample):.4f}')
ci_lower = np.percentile(min_per_sample, 2.5)
ci_upper = np.percentile(min_per_sample, 97.5)
ax.axvspan(ci_lower, ci_upper, alpha=0.15, color='blue',
           label=f'95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]')
ax.set_title("(c) 최대 감육 기울기 분포 (Order Statistic)")
ax.set_xlabel("기울기 (beta)")
ax.legend(fontsize=8)

# (d) t=5 최대 감육 두께 예측
ax = axes[1, 1]
t_pred = 5
pred_min_thickness = 1.0 + min_per_sample * t_pred
ax.hist(pred_min_thickness, bins=80, density=True, alpha=0.7, color='coral',
        label='최대 감육 포인트의\nt=5 두께 분포')
ax.axvline(1.0 + true_betas.min() * t_pred, color='red', linestyle='--',
           label=f'실제 최소 두께={1.0 + true_betas.min()*t_pred:.3f}')
ci_t_lower = np.percentile(pred_min_thickness, 2.5)
ci_t_upper = np.percentile(pred_min_thickness, 97.5)
ax.axvspan(ci_t_lower, ci_t_upper, alpha=0.15, color='red',
           label=f'95% CI=[{ci_t_lower:.3f}, {ci_t_upper:.3f}]')
ax.set_title(f"(d) t={t_pred} 최대 감육 두께 예측 (MCMC)")
ax.set_xlabel("노말라이즈 두께")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("mcmc_bayes_result.png", dpi=150)
plt.show()

# ============================================================
# 5. 결과 요약
# ============================================================
print("\n" + "="*50)
print("MCMC 결과 요약")
print("="*50)
print(f"mu_beta 사후평균: {trace.posterior['mu_beta'].mean().values:.4f}")
print(f"sigma_beta 사후평균: {trace.posterior['sigma_beta'].mean().values:.4f}")
print(f"eps 사후평균: {trace.posterior['eps'].mean().values:.4f}")
print(f"\n최대 감육 포인트 #{worst_idx}:")
print(f"  OLS 기울기:  {ols_betas[worst_idx]:.4f}")
print(f"  실제 기울기: {true_betas[worst_idx]:.4f}")
print(f"  사후 평균:   {post_means[worst_idx]:.4f}")
print(f"  사후 표준편차: {post_stds[worst_idx]:.4f}")
print(f"\n최대 감육 기울기 (Order Statistic from MCMC):")
print(f"  추정 평균: {np.mean(min_per_sample):.4f}")
print(f"  95% CI:    [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"\nt=5 최대 감육 두께:")
print(f"  추정 평균: {1.0 + np.mean(min_per_sample)*5:.3f}")
print(f"  95% CI:    [{ci_t_lower:.3f}, {ci_t_upper:.3f}]")
