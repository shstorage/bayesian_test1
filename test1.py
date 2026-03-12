import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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
true_mu_beta = -0.02       # 전체 평균 감육률
true_sigma_beta = 0.008    # 포인트별 감육률 편차
noise_sd = 0.01            # 측정 노이즈

true_betas = np.zeros(n_points)
ols_betas = np.zeros(n_points)
ols_alphas = np.zeros(n_points)
residuals_all = []

for i in range(n_points):
    true_beta_i = np.random.normal(true_mu_beta, true_sigma_beta)
    true_betas[i] = true_beta_i

    thickness = 1.0 + true_beta_i * times + np.random.normal(0, noise_sd, n_meas)

    # OLS 선형회귀: thickness = alpha + beta * time
    slope, intercept, _, _, _ = stats.linregress(times, thickness)
    ols_betas[i] = slope
    ols_alphas[i] = intercept

    # 잔차 저장
    predicted = intercept + slope * times
    residuals_all.append(thickness - predicted)

residuals_all = np.concatenate(residuals_all)

print(f"데이터 포인트: {n_points}개, 측정시기: {times}")
print(f"t=5 기준 실제 두께 범위: {1.0 + true_betas.min()*5:.3f} ~ {1.0 + true_betas.max()*5:.3f}")

# ============================================================
# 2. 풀링 표준오차 계산
# ============================================================
# 각 포인트 자유도 = n_meas - 2 = 1, 전체 자유도 = 156 * 1 = 156
df_per_point = n_meas - 2
df_total = n_points * df_per_point
s_p = np.sqrt(np.sum(residuals_all**2) / df_total)  # 풀링 표준오차

# 기울기의 표준오차 (모든 포인트 동일한 시간 구조)
Sxx = np.sum((times - times.mean())**2)
se_beta = s_p / np.sqrt(Sxx)  # 각 포인트 beta의 Likelihood 폭

print(f"\n풀링 표준오차 s_p: {s_p:.4f}")
print(f"기울기 표준오차 se_beta: {se_beta:.4f}")

# ============================================================
# 3. Empirical Bayes: 사후분포 계산 (공식)
# ============================================================
# Prior: N(mu_prior, sigma_prior) ← 156개 OLS beta의 평균, 표준편차
mu_prior = np.mean(ols_betas)
sigma_prior = np.std(ols_betas, ddof=1)

print(f"\nPrior: N({mu_prior:.4f}, {sigma_prior:.4f})")
print(f"Likelihood 폭 (se_beta): {se_beta:.4f}")

# 정규분포 × 정규분포 = 정규분포 (closed-form)
# Posterior_i: N(post_mean_i, post_std)
prior_precision = 1.0 / sigma_prior**2
likelihood_precision = 1.0 / se_beta**2
post_precision = prior_precision + likelihood_precision
post_var = 1.0 / post_precision
post_std = np.sqrt(post_var)

# 각 포인트별 사후 평균 (포인트마다 다름)
post_means = (mu_prior * prior_precision + ols_betas * likelihood_precision) / post_precision

# 사후 표준편차 (모든 포인트 동일 - 같은 시간 구조이므로)
print(f"사후분포 표준편차 (전 포인트 동일): {post_std:.4f}")

# ============================================================
# 4. 최대 감육 포인트의 사후분포 (Order Statistic via Monte Carlo)
# ============================================================
n_sim = 100000

# 각 포인트 사후분포에서 샘플 뽑기 → 156개 중 min 기록
samples = np.random.normal(
    post_means.reshape(1, -1),     # (1, 156)
    post_std,                       # 스칼라 (전부 동일)
    size=(n_sim, n_points)          # (100000, 156)
)
min_per_sim = samples.min(axis=1)  # 각 시뮬레이션에서 156개 중 최솟값

# ============================================================
# 5. 시각화
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) OLS beta vs 사후 평균 (shrinkage 확인)
ax = axes[0, 0]
ax.scatter(ols_betas, post_means, alpha=0.5, s=15, label='포인트별')
lim = [min(ols_betas.min(), post_means.min()) - 0.005,
       max(ols_betas.max(), post_means.max()) + 0.005]
ax.plot(lim, lim, 'r--', label='y=x (shrinkage 없음)')
ax.axhline(mu_prior, color='gray', linestyle=':', alpha=0.5, label=f'Prior 평균={mu_prior:.3f}')
ax.set_xlabel("OLS 기울기")
ax.set_ylabel("사후 평균 기울기")
ax.set_title("(a) Shrinkage 효과")
ax.legend(fontsize=8)

# (b) 가장 감육 심한 포인트의 사후분포
worst_idx = np.argmin(ols_betas)
ax = axes[0, 1]
x_range = np.linspace(post_means[worst_idx] - 4*post_std,
                       post_means[worst_idx] + 4*post_std, 200)
ax.plot(x_range, stats.norm.pdf(x_range, post_means[worst_idx], post_std),
        'b-', lw=2, label=f'사후분포')
ax.axvline(ols_betas[worst_idx], color='green', linestyle='--',
           label=f'OLS={ols_betas[worst_idx]:.4f}')
ax.axvline(true_betas[worst_idx], color='red', linestyle='--',
           label=f'실제={true_betas[worst_idx]:.4f}')
ax.axvline(post_means[worst_idx], color='blue', linestyle=':',
           label=f'사후평균={post_means[worst_idx]:.4f}')
ax.set_title(f"(b) 포인트 {worst_idx} (OLS 최대감육) 사후분포")
ax.set_xlabel("기울기 (beta)")
ax.legend(fontsize=8)

# (c) 최대 감육 포인트의 사후분포 (Order Statistic)
ax = axes[1, 0]
ax.hist(min_per_sim, bins=80, density=True, alpha=0.7, color='steelblue',
        label='min(156개) 분포')
ax.axvline(true_betas.min(), color='red', linestyle='--',
           label=f'실제 min={true_betas.min():.4f}')
ax.axvline(np.mean(min_per_sim), color='blue', linestyle=':',
           label=f'추정 평균={np.mean(min_per_sim):.4f}')
# 95% credible interval
ci_lower = np.percentile(min_per_sim, 2.5)
ci_upper = np.percentile(min_per_sim, 97.5)
ax.axvspan(ci_lower, ci_upper, alpha=0.15, color='blue',
           label=f'95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]')
ax.set_title("(c) 최대 감육 기울기의 분포 (Order Statistic)")
ax.set_xlabel("기울기 (beta)")
ax.legend(fontsize=8)

# (d) 전체 사후분포 요약 + t=5 예측 두께
ax = axes[1, 1]
t_pred = 5
pred_thickness = 1.0 + post_means * t_pred
pred_min_thickness = 1.0 + min_per_sim * t_pred

ax.hist(pred_min_thickness, bins=80, density=True, alpha=0.7, color='coral',
        label='최대 감육 포인트의\nt=5 두께 분포')
ax.axvline(1.0 + true_betas.min() * t_pred, color='red', linestyle='--',
           label=f'실제 최소 두께={1.0 + true_betas.min()*t_pred:.3f}')
ci_t_lower = np.percentile(pred_min_thickness, 2.5)
ci_t_upper = np.percentile(pred_min_thickness, 97.5)
ax.axvspan(ci_t_lower, ci_t_upper, alpha=0.15, color='red',
           label=f'95% CI=[{ci_t_lower:.3f}, {ci_t_upper:.3f}]')
ax.set_title(f"(d) t={t_pred} 최대 감육 두께 예측")
ax.set_xlabel("노말라이즈 두께")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("empirical_bayes_result.png", dpi=150)
plt.show()

# ============================================================
# 6. 결과 요약 출력
# ============================================================
print("\n" + "="*50)
print("결과 요약")
print("="*50)
print(f"Prior: N({mu_prior:.4f}, {sigma_prior:.4f})")
print(f"Likelihood 폭 (se_beta): {se_beta:.4f}")
print(f"사후분포 폭 (전 포인트 동일): {post_std:.4f}")
print(f"\n최대 감육 포인트 #{worst_idx}:")
print(f"  OLS 기울기:  {ols_betas[worst_idx]:.4f}")
print(f"  실제 기울기: {true_betas[worst_idx]:.4f}")
print(f"  사후 평균:   {post_means[worst_idx]:.4f}")
print(f"\n최대 감육 기울기 (Order Statistic, 몬테카를로):")
print(f"  추정 평균: {np.mean(min_per_sim):.4f}")
print(f"  95% CI:    [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"\nt=5 최대 감육 두께:")
print(f"  추정 평균: {1.0 + np.mean(min_per_sim)*5:.3f}")
print(f"  95% CI:    [{ci_t_lower:.3f}, {ci_t_upper:.3f}]")
