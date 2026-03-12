"""
개선된 베이즈 방법: 국부 감육 문제 해결
==========================================

문제: 기존 계층적 베이즈에서 감육 없는 포인트(~85%)가 다수이므로
      전체 평균이 "감육 없음" 쪽에 치우쳐, 최대 감육 포인트가
      과소추정(비보수적)됨.

해결: 
  1) 기존 방법의 문제를 실제로 보여줌
  2) 방법 A: 2단계 접근 (감육 의심 포인트만 선별 → 베이즈)
  3) 방법 B: 혼합분포 프라이어 (감육/비감육을 자동 분류)
  4) 비교 분석
"""

import numpy as np
import json
from scipy import stats

np.random.seed(42)

# =============================================================
# 1. 데이터 로드
# =============================================================
with open('/home/claude/pipe_thinning_data.json', 'r') as f:
    raw = json.load(f)


def get_ols_slopes(dataset_dict):
    """각 포인트의 OLS 기울기와 잔차 계산"""
    times = np.array(dataset_dict['times'])
    data = np.array(dataset_dict['data'])
    n_meas, rows, cols = data.shape
    
    X = np.column_stack([np.ones(n_meas), times])
    XtX_inv = np.linalg.inv(X.T @ X)
    
    slopes = np.zeros((rows, cols))
    intercepts = np.zeros((rows, cols))
    residuals = np.zeros((rows, cols, n_meas))
    
    for i in range(rows):
        for j in range(cols):
            y = data[:, i, j]
            beta = XtX_inv @ (X.T @ y)
            intercepts[i, j] = beta[0]
            slopes[i, j] = beta[1]
            residuals[i, j, :] = y - X @ beta
    
    # 풀링된 sigma
    dof = rows * cols * (n_meas - 2)
    sigma_pooled = np.sqrt(np.sum(residuals**2) / dof)
    
    return slopes, intercepts, residuals, sigma_pooled, times, data


def bayesian_full_pool(dataset_dict, n_samples=5000, burnin=2000):
    """
    기존 방법: 전체 포인트를 하나의 계층으로 (문제 있는 방법)
    → 비감육 포인트가 감육 포인트를 과소추정하게 만듦
    """
    times = np.array(dataset_dict['times'])
    data = np.array(dataset_dict['data'])
    n_meas, rows, cols = data.shape
    n_points = rows * cols
    
    Y = np.zeros((n_points, n_meas))
    for k in range(n_points):
        i, j = k // cols, k % cols
        Y[k, :] = data[:, i, j]
    
    t = times
    t_c = t - t.mean()
    sum_t2 = np.sum(t_c**2)
    
    # 초기값 (OLS)
    X = np.column_stack([np.ones(n_meas), t_c])
    XtX_inv = np.linalg.inv(X.T @ X)
    
    alpha_k = np.zeros(n_points)
    beta_k = np.zeros(n_points)
    for k in range(n_points):
        coef = XtX_inv @ (X.T @ Y[k])
        alpha_k[k] = coef[0]
        beta_k[k] = coef[1]
    
    mu_beta = beta_k.mean()
    tau_beta = max(beta_k.std(), 0.001)
    sigma = 0.01
    mu_alpha = alpha_k.mean()
    tau_alpha = max(alpha_k.std(), 0.001)
    
    total = n_samples + burnin
    beta_samples = np.zeros((n_samples, n_points))
    
    for s in range(total):
        # β_k
        prec = sum_t2 / sigma**2 + 1.0 / tau_beta**2
        var_b = 1.0 / prec
        for k in range(n_points):
            resid = Y[k, :] - alpha_k[k]
            m = var_b * (np.sum(resid * t_c) / sigma**2 + mu_beta / tau_beta**2)
            beta_k[k] = np.random.normal(m, np.sqrt(var_b))
        
        # α_k
        prec_a = n_meas / sigma**2 + 1.0 / tau_alpha**2
        var_a = 1.0 / prec_a
        for k in range(n_points):
            resid = Y[k, :] - beta_k[k] * t_c
            m = var_a * (np.sum(resid) / sigma**2 + mu_alpha / tau_alpha**2)
            alpha_k[k] = np.random.normal(m, np.sqrt(var_a))
        
        # μ_β
        prec_mu = n_points / tau_beta**2 + 1.0 / 0.05**2
        mu_beta = np.random.normal(
            np.sum(beta_k) / tau_beta**2 / prec_mu, np.sqrt(1.0 / prec_mu))
        
        # μ_α
        prec_mu_a = n_points / tau_alpha**2 + 1.0 / 0.1**2
        mu_alpha = np.random.normal(
            (np.sum(alpha_k) / tau_alpha**2 + 1.0 / 0.1**2) / prec_mu_a,
            np.sqrt(1.0 / prec_mu_a))
        
        # τ_β
        a = 0.5 * n_points + 1
        b = 0.5 * np.sum((beta_k - mu_beta)**2) + 0.001
        tau_beta = np.sqrt(max(1.0 / np.random.gamma(a, 1.0 / b), 1e-10))
        
        # τ_α
        a_a = 0.5 * n_points + 1
        b_a = 0.5 * np.sum((alpha_k - mu_alpha)**2) + 0.001
        tau_alpha = np.sqrt(max(1.0 / np.random.gamma(a_a, 1.0 / b_a), 1e-10))
        
        # σ
        sse = sum(np.sum((Y[k] - alpha_k[k] - beta_k[k] * t_c)**2) for k in range(n_points))
        n_total = n_points * n_meas
        sigma = np.sqrt(max(1.0 / np.random.gamma(
            0.5 * n_total + 1, 1.0 / (0.5 * sse + 0.001)), 1e-10))
        
        if s >= burnin:
            beta_samples[s - burnin] = beta_k.copy()
    
    return beta_samples


def bayesian_two_stage(dataset_dict, threshold_percentile=25, 
                        n_samples=5000, burnin=2000):
    """
    방법 A: 2단계 접근
    1단계: OLS 기울기로 감육 의심 포인트 선별
    2단계: 선별된 포인트만으로 계층적 베이즈 적용
    
    핵심: 감육 포인트끼리만 정보를 공유!
    """
    times = np.array(dataset_dict['times'])
    data = np.array(dataset_dict['data'])
    n_meas, rows, cols = data.shape
    n_points = rows * cols
    
    # 1단계: OLS로 감육 의심 포인트 선별
    slopes_ols, _, _, sigma_ols, _, _ = get_ols_slopes(dataset_dict)
    slopes_flat = slopes_ols.flatten()
    
    # 임계값: 하위 percentile (음의 기울기가 큰 것)
    threshold = np.percentile(slopes_flat, threshold_percentile)
    thinning_mask = slopes_flat <= threshold
    thinning_indices = np.where(thinning_mask)[0]
    n_thinning = len(thinning_indices)
    
    print(f"  2단계 접근: {n_thinning}개 감육 의심 포인트 선별 "
          f"(전체 {n_points}개 중, 임계값={threshold*100:.3f} %/yr)")
    
    # 데이터 구성
    Y = np.zeros((n_points, n_meas))
    for k in range(n_points):
        i, j = k // cols, k % cols
        Y[k, :] = data[:, i, j]
    
    t = times
    t_c = t - t.mean()
    sum_t2 = np.sum(t_c**2)
    
    # 2단계: 선별된 포인트만으로 계층적 베이즈
    Y_sel = Y[thinning_indices]
    n_sel = n_thinning
    
    # 초기값
    X = np.column_stack([np.ones(n_meas), t_c])
    XtX_inv = np.linalg.inv(X.T @ X)
    
    alpha_k = np.zeros(n_sel)
    beta_k = np.zeros(n_sel)
    for idx, k in enumerate(thinning_indices):
        coef = XtX_inv @ (X.T @ Y[k])
        alpha_k[idx] = coef[0]
        beta_k[idx] = coef[1]
    
    mu_beta = beta_k.mean()
    tau_beta = max(beta_k.std(), 0.001)
    sigma = sigma_ols
    mu_alpha = alpha_k.mean()
    tau_alpha = max(alpha_k.std(), 0.001)
    
    total = n_samples + burnin
    beta_samples = np.zeros((n_samples, n_sel))
    
    for s in range(total):
        prec = sum_t2 / sigma**2 + 1.0 / tau_beta**2
        var_b = 1.0 / prec
        for idx in range(n_sel):
            resid = Y_sel[idx, :] - alpha_k[idx]
            m = var_b * (np.sum(resid * t_c) / sigma**2 + mu_beta / tau_beta**2)
            beta_k[idx] = np.random.normal(m, np.sqrt(var_b))
        
        prec_a = n_meas / sigma**2 + 1.0 / tau_alpha**2
        var_a = 1.0 / prec_a
        for idx in range(n_sel):
            resid = Y_sel[idx, :] - beta_k[idx] * t_c
            m = var_a * (np.sum(resid) / sigma**2 + mu_alpha / tau_alpha**2)
            alpha_k[idx] = np.random.normal(m, np.sqrt(var_a))
        
        prec_mu = n_sel / tau_beta**2 + 1.0 / 0.05**2
        mu_beta = np.random.normal(
            np.sum(beta_k) / tau_beta**2 / prec_mu, np.sqrt(1.0 / prec_mu))
        
        prec_mu_a = n_sel / tau_alpha**2 + 1.0 / 0.1**2
        mu_alpha = np.random.normal(
            (np.sum(alpha_k) / tau_alpha**2 + 1.0 / 0.1**2) / prec_mu_a,
            np.sqrt(1.0 / prec_mu_a))
        
        a = 0.5 * n_sel + 1
        b = 0.5 * np.sum((beta_k - mu_beta)**2) + 0.001
        tau_beta = np.sqrt(max(1.0 / np.random.gamma(a, 1.0 / b), 1e-10))
        
        a_a = 0.5 * n_sel + 1
        b_a = 0.5 * np.sum((alpha_k - mu_alpha)**2) + 0.001
        tau_alpha = np.sqrt(max(1.0 / np.random.gamma(a_a, 1.0 / b_a), 1e-10))
        
        sse = sum(np.sum((Y_sel[idx] - alpha_k[idx] - beta_k[idx] * t_c)**2) 
                  for idx in range(n_sel))
        n_total = n_sel * n_meas
        sigma = np.sqrt(max(1.0 / np.random.gamma(
            0.5 * n_total + 1, 1.0 / (0.5 * sse + 0.001)), 1e-10))
        
        if s >= burnin:
            beta_samples[s - burnin] = beta_k.copy()
    
    # 결과를 전체 그리드에 매핑
    beta_full = np.full((n_samples, n_points), np.nan)
    beta_full[:, thinning_indices] = beta_samples
    
    return beta_full, thinning_indices, thinning_mask


def bayesian_mixture(dataset_dict, n_samples=5000, burnin=2000):
    """
    방법 B: 혼합분포 프라이어
    β_k ~ π × N(μ₁, τ₁²) + (1-π) × N(μ₂, τ₂²)
    
    Component 1: 감육 없는 포인트 (기울기 ≈ 0)
    Component 2: 감육 있는 포인트 (기울기 < 0)
    
    각 포인트가 어떤 component에 속하는지도 추정!
    """
    times = np.array(dataset_dict['times'])
    data = np.array(dataset_dict['data'])
    n_meas, rows, cols = data.shape
    n_points = rows * cols
    
    Y = np.zeros((n_points, n_meas))
    for k in range(n_points):
        i, j = k // cols, k % cols
        Y[k, :] = data[:, i, j]
    
    t = times
    t_c = t - t.mean()
    sum_t2 = np.sum(t_c**2)
    
    # 초기값
    X = np.column_stack([np.ones(n_meas), t_c])
    XtX_inv = np.linalg.inv(X.T @ X)
    
    alpha_k = np.zeros(n_points)
    beta_k = np.zeros(n_points)
    for k in range(n_points):
        coef = XtX_inv @ (X.T @ Y[k])
        alpha_k[k] = coef[0]
        beta_k[k] = coef[1]
    
    # 혼합분포 초기값
    # Component 1: 비감육 (기울기 ≈ 0)
    mu1 = 0.0
    tau1 = 0.003
    # Component 2: 감육 (기울기 < 0)
    mu2 = np.percentile(beta_k, 10)
    tau2 = max(np.std(beta_k[beta_k < np.percentile(beta_k, 25)]), 0.002)
    
    pi_mix = 0.8  # 비감육 비율 (사전 추정)
    z_k = np.zeros(n_points, dtype=int)  # 0=비감육, 1=감육
    z_k[beta_k < np.percentile(beta_k, 25)] = 1
    
    sigma = 0.01
    mu_alpha = alpha_k.mean()
    tau_alpha = max(alpha_k.std(), 0.001)
    
    total = n_samples + burnin
    beta_samples = np.zeros((n_samples, n_points))
    z_samples = np.zeros((n_samples, n_points), dtype=int)
    mu2_samples = np.zeros(n_samples)
    tau2_samples = np.zeros(n_samples)
    pi_samples = np.zeros(n_samples)
    
    for s in range(total):
        # --- β_k 샘플링 (component에 따라 다른 프라이어) ---
        for k in range(n_points):
            if z_k[k] == 0:
                mu_prior, tau_prior = mu1, tau1
            else:
                mu_prior, tau_prior = mu2, tau2
            
            prec = sum_t2 / sigma**2 + 1.0 / tau_prior**2
            var_b = 1.0 / prec
            resid = Y[k, :] - alpha_k[k]
            m = var_b * (np.sum(resid * t_c) / sigma**2 + mu_prior / tau_prior**2)
            beta_k[k] = np.random.normal(m, np.sqrt(var_b))
        
        # --- α_k 샘플링 ---
        prec_a = n_meas / sigma**2 + 1.0 / tau_alpha**2
        var_a = 1.0 / prec_a
        for k in range(n_points):
            resid = Y[k, :] - beta_k[k] * t_c
            m = var_a * (np.sum(resid) / sigma**2 + mu_alpha / tau_alpha**2)
            alpha_k[k] = np.random.normal(m, np.sqrt(var_a))
        
        # --- z_k 샘플링 (component 할당) ---
        for k in range(n_points):
            # 각 component에서의 likelihood
            log_p0 = np.log(pi_mix + 1e-10) - 0.5 * ((beta_k[k] - mu1) / tau1)**2
            log_p1 = np.log(1 - pi_mix + 1e-10) - 0.5 * ((beta_k[k] - mu2) / tau2)**2
            
            # numerical stability
            max_log = max(log_p0, log_p1)
            p0 = np.exp(log_p0 - max_log)
            p1 = np.exp(log_p1 - max_log)
            prob1 = p1 / (p0 + p1)
            
            z_k[k] = 1 if np.random.random() < prob1 else 0
        
        # --- π 샘플링 (Beta-Binomial 켤레) ---
        n0 = np.sum(z_k == 0)
        n1 = np.sum(z_k == 1)
        pi_mix = np.random.beta(n0 + 2, n1 + 2)  # Beta(2,2) prior
        
        # --- μ₁ 고정 (비감육은 0 근처) ---
        # mu1은 0으로 고정
        
        # --- μ₂, τ₂ 샘플링 (감육 component) ---
        idx1 = z_k == 1
        if np.sum(idx1) > 1:
            beta_1 = beta_k[idx1]
            n1_count = len(beta_1)
            
            # μ₂ 
            prec_mu2 = n1_count / tau2**2 + 1.0 / 0.05**2
            mu2 = np.random.normal(
                np.sum(beta_1) / tau2**2 / prec_mu2, np.sqrt(1.0 / prec_mu2))
            
            # τ₂
            a2 = 0.5 * n1_count + 1
            b2 = 0.5 * np.sum((beta_1 - mu2)**2) + 0.001
            tau2 = np.sqrt(max(1.0 / np.random.gamma(a2, 1.0 / b2), 1e-10))
        
        # --- τ₁ 샘플링 (비감육 component) ---
        idx0 = z_k == 0
        if np.sum(idx0) > 1:
            beta_0 = beta_k[idx0]
            n0_count = len(beta_0)
            a0 = 0.5 * n0_count + 1
            b0 = 0.5 * np.sum((beta_0 - mu1)**2) + 0.001
            tau1 = np.sqrt(max(1.0 / np.random.gamma(a0, 1.0 / b0), 1e-10))
        
        # --- μ_α, τ_α ---
        prec_mu_a = n_points / tau_alpha**2 + 1.0 / 0.1**2
        mu_alpha = np.random.normal(
            (np.sum(alpha_k) / tau_alpha**2 + 1.0 / 0.1**2) / prec_mu_a,
            np.sqrt(1.0 / prec_mu_a))
        a_a = 0.5 * n_points + 1
        b_a = 0.5 * np.sum((alpha_k - mu_alpha)**2) + 0.001
        tau_alpha = np.sqrt(max(1.0 / np.random.gamma(a_a, 1.0 / b_a), 1e-10))
        
        # --- σ ---
        sse = sum(np.sum((Y[k] - alpha_k[k] - beta_k[k] * t_c)**2) 
                  for k in range(n_points))
        n_total = n_points * n_meas
        sigma = np.sqrt(max(1.0 / np.random.gamma(
            0.5 * n_total + 1, 1.0 / (0.5 * sse + 0.001)), 1e-10))
        
        if s >= burnin:
            idx = s - burnin
            beta_samples[idx] = beta_k.copy()
            z_samples[idx] = z_k.copy()
            mu2_samples[idx] = mu2
            tau2_samples[idx] = tau2
            pi_samples[idx] = pi_mix
        
        if (s + 1) % 2000 == 0:
            print(f"    MCMC {s+1}/{total} | π={pi_mix:.2f} | μ₂={mu2*100:.3f}%/yr | "
                  f"τ₂={tau2*100:.4f}%/yr | n_thinning={n1}")
    
    return beta_samples, z_samples, mu2_samples, tau2_samples, pi_samples


# =============================================================
# 실행 및 비교
# =============================================================
print("=" * 70)
print("국부 감육 문제: 기존 방법 vs 개선된 방법 비교")
print("=" * 70)

for ds_name, ds_key in [("Set A (3회 측정)", "set_A"), ("Set B (6회 측정)", "set_B")]:
    ds = raw['datasets'][ds_key]
    times = np.array(ds['times'])
    data = np.array(ds['data'])
    n_meas, rows, cols = data.shape
    
    print(f"\n{'='*70}")
    print(f"  {ds_name}")
    print(f"{'='*70}")
    
    # True 감육률 (데이터 생성 시 설정)
    true_rate = np.array(ds['annual_rate'])
    true_worst_idx = np.unravel_index(true_rate.argmax(), true_rate.shape)
    true_worst_rate = -true_rate[true_worst_idx]  # 음수로 변환
    print(f"\n  실제 최대 감육률: {true_worst_rate*100:.4f} %/yr at {true_worst_idx}")
    
    # OLS
    slopes_ols, _, _, sigma_ols, _, _ = get_ols_slopes(ds)
    ols_worst_idx = np.unravel_index(slopes_ols.argmin(), slopes_ols.shape)
    print(f"\n  [OLS] 최대 감육률: {slopes_ols.min()*100:.4f} %/yr at {ols_worst_idx}")
    
    # 기존 베이즈 (전체 풀링 - 문제 있는 방법)
    print(f"\n  --- 기존 베이즈 (전체 풀링) ---")
    beta_full = bayesian_full_pool(ds, n_samples=5000, burnin=2000)
    full_mean = beta_full.mean(axis=0).reshape(rows, cols)
    full_worst_idx = np.unravel_index(full_mean.argmin(), full_mean.shape)
    k_worst = full_worst_idx[0] * cols + full_worst_idx[1]
    full_ci = np.percentile(beta_full[:, k_worst], [2.5, 97.5])
    print(f"  최대 감육률: {full_mean[full_worst_idx]*100:.4f} %/yr at {full_worst_idx}")
    print(f"  95% CI: [{full_ci[0]*100:.4f}, {full_ci[1]*100:.4f}] %/yr")
    print(f"  ⚠ Shrinkage: {(slopes_ols.min() - full_mean.min())*100:.4f} %/yr "
          f"(전체 평균 쪽으로 당겨짐 → 비보수적!)")
    
    # 2단계 접근
    print(f"\n  --- 개선 방법 A: 2단계 접근 ---")
    beta_2stage, thin_idx, thin_mask = bayesian_two_stage(
        ds, threshold_percentile=25, n_samples=5000, burnin=2000)
    
    # 선별된 포인트 중 최대 감육
    valid_cols = thin_idx
    beta_sel = beta_2stage[:, valid_cols]
    sel_mean = beta_sel.mean(axis=0)
    worst_in_sel = np.argmin(sel_mean)
    worst_k = valid_cols[worst_in_sel]
    worst_i, worst_j = worst_k // cols, worst_k % cols
    sel_ci = np.percentile(beta_sel[:, worst_in_sel], [2.5, 97.5])
    print(f"  최대 감육률: {sel_mean[worst_in_sel]*100:.4f} %/yr at ({worst_i}, {worst_j})")
    print(f"  95% CI: [{sel_ci[0]*100:.4f}, {sel_ci[1]*100:.4f}] %/yr")
    
    # 혼합분포 프라이어
    print(f"\n  --- 개선 방법 B: 혼합분포 프라이어 ---")
    beta_mix, z_mix, mu2_mix, tau2_mix, pi_mix = bayesian_mixture(
        ds, n_samples=5000, burnin=2000)
    
    mix_mean = beta_mix.mean(axis=0).reshape(rows, cols)
    mix_worst_idx = np.unravel_index(mix_mean.argmin(), mix_mean.shape)
    k_worst_mix = mix_worst_idx[0] * cols + mix_worst_idx[1]
    mix_ci = np.percentile(beta_mix[:, k_worst_mix], [2.5, 97.5])
    
    # 감육 component 확률
    thinning_prob = z_mix.mean(axis=0).reshape(rows, cols)
    
    print(f"  최대 감육률: {mix_mean[mix_worst_idx]*100:.4f} %/yr at {mix_worst_idx}")
    print(f"  95% CI: [{mix_ci[0]*100:.4f}, {mix_ci[1]*100:.4f}] %/yr")
    print(f"  비감육 비율 (π): {pi_mix.mean():.3f}")
    print(f"  감육 component 평균: {mu2_mix.mean()*100:.4f} %/yr")
    
    # ===== 종합 비교 =====
    print(f"\n  {'='*60}")
    print(f"  종합 비교 (최대 감육 포인트)")
    print(f"  {'='*60}")
    print(f"  {'방법':<25} {'기울기 (%/yr)':>14} {'95% CI 하한':>14} {'95% CI 상한':>14}")
    print(f"  {'-'*25} {'-'*14} {'-'*14} {'-'*14}")
    print(f"  {'실제값':<25} {true_worst_rate*100:>14.4f} {'':>14} {'':>14}")
    
    # OLS CI
    se_ols = sigma_ols * np.sqrt(np.linalg.inv(
        np.column_stack([np.ones(n_meas), times]).T @ 
        np.column_stack([np.ones(n_meas), times]))[1, 1])
    dof_ols = rows * cols * (n_meas - 2)
    t_crit = stats.t.ppf(0.975, dof_ols)
    ols_ci = [slopes_ols.min() - t_crit * se_ols, slopes_ols.min() + t_crit * se_ols]
    print(f"  {'OLS':<25} {slopes_ols.min()*100:>14.4f} {ols_ci[0]*100:>14.4f} {ols_ci[1]*100:>14.4f}")
    
    print(f"  {'베이즈(전체풀링) ⚠':<25} {full_mean.min()*100:>14.4f} {full_ci[0]*100:>14.4f} {full_ci[1]*100:>14.4f}")
    print(f"  {'2단계 접근 ✓':<25} {sel_mean.min()*100:>14.4f} {sel_ci[0]*100:>14.4f} {sel_ci[1]*100:>14.4f}")
    print(f"  {'혼합분포 ✓':<25} {mix_mean.min()*100:>14.4f} {mix_ci[0]*100:>14.4f} {mix_ci[1]*100:>14.4f}")
    
    # 보수성 평가
    print(f"\n  보수성 평가 (CI 하한이 실제값보다 더 나쁜가?):")
    checks = [
        ("OLS", ols_ci[0]),
        ("베이즈(전체풀링)", full_ci[0]),
        ("2단계 접근", sel_ci[0]),
        ("혼합분포", mix_ci[0]),
    ]
    for name, ci_lo in checks:
        is_conservative = ci_lo <= true_worst_rate
        symbol = "✅ 보수적" if is_conservative else "❌ 비보수적"
        print(f"    {name:<20}: CI 하한={ci_lo*100:.4f} vs 실제={true_worst_rate*100:.4f} → {symbol}")


# =============================================================
# 시각화
# =============================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Set A에 대해 상세 비교 시각화
    ds = raw['datasets']['set_A']
    slopes_ols, _, _, _, times, data_arr = get_ols_slopes(ds)
    rows, cols_grid = 12, 13
    
    beta_full_A = bayesian_full_pool(ds, n_samples=5000, burnin=2000)
    beta_2s_A, thin_idx_A, _ = bayesian_two_stage(ds, n_samples=5000, burnin=2000)
    beta_mix_A, z_mix_A, _, _, _ = bayesian_mixture(ds, n_samples=5000, burnin=2000)
    
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    fig.suptitle('Localized Thinning Problem: Method Comparison (Set A, 3 measurements)', 
                  fontsize=14, fontweight='bold')
    
    # Row 0: 기울기 맵 비교
    full_mean = beta_full_A.mean(axis=0).reshape(rows, cols_grid)
    mix_mean = beta_mix_A.mean(axis=0).reshape(rows, cols_grid)
    
    # 2-stage mean (선별된 포인트만)
    twostage_mean = slopes_ols.copy()  # 비선별 포인트는 OLS 유지
    for idx_pos, k in enumerate(thin_idx_A):
        i, j = k // cols_grid, k % cols_grid
        twostage_mean[i, j] = beta_2s_A[:, k].mean()
    
    true_rate = -np.array(ds['annual_rate'])
    
    vmin = min(slopes_ols.min(), full_mean.min(), mix_mean.min(), true_rate.min()) * 100
    vmax = 0.5
    
    titles = ['True Rate', 'OLS', 'Bayes (Full Pool) ⚠', 'Bayes (Mixture) ✓']
    maps = [true_rate * 100, slopes_ols * 100, full_mean * 100, mix_mean * 100]
    
    for col, (title, m) in enumerate(zip(titles, maps)):
        im = axes[0, col].imshow(m, cmap='RdYlGn', aspect='auto', origin='lower',
                                   vmin=vmin, vmax=vmax)
        axes[0, col].set_title(title, fontsize=11)
        plt.colorbar(im, ax=axes[0, col], shrink=0.8, label='%/yr')
    
    # Row 1: 핵심 분석
    # 1) 최대 감육 포인트의 사후분포 비교
    worst_k = np.unravel_index(slopes_ols.argmin(), slopes_ols.shape)
    k_flat = worst_k[0] * cols_grid + worst_k[1]
    true_val = true_rate[worst_k] * 100
    
    ax = axes[1, 0]
    ax.hist(beta_full_A[:, k_flat] * 100, bins=40, density=True, alpha=0.5, 
            color='orange', label='Full Pool')
    ax.hist(beta_mix_A[:, k_flat] * 100, bins=40, density=True, alpha=0.5, 
            color='green', label='Mixture')
    if k_flat in thin_idx_A:
        ax.hist(beta_2s_A[:, k_flat] * 100, bins=40, density=True, alpha=0.5, 
                color='blue', label='2-Stage')
    ax.axvline(slopes_ols[worst_k] * 100, color='red', linestyle='--', linewidth=2, label='OLS')
    ax.axvline(true_val, color='black', linestyle='-', linewidth=2, label='True')
    ax.set_title(f'Worst Point Posterior at {worst_k}')
    ax.set_xlabel('Thinning Rate (%/yr)')
    ax.legend(fontsize=7)
    
    # 2) Shrinkage 비교 (기울기 순서)
    ax = axes[1, 1]
    sort_idx = np.argsort(slopes_ols.flatten())
    ax.plot(slopes_ols.flatten()[sort_idx] * 100, 'r-', alpha=0.7, label='OLS', linewidth=1.5)
    ax.plot(full_mean.flatten()[sort_idx] * 100, 'orange', alpha=0.7, 
            label='Full Pool', linewidth=1.5)
    ax.plot(mix_mean.flatten()[sort_idx] * 100, 'g-', alpha=0.7, 
            label='Mixture', linewidth=1.5)
    ax.plot(true_rate.flatten()[sort_idx] * 100, 'k--', alpha=0.5, 
            label='True', linewidth=1)
    ax.set_xlabel('Point (sorted by OLS slope)')
    ax.set_ylabel('Thinning Rate (%/yr)')
    ax.set_title('Sorted Slopes: Shrinkage Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3) 감육 확률 맵 (혼합분포)
    thinning_prob = z_mix_A.mean(axis=0).reshape(rows, cols_grid)
    im = axes[1, 2].imshow(thinning_prob, cmap='Reds', aspect='auto', origin='lower',
                             vmin=0, vmax=1)
    axes[1, 2].set_title('P(Thinning Component)\n(Mixture Model)')
    plt.colorbar(im, ax=axes[1, 2], shrink=0.8, label='Probability')
    
    # 4) CI 하한 비교 (보수성)
    ax = axes[1, 3]
    methods = ['OLS', 'Full\nPool⚠', '2-Stage✓', 'Mixture✓']
    
    ols_ci_lo = slopes_ols[worst_k] * 100 - 1.96 * 0.01 * 100 * np.sqrt(
        np.linalg.inv(np.column_stack([np.ones(len(times)), times]).T @ 
        np.column_stack([np.ones(len(times)), times]))[1, 1])
    
    ci_lowers = [
        ols_ci_lo,
        np.percentile(beta_full_A[:, k_flat], 2.5) * 100,
        np.percentile(beta_2s_A[:, k_flat], 2.5) * 100 if k_flat in thin_idx_A else np.nan,
        np.percentile(beta_mix_A[:, k_flat], 2.5) * 100,
    ]
    ci_uppers = [
        slopes_ols[worst_k] * 100 + 1.96 * 0.01 * 100 * np.sqrt(
            np.linalg.inv(np.column_stack([np.ones(len(times)), times]).T @ 
            np.column_stack([np.ones(len(times)), times]))[1, 1]),
        np.percentile(beta_full_A[:, k_flat], 97.5) * 100,
        np.percentile(beta_2s_A[:, k_flat], 97.5) * 100 if k_flat in thin_idx_A else np.nan,
        np.percentile(beta_mix_A[:, k_flat], 97.5) * 100,
    ]
    means = [
        slopes_ols[worst_k] * 100,
        beta_full_A[:, k_flat].mean() * 100,
        beta_2s_A[:, k_flat].mean() * 100 if k_flat in thin_idx_A else np.nan,
        beta_mix_A[:, k_flat].mean() * 100,
    ]
    
    colors = ['blue', 'orange', 'green', 'purple']
    for i, (m, lo, hi, c, label) in enumerate(zip(means, ci_lowers, ci_uppers, colors, methods)):
        if not np.isnan(m):
            ax.errorbar(i, m, yerr=[[m-lo], [hi-m]], fmt='o', capsize=5, 
                       color=c, markersize=8, linewidth=2)
    
    ax.axhline(true_val, color='black', linestyle='--', linewidth=2, label='True rate')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('Thinning Rate (%/yr)')
    ax.set_title('95% CI Comparison\n(Worst Point)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig('/home/claude/improved_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n시각화 저장: /home/claude/improved_comparison.png")

except ImportError:
    print("\nmatplotlib 미설치 - 시각화 생략")

print("\n" + "="*70)
print("분석 완료!")
print("="*70)
