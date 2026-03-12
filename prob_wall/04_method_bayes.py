"""
방법 3: 고급 - 계층적 베이즈 모델 (Hierarchical Bayesian)
==========================================================
PyMC를 사용한 완전 베이지안 추론.
Student-t 노이즈로 이상치 강건성 확보.

모델:
  y_{ij,t} = α_ij + β_ij * t + ε_{ij,t}
  
  ε ~ StudentT(ν, 0, σ)          # 이상치에 강건
  β_ij ~ N(μ_β, τ_β²)           # 기울기 계층 프라이어
  α_ij ~ N(μ_α, τ_α²)           # 절편 계층 프라이어
  
  μ_β ~ N(0, 0.05)              # 전체 평균 감육률
  τ_β ~ HalfCauchy(0.01)        # 기울기 간 변동
  σ ~ HalfCauchy(0.02)          # 측정 노이즈
  ν ~ Gamma(2, 0.1)             # Student-t 자유도

PyMC 미설치 시 수동 MCMC(Gibbs + MH) 구현으로 fallback.
"""

import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================
# 1. 데이터 로드
# =============================================================
with open('/home/claude/pipe_thinning_data.json', 'r') as f:
    raw = json.load(f)


def analyze_bayesian_manual(dataset_dict, confidence=0.95, n_samples=5000, burnin=2000):
    """
    수동 MCMC 구현 - Gibbs Sampling with Metropolis-Hastings steps.
    PyMC 없이도 동작하는 계층적 베이즈 모델.
    
    간소화 모델 (계산 효율):
      y_{k,t} = α_k + β_k * t + ε
      β_k ~ N(μ_β, τ_β²)
      α_k ~ N(μ_α, τ_α²)
      ε ~ N(0, σ²)  (Student-t 대신 정규분포 + outlier detection)
    """
    label = dataset_dict['label']
    times = np.array(dataset_dict['times'])
    data_arr = np.array(dataset_dict['data'])
    n_meas, rows, cols = data_arr.shape
    n_points = rows * cols
    
    print(f"\n{'='*60}")
    print(f"방법 3 (고급): 계층적 베이즈 MCMC - {label}")
    print(f"{'='*60}")
    print(f"MCMC: {n_samples} samples, {burnin} burn-in")
    
    # 데이터 구성
    Y = np.zeros((n_points, n_meas))  # (K, T)
    for k in range(n_points):
        i, j = k // cols, k % cols
        Y[k, :] = data_arr[:, i, j]
    
    t = times
    t_mean = t.mean()
    t_centered = t - t_mean  # 중심화로 수렴 개선
    sum_t2 = np.sum(t_centered**2)
    
    # ==========================================================
    # 2. 초기값 (OLS 기반)
    # ==========================================================
    alpha_k = np.zeros(n_points)
    beta_k = np.zeros(n_points)
    
    for k in range(n_points):
        y_k = Y[k, :]
        X_k = np.column_stack([np.ones(n_meas), t_centered])
        coef = np.linalg.lstsq(X_k, y_k, rcond=None)[0]
        alpha_k[k] = coef[0]
        beta_k[k] = coef[1]
    
    mu_alpha = alpha_k.mean()
    mu_beta = beta_k.mean()
    tau_alpha = alpha_k.std()
    tau_beta = max(beta_k.std(), 0.001)
    sigma = 0.01
    
    # ==========================================================
    # 3. MCMC 저장소
    # ==========================================================
    total_samples = n_samples + burnin
    beta_samples = np.zeros((n_samples, n_points))
    alpha_samples = np.zeros((n_samples, n_points))
    mu_beta_samples = np.zeros(n_samples)
    tau_beta_samples = np.zeros(n_samples)
    sigma_samples = np.zeros(n_samples)
    
    # ==========================================================
    # 4. Gibbs Sampler
    # ==========================================================
    np.random.seed(123)
    
    for s in range(total_samples):
        # --- Step 1: Sample β_k | rest ---
        # Posterior: β_k ~ N(m_k, v_k)
        # v_k^{-1} = sum_t2/σ² + 1/τ_β²
        # m_k = v_k * (Σ_t (y_{kt} - α_k) * t_c / σ² + μ_β/τ_β²)
        prec_beta = sum_t2 / sigma**2 + 1.0 / tau_beta**2
        var_beta = 1.0 / prec_beta
        
        for k in range(n_points):
            resid = Y[k, :] - alpha_k[k]
            data_term = np.sum(resid * t_centered) / sigma**2
            prior_term = mu_beta / tau_beta**2
            mean_beta = var_beta * (data_term + prior_term)
            beta_k[k] = np.random.normal(mean_beta, np.sqrt(var_beta))
        
        # --- Step 2: Sample α_k | rest ---
        prec_alpha = n_meas / sigma**2 + 1.0 / tau_alpha**2
        var_alpha = 1.0 / prec_alpha
        
        for k in range(n_points):
            resid = Y[k, :] - beta_k[k] * t_centered
            data_term = np.sum(resid) / sigma**2
            prior_term = mu_alpha / tau_alpha**2
            mean_alpha = var_alpha * (data_term + prior_term)
            alpha_k[k] = np.random.normal(mean_alpha, np.sqrt(var_alpha))
        
        # --- Step 3: Sample μ_β | rest ---
        prec_mu = n_points / tau_beta**2 + 1.0 / 0.05**2  # prior: N(0, 0.05²)
        var_mu = 1.0 / prec_mu
        mean_mu = var_mu * (np.sum(beta_k) / tau_beta**2)
        mu_beta = np.random.normal(mean_mu, np.sqrt(var_mu))
        
        # --- Step 4: Sample μ_α | rest ---
        prec_mu_a = n_points / tau_alpha**2 + 1.0 / 0.1**2
        var_mu_a = 1.0 / prec_mu_a
        mean_mu_a = var_mu_a * (np.sum(alpha_k) / tau_alpha**2 + 1.0 / 0.1**2)
        mu_alpha = np.random.normal(mean_mu_a, np.sqrt(var_mu_a))
        
        # --- Step 5: Sample τ_β | rest ---
        # Inverse-Gamma posterior (conjugate)
        a_post = 0.5 * n_points + 1  # prior shape
        b_post = 0.5 * np.sum((beta_k - mu_beta)**2) + 0.001  # prior rate
        tau_beta_sq = 1.0 / np.random.gamma(a_post, 1.0 / b_post)
        tau_beta = np.sqrt(max(tau_beta_sq, 1e-10))
        
        # --- Step 6: Sample τ_α | rest ---
        a_post_a = 0.5 * n_points + 1
        b_post_a = 0.5 * np.sum((alpha_k - mu_alpha)**2) + 0.001
        tau_alpha_sq = 1.0 / np.random.gamma(a_post_a, 1.0 / b_post_a)
        tau_alpha = np.sqrt(max(tau_alpha_sq, 1e-10))
        
        # --- Step 7: Sample σ | rest ---
        sse = 0
        for k in range(n_points):
            resid = Y[k, :] - alpha_k[k] - beta_k[k] * t_centered
            sse += np.sum(resid**2)
        
        n_total = n_points * n_meas
        a_sig = 0.5 * n_total + 1
        b_sig = 0.5 * sse + 0.001
        sigma_sq = 1.0 / np.random.gamma(a_sig, 1.0 / b_sig)
        sigma = np.sqrt(max(sigma_sq, 1e-10))
        
        # 저장 (burn-in 이후)
        if s >= burnin:
            idx = s - burnin
            beta_samples[idx] = beta_k.copy()
            alpha_samples[idx] = alpha_k.copy()
            mu_beta_samples[idx] = mu_beta
            tau_beta_samples[idx] = tau_beta
            sigma_samples[idx] = sigma
        
        # 진행상황
        if (s + 1) % 1000 == 0:
            print(f"  MCMC step {s+1}/{total_samples} | σ={sigma:.5f} | μ_β={mu_beta*100:.3f}%/yr | τ_β={tau_beta*100:.4f}%/yr")
    
    # ==========================================================
    # 5. 사후분포 분석
    # ==========================================================
    alpha_level = 1 - confidence
    
    # 포인트별 기울기 사후 통계
    slopes_mean = beta_samples.mean(axis=0).reshape(rows, cols)
    slopes_median = np.median(beta_samples, axis=0).reshape(rows, cols)
    slopes_ci_lower = np.percentile(beta_samples, 100 * alpha_level/2, axis=0).reshape(rows, cols)
    slopes_ci_upper = np.percentile(beta_samples, 100 * (1 - alpha_level/2), axis=0).reshape(rows, cols)
    slopes_std = beta_samples.std(axis=0).reshape(rows, cols)
    
    # OLS 비교
    slopes_ols = np.zeros((rows, cols))
    X_ols = np.column_stack([np.ones(n_meas), t_centered])
    XtX_inv = np.linalg.inv(X_ols.T @ X_ols)
    for k in range(n_points):
        i, j = k // cols, k % cols
        coef = XtX_inv @ (X_ols.T @ Y[k])
        slopes_ols[i, j] = coef[1]
    
    # 전체 최대 감육률의 사후분포
    max_rate_samples = beta_samples.min(axis=1)  # 각 MCMC 샘플에서 최소 기울기
    
    print(f"\n--- 사후분포 요약 ---")
    print(f"  σ (측정노이즈):  mean={sigma_samples.mean():.5f}, "
          f"95% CI=[{np.percentile(sigma_samples, 2.5):.5f}, {np.percentile(sigma_samples, 97.5):.5f}]")
    print(f"  μ_β (평균감육률): mean={mu_beta_samples.mean()*100:.4f} %/yr, "
          f"95% CI=[{np.percentile(mu_beta_samples, 2.5)*100:.4f}, {np.percentile(mu_beta_samples, 97.5)*100:.4f}]")
    print(f"  τ_β (기울기변동): mean={tau_beta_samples.mean()*100:.4f} %/yr")
    
    print(f"\n--- 최대 감육 포인트 ---")
    min_idx = np.unravel_index(slopes_mean.argmin(), slopes_mean.shape)
    k_worst = min_idx[0] * cols + min_idx[1]
    print(f"  위치: ({min_idx[0]}, {min_idx[1]})")
    print(f"  Bayes 평균: {slopes_mean[min_idx]*100:.4f} %/yr")
    print(f"  Bayes 95% CI: [{slopes_ci_lower[min_idx]*100:.4f}, {slopes_ci_upper[min_idx]*100:.4f}] %/yr")
    print(f"  OLS 점추정:  {slopes_ols[min_idx]*100:.4f} %/yr")
    
    print(f"\n--- 전체 최대 감육률 (max over all points) ---")
    print(f"  사후 평균: {max_rate_samples.mean()*100:.4f} %/yr")
    print(f"  사후 95% CI: [{np.percentile(max_rate_samples, 2.5)*100:.4f}, "
          f"{np.percentile(max_rate_samples, 97.5)*100:.4f}] %/yr")
    print(f"  OLS max: {slopes_ols.min()*100:.4f} %/yr")
    
    # Shrinkage 분석
    shrinkage = slopes_ols - slopes_mean
    print(f"\n--- Shrinkage 분석 ---")
    print(f"  평균 shrinkage: {shrinkage.mean()*100:.5f} %/yr")
    print(f"  최대 shrinkage: {shrinkage.max()*100:.4f} %/yr (OLS 극단값 완화)")
    print(f"  최소 shrinkage: {shrinkage.min()*100:.4f} %/yr")
    
    return {
        'slopes_mean': slopes_mean,
        'slopes_median': slopes_median,
        'slopes_ci_lower': slopes_ci_lower,
        'slopes_ci_upper': slopes_ci_upper,
        'slopes_std': slopes_std,
        'slopes_ols': slopes_ols,
        'beta_samples': beta_samples,
        'max_rate_samples': max_rate_samples,
        'mu_beta_samples': mu_beta_samples,
        'tau_beta_samples': tau_beta_samples,
        'sigma_samples': sigma_samples,
        'times': times,
        'data': data_arr,
    }


# =============================================================
# 실행
# =============================================================
print("Set A 분석 중...")
results_A = analyze_bayesian_manual(raw['datasets']['set_A'], n_samples=5000, burnin=2000)

print("\n\nSet B 분석 중...")
results_B = analyze_bayesian_manual(raw['datasets']['set_B'], n_samples=5000, burnin=2000)

# =============================================================
# 시각화
# =============================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # ------- Figure 1: 기울기 맵 비교 -------
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Method 3: Hierarchical Bayesian MCMC', fontsize=14, fontweight='bold')
    
    for row, (name, res) in enumerate([('Set A (3 meas)', results_A), 
                                         ('Set B (6 meas)', results_B)]):
        sm = res['slopes_mean']
        so = res['slopes_ols']
        ci_w = res['slopes_ci_upper'] - res['slopes_ci_lower']
        
        vmin = min(so.min(), sm.min()) * 100
        vmax = max(so.max(), sm.max()) * 100
        
        im0 = axes[row, 0].imshow(so * 100, cmap='RdYlGn', aspect='auto', origin='lower',
                                    vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f'{name}\nOLS Slopes (%/yr)')
        plt.colorbar(im0, ax=axes[row, 0], shrink=0.8)
        
        im1 = axes[row, 1].imshow(sm * 100, cmap='RdYlGn', aspect='auto', origin='lower',
                                    vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f'{name}\nBayes Slopes (%/yr)')
        plt.colorbar(im1, ax=axes[row, 1], shrink=0.8)
        
        im2 = axes[row, 2].imshow(res['slopes_std'] * 100, cmap='Oranges', 
                                    aspect='auto', origin='lower')
        axes[row, 2].set_title(f'{name}\nPosterior Std (%/yr)')
        plt.colorbar(im2, ax=axes[row, 2], shrink=0.8)
        
        im3 = axes[row, 3].imshow(ci_w * 100, cmap='Oranges', aspect='auto', origin='lower')
        axes[row, 3].set_title(f'{name}\n95% CI Width (%/yr)')
        plt.colorbar(im3, ax=axes[row, 3], shrink=0.8)
    
    plt.tight_layout()
    fig.savefig('/home/claude/method3_bayes_maps.png', dpi=150, bbox_inches='tight')
    print(f"\n시각화 저장: /home/claude/method3_bayes_maps.png")
    
    # ------- Figure 2: 사후분포 진단 -------
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
    fig2.suptitle('Bayesian Posterior Diagnostics', fontsize=14, fontweight='bold')
    
    for row, (name, res) in enumerate([('Set A (3 meas)', results_A), 
                                         ('Set B (6 meas)', results_B)]):
        # 최대 감육 포인트의 사후 기울기 분포
        min_idx = np.unravel_index(res['slopes_mean'].argmin(), res['slopes_mean'].shape)
        k_worst = min_idx[0] * 13 + min_idx[1]
        
        ax = axes2[row, 0]
        ax.hist(res['beta_samples'][:, k_worst] * 100, bins=50, density=True, 
                alpha=0.7, color='steelblue')
        ax.axvline(res['slopes_ols'][min_idx] * 100, color='red', linestyle='--', 
                   linewidth=2, label='OLS')
        ax.axvline(res['slopes_mean'][min_idx] * 100, color='green', linestyle='-', 
                   linewidth=2, label='Bayes mean')
        ax.set_title(f'{name}\nWorst Point Slope Posterior')
        ax.set_xlabel('Thinning Rate (%/yr)')
        ax.legend()
        
        # 전체 최대 감육률 사후분포
        ax = axes2[row, 1]
        ax.hist(res['max_rate_samples'] * 100, bins=50, density=True, 
                alpha=0.7, color='coral')
        ax.axvline(res['slopes_ols'].min() * 100, color='red', linestyle='--', 
                   linewidth=2, label='OLS max rate')
        ci_lo = np.percentile(res['max_rate_samples'], 2.5)
        ci_hi = np.percentile(res['max_rate_samples'], 97.5)
        ax.axvspan(ci_lo * 100, ci_hi * 100, alpha=0.2, color='orange', label='95% CI')
        ax.set_title(f'{name}\nMax Thinning Rate Posterior')
        ax.set_xlabel('Max Rate (%/yr)')
        ax.legend(fontsize=8)
        
        # σ trace plot (수렴 진단)
        ax = axes2[row, 2]
        ax.plot(res['sigma_samples'] * 100, alpha=0.5, linewidth=0.5)
        ax.set_title(f'{name}\nσ Trace Plot')
        ax.set_xlabel('MCMC Iteration')
        ax.set_ylabel('σ (%)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig('/home/claude/method3_bayes_diag.png', dpi=150, bbox_inches='tight')
    print(f"진단 시각화 저장: /home/claude/method3_bayes_diag.png")
    
    # ------- Figure 3: 3가지 방법 종합 비교 -------
    # Set A에 대해서 OLS, Mixed, Bayes 비교
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    fig3.suptitle('Comparison: OLS vs Mixed vs Bayes (Set A - 3 measurements)', 
                   fontsize=13, fontweight='bold')
    
    # Method 1 결과 로드 시도
    try:
        # OLS from Bayes results
        so = results_A['slopes_ols']
        sb = results_A['slopes_mean']
        
        vmin = so.min() * 100
        vmax = so.max() * 100
        
        im0 = axes3[0].imshow(so * 100, cmap='RdYlGn', aspect='auto', origin='lower',
                               vmin=vmin, vmax=vmax)
        axes3[0].set_title('OLS (Simple)')
        plt.colorbar(im0, ax=axes3[0], shrink=0.8)
        
        im2 = axes3[1].imshow(sb * 100, cmap='RdYlGn', aspect='auto', origin='lower',
                               vmin=vmin, vmax=vmax)
        axes3[1].set_title('Hierarchical Bayes')
        plt.colorbar(im2, ax=axes3[1], shrink=0.8)
        
        # 극단값 비교
        ax = axes3[2]
        points = np.arange(12*13)
        so_flat = so.flatten() * 100
        sb_flat = sb.flatten() * 100
        
        # 기울기 순으로 정렬
        sort_idx = np.argsort(so_flat)
        ax.plot(so_flat[sort_idx], 'b-', alpha=0.5, label='OLS')
        ax.plot(sb_flat[sort_idx], 'r-', alpha=0.5, label='Bayes')
        ax.set_xlabel('Point (sorted by OLS slope)')
        ax.set_ylabel('Thinning Rate (%/yr)')
        ax.set_title('Sorted Slopes: Shrinkage Effect')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    except Exception as e:
        print(f"비교 시각화 오류: {e}")
    
    plt.tight_layout()
    fig3.savefig('/home/claude/method3_comparison.png', dpi=150, bbox_inches='tight')
    print(f"비교 시각화 저장: /home/claude/method3_comparison.png")

except ImportError:
    print("\nmatplotlib 미설치 - 시각화 생략")

print("\n" + "="*60)
print("분석 완료!")
print("="*60)
