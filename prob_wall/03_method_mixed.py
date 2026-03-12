"""
방법 2: 중간 - 혼합효과모델 (Random Slope Model)
==================================================
각 그리드 포인트의 기울기를 random effect로 모델링.
포인트 간 기울기의 분포를 추정하여 shrinkage 효과 획득.
데이터가 적은 포인트는 전체 평균 쪽으로 수축됨.

모델:
  y_{ij,t} = (α + a_ij) + (β + b_ij) * t + ε_{ij,t}
  
  α, β: 고정효과 (전체 평균 절편, 전체 평균 기울기)
  a_ij, b_ij: 랜덤효과 (포인트별 편차)
  [a_ij, b_ij] ~ N(0, G)
  ε ~ N(0, σ²)
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


def analyze_mixed_effects(dataset_dict, confidence=0.95):
    """혼합효과모델 기반 감육률 추정"""
    
    label = dataset_dict['label']
    times = np.array(dataset_dict['times'])
    data = np.array(dataset_dict['data'])  # (n_meas, rows, cols)
    n_meas, rows, cols = data.shape
    n_points = rows * cols
    
    print(f"\n{'='*60}")
    print(f"방법 2 (중간): 혼합효과모델 - {label}")
    print(f"{'='*60}")
    
    # ==========================================================
    # 2. 데이터를 long format으로 변환
    # ==========================================================
    records = []
    for i in range(rows):
        for j in range(cols):
            point_id = i * cols + j
            for t_idx in range(n_meas):
                records.append({
                    'point_id': point_id,
                    'row': i,
                    'col': j,
                    'time': times[t_idx],
                    'thickness': data[t_idx, i, j]
                })
    
    n_obs = len(records)
    point_ids = np.array([r['point_id'] for r in records])
    t_vec = np.array([r['time'] for r in records])
    y_vec = np.array([r['thickness'] for r in records])
    
    print(f"총 관측치: {n_obs}")
    
    # ==========================================================
    # 3. EM 알고리즘으로 혼합효과모델 적합
    # ==========================================================
    # 고정효과: β_fixed = [α, β] (전체 평균 절편, 기울기)
    # 랜덤효과: u_k = [a_k, b_k] for each point k
    # G: 2x2 랜덤효과 공분산행렬
    # σ²: 잔차 분산
    
    # 고정효과 디자인 행렬
    X_full = np.column_stack([np.ones(n_obs), t_vec])
    
    # 랜덤효과 디자인 행렬 (포인트별)
    # Z_k = X_k (같은 구조: [1, t])
    
    # 초기값
    beta_fixed = np.linalg.lstsq(X_full, y_vec, rcond=None)[0]
    sigma2 = 0.001  # 잔차 분산
    G = np.array([[0.001, 0], [0, 0.0001]])  # 랜덤효과 공분산
    
    # 포인트별 데이터 구성
    point_data = {}
    for k in range(n_points):
        mask = point_ids == k
        point_data[k] = {
            'X': X_full[mask],
            'y': y_vec[mask],
            'n': mask.sum()
        }
    
    # EM 반복
    MAX_ITER = 200
    TOL = 1e-8
    
    for iteration in range(MAX_ITER):
        G_inv = np.linalg.inv(G)
        
        # E-step: 각 포인트의 랜덤효과 사후분포 계산
        u_hat = np.zeros((n_points, 2))  # [a_k, b_k]
        D_hat = np.zeros((n_points, 2, 2))  # 사후 공분산
        
        for k in range(n_points):
            X_k = point_data[k]['X']  # (n_k, 2)
            y_k = point_data[k]['y']  # (n_k,)
            Z_k = X_k  # 같은 디자인
            
            # 사후 공분산: D_k = (Z_k'Z_k/σ² + G^{-1})^{-1}
            D_k = np.linalg.inv(Z_k.T @ Z_k / sigma2 + G_inv)
            D_hat[k] = D_k
            
            # 사후 평균: u_k = D_k * Z_k'(y_k - X_k β) / σ²
            resid_k = y_k - X_k @ beta_fixed
            u_hat[k] = D_k @ (Z_k.T @ resid_k / sigma2)
        
        # M-step
        # 고정효과 업데이트
        y_adj = y_vec.copy()
        for k in range(n_points):
            mask = point_ids == k
            Z_k = point_data[k]['X']
            y_adj[mask] -= Z_k @ u_hat[k]
        
        beta_fixed_new = np.linalg.lstsq(X_full, y_adj, rcond=None)[0]
        
        # σ² 업데이트
        sse = 0
        trace_term = 0
        for k in range(n_points):
            X_k = point_data[k]['X']
            y_k = point_data[k]['y']
            Z_k = X_k
            resid_k = y_k - X_k @ beta_fixed_new - Z_k @ u_hat[k]
            sse += np.sum(resid_k**2)
            # trace(Z_k D_k Z_k') 보정 항
            trace_term += np.trace(Z_k @ D_hat[k] @ Z_k.T)
        
        sigma2_new = (sse + trace_term) / n_obs
        
        # G 업데이트
        G_new = np.zeros((2, 2))
        for k in range(n_points):
            G_new += np.outer(u_hat[k], u_hat[k]) + D_hat[k]
        G_new /= n_points
        
        # 수렴 체크
        diff = (np.abs(beta_fixed_new - beta_fixed).max() + 
                abs(sigma2_new - sigma2) + 
                np.abs(G_new - G).max())
        
        beta_fixed = beta_fixed_new
        sigma2 = sigma2_new
        G = G_new
        
        if diff < TOL:
            print(f"  EM 수렴: {iteration+1} 반복")
            break
    else:
        print(f"  EM 최대 반복({MAX_ITER}) 도달")
    
    # ==========================================================
    # 4. 결과 정리
    # ==========================================================
    sigma = np.sqrt(sigma2)
    
    # 포인트별 기울기 = β_fixed[1] + b_k
    slopes_mixed = np.zeros((rows, cols))
    slopes_ci_lower = np.zeros((rows, cols))
    slopes_ci_upper = np.zeros((rows, cols))
    
    alpha_level = 1 - confidence
    z_crit = 1.96  # 대표본 근사 (or t 사용 가능)
    
    for k in range(n_points):
        i, j = k // cols, k % cols
        slope_k = beta_fixed[1] + u_hat[k, 1]
        slope_se_k = np.sqrt(D_hat[k, 1, 1])  # 사후 표준오차
        
        slopes_mixed[i, j] = slope_k
        slopes_ci_lower[i, j] = slope_k - z_crit * slope_se_k
        slopes_ci_upper[i, j] = slope_k + z_crit * slope_se_k
    
    # OLS 기울기 비교용
    slopes_ols = np.zeros((rows, cols))
    XtX_inv = np.linalg.inv(X_full[:n_meas].T @ X_full[:n_meas])
    for i in range(rows):
        for j in range(cols):
            y_ij = data[:, i, j]
            X_ij = np.column_stack([np.ones(n_meas), times])
            beta_ij = XtX_inv @ (X_ij.T @ y_ij)
            slopes_ols[i, j] = beta_ij[1]
    
    print(f"\n--- 모델 파라미터 ---")
    print(f"  고정효과 절편 (α): {beta_fixed[0]:.5f}")
    print(f"  고정효과 기울기 (β): {beta_fixed[1]*100:.4f} %/yr")
    print(f"  잔차 σ: {sigma:.5f} ({sigma*100:.3f}%)")
    print(f"  랜덤효과 G:")
    print(f"    Var(a): {G[0,0]:.6f}, Var(b): {G[1,1]:.6f}")
    print(f"    Corr(a,b): {G[0,1]/np.sqrt(G[0,0]*G[1,1]):.3f}")
    
    print(f"\n--- 기울기 비교 ---")
    print(f"  OLS 최소 기울기: {slopes_ols.min()*100:.4f} %/yr")
    print(f"  Mixed 최소 기울기: {slopes_mixed.min()*100:.4f} %/yr")
    print(f"  Shrinkage 효과: {(slopes_ols.min() - slopes_mixed.min())*100:.4f} %/yr")
    
    # 최대 감육 포인트
    min_idx = np.unravel_index(slopes_mixed.argmin(), slopes_mixed.shape)
    print(f"\n--- 최대 감육 포인트: ({min_idx[0]}, {min_idx[1]}) ---")
    print(f"  Mixed 기울기: {slopes_mixed[min_idx]*100:.4f} %/yr")
    print(f"  95% CI: [{slopes_ci_lower[min_idx]*100:.4f}, {slopes_ci_upper[min_idx]*100:.4f}] %/yr")
    print(f"  OLS 기울기:   {slopes_ols[min_idx]*100:.4f} %/yr")
    
    return {
        'slopes_mixed': slopes_mixed,
        'slopes_ols': slopes_ols,
        'ci_lower': slopes_ci_lower,
        'ci_upper': slopes_ci_upper,
        'beta_fixed': beta_fixed,
        'sigma': sigma,
        'G': G,
        'u_hat': u_hat,
        'D_hat': D_hat,
        'times': times,
        'data': data,
    }


# =============================================================
# 실행
# =============================================================
results_A = analyze_mixed_effects(raw['datasets']['set_A'])
results_B = analyze_mixed_effects(raw['datasets']['set_B'])

# =============================================================
# 시각화
# =============================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Method 2: Mixed Effects Model (Random Slopes)', fontsize=14, fontweight='bold')
    
    for row, (name, res) in enumerate([('Set A (3 meas)', results_A), 
                                         ('Set B (6 meas)', results_B)]):
        sm = res['slopes_mixed']
        so = res['slopes_ols']
        ci_w = res['ci_upper'] - res['ci_lower']
        
        # OLS 기울기
        vmin, vmax = min(so.min(), sm.min()) * 100, max(so.max(), sm.max()) * 100
        im0 = axes[row, 0].imshow(so * 100, cmap='RdYlGn', aspect='auto', origin='lower',
                                    vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f'{name}\nOLS Slopes (%/yr)')
        plt.colorbar(im0, ax=axes[row, 0], shrink=0.8)
        
        # Mixed 기울기
        im1 = axes[row, 1].imshow(sm * 100, cmap='RdYlGn', aspect='auto', origin='lower',
                                    vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f'{name}\nMixed Slopes (%/yr)')
        plt.colorbar(im1, ax=axes[row, 1], shrink=0.8)
        
        # Shrinkage (OLS - Mixed)
        shrink = (so - sm) * 100
        im2 = axes[row, 2].imshow(shrink, cmap='coolwarm', aspect='auto', origin='lower',
                                    vmin=-shrink.max(), vmax=shrink.max())
        axes[row, 2].set_title(f'{name}\nShrinkage: OLS - Mixed (%/yr)')
        plt.colorbar(im2, ax=axes[row, 2], shrink=0.8)
        
        # CI 폭 맵
        im3 = axes[row, 3].imshow(ci_w * 100, cmap='Oranges', aspect='auto', origin='lower')
        axes[row, 3].set_title(f'{name}\n95% CI Width (%/yr)')
        plt.colorbar(im3, ax=axes[row, 3], shrink=0.8)
    
    plt.tight_layout()
    fig.savefig('/home/claude/method2_mixed.png', dpi=150, bbox_inches='tight')
    print(f"\n시각화 저장: /home/claude/method2_mixed.png")
    
    # 추가: shrinkage 히스토그램 비교
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle('OLS vs Mixed Effects: Slope Distribution', fontsize=13, fontweight='bold')
    
    for idx, (name, res) in enumerate([('Set A (3 meas)', results_A), 
                                         ('Set B (6 meas)', results_B)]):
        ax = axes2[idx]
        so_flat = res['slopes_ols'].flatten() * 100
        sm_flat = res['slopes_mixed'].flatten() * 100
        
        ax.hist(so_flat, bins=30, alpha=0.5, label='OLS', color='blue', density=True)
        ax.hist(sm_flat, bins=30, alpha=0.5, label='Mixed', color='red', density=True)
        ax.axvline(so_flat.min(), color='blue', linestyle='--', label=f'OLS min={so_flat.min():.3f}')
        ax.axvline(sm_flat.min(), color='red', linestyle='--', label=f'Mixed min={sm_flat.min():.3f}')
        ax.set_xlabel('Thinning Rate (%/yr)')
        ax.set_ylabel('Density')
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig('/home/claude/method2_comparison.png', dpi=150, bbox_inches='tight')
    print(f"비교 시각화 저장: /home/claude/method2_comparison.png")

except ImportError:
    print("\nmatplotlib 미설치 - 시각화 생략")
