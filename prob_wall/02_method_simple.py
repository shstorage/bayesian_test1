"""
방법 1: 간단 - 포인트별 OLS + 풀링된 σ로 기울기 신뢰구간
============================================================
각 그리드 포인트에서 독립적으로 선형회귀 수행.
모든 포인트의 잔차를 풀링하여 공통 σ 추정.
기울기의 t-분포 기반 신뢰구간 산출.
"""

import numpy as np
import json
from scipy import stats

# =============================================================
# 1. 데이터 로드
# =============================================================
with open('/home/claude/pipe_thinning_data.json', 'r') as f:
    raw = json.load(f)

def analyze_simple(dataset_dict, confidence=0.95):
    """포인트별 OLS + 풀링된 σ 기반 기울기 신뢰구간"""
    
    label = dataset_dict['label']
    times = np.array(dataset_dict['times'])
    data = np.array(dataset_dict['data'])  # (n_meas, rows, cols)
    n_meas, rows, cols = data.shape
    n_points = rows * cols
    
    print(f"\n{'='*60}")
    print(f"방법 1 (간단): 포인트별 OLS - {label}")
    print(f"{'='*60}")
    print(f"측정 횟수: {n_meas}, 그리드: {rows}x{cols}, 신뢰수준: {confidence*100}%")
    
    # ==========================================================
    # 2. 포인트별 OLS 회귀
    # ==========================================================
    slopes = np.zeros((rows, cols))        # β (기울기 = 감육률)
    intercepts = np.zeros((rows, cols))    # α (초기 두께)
    residuals_all = []                     # 풀링용 잔차
    slope_se_individual = np.zeros((rows, cols))  # 개별 SE
    
    # OLS 행렬 준비 (모든 포인트가 같은 시점이므로 한번만)
    X = np.column_stack([np.ones(n_meas), times])  # [1, t]
    XtX_inv = np.linalg.inv(X.T @ X)
    
    for i in range(rows):
        for j in range(cols):
            y = data[:, i, j]
            
            # OLS: β = (X'X)^{-1} X'y
            beta = XtX_inv @ (X.T @ y)
            intercepts[i, j] = beta[0]
            slopes[i, j] = beta[1]
            
            # 잔차
            y_hat = X @ beta
            resid = y - y_hat
            residuals_all.append(resid)
            
            # 개별 포인트의 기울기 SE
            if n_meas > 2:
                s2_ind = np.sum(resid**2) / (n_meas - 2)
                slope_se_individual[i, j] = np.sqrt(s2_ind * XtX_inv[1, 1])
    
    residuals_all = np.concatenate(residuals_all)
    
    # ==========================================================
    # 3. 풀링된 σ 추정
    # ==========================================================
    dof_total = n_points * (n_meas - 2)  # 총 자유도
    sigma_pooled = np.sqrt(np.sum(residuals_all**2) / dof_total)
    
    # 풀링된 σ 기반 기울기 SE
    slope_se_pooled = sigma_pooled * np.sqrt(XtX_inv[1, 1])
    
    print(f"\n--- 풀링된 측정 노이즈 ---")
    print(f"  σ_pooled = {sigma_pooled:.5f} (normalized thickness 단위)")
    print(f"  σ_pooled = {sigma_pooled*100:.3f}% (두께 대비)")
    print(f"  자유도: {dof_total}")
    
    # ==========================================================
    # 4. 신뢰구간 산출
    # ==========================================================
    alpha = 1 - confidence
    
    # 방법 A: 풀링된 σ 사용 (보수적, 작은 n에서 유리)
    t_crit_pooled = stats.t.ppf(1 - alpha/2, dof_total)
    ci_lower_pooled = slopes - t_crit_pooled * slope_se_pooled
    ci_upper_pooled = slopes + t_crit_pooled * slope_se_pooled
    
    # 방법 B: 개별 σ 사용 (참고용)
    dof_ind = n_meas - 2
    t_crit_ind = stats.t.ppf(1 - alpha/2, dof_ind)
    ci_lower_ind = slopes - t_crit_ind * slope_se_individual
    ci_upper_ind = slopes + t_crit_ind * slope_se_individual
    
    # ==========================================================
    # 5. 결과 요약
    # ==========================================================
    print(f"\n--- 기울기 (감육률) 분포 ---")
    print(f"  평균: {slopes.mean()*100:.4f} %/yr")
    print(f"  최소 (최대 감육): {slopes.min()*100:.4f} %/yr")
    print(f"  최대 (최소 감육): {slopes.max()*100:.4f} %/yr")
    
    # 최대 감육 포인트
    min_idx = np.unravel_index(slopes.argmin(), slopes.shape)
    print(f"\n--- 최대 감육 포인트: ({min_idx[0]}, {min_idx[1]}) ---")
    print(f"  기울기: {slopes[min_idx]*100:.4f} %/yr")
    print(f"  풀링σ 95% CI: [{ci_lower_pooled[min_idx]*100:.4f}, {ci_upper_pooled[min_idx]*100:.4f}] %/yr")
    print(f"  개별σ 95% CI: [{ci_lower_ind[min_idx]*100:.4f}, {ci_upper_ind[min_idx]*100:.4f}] %/yr")
    print(f"  측정값: {[f'{data[t, min_idx[0], min_idx[1]]:.4f}' for t in range(n_meas)]}")
    
    # 최대 감육률의 보수적 추정 (CI 하한)
    worst_rate_pooled = ci_lower_pooled.min()
    worst_rate_ind = ci_lower_ind.min()
    print(f"\n--- 보수적 최대 감육률 (CI 하한 중 최소) ---")
    print(f"  풀링σ: {worst_rate_pooled*100:.4f} %/yr")
    print(f"  개별σ: {worst_rate_ind*100:.4f} %/yr")
    
    return {
        'slopes': slopes,
        'intercepts': intercepts,
        'sigma_pooled': sigma_pooled,
        'ci_lower_pooled': ci_lower_pooled,
        'ci_upper_pooled': ci_upper_pooled,
        'ci_lower_ind': ci_lower_ind,
        'ci_upper_ind': ci_upper_ind,
        'slope_se_pooled': slope_se_pooled,
        'slope_se_individual': slope_se_individual,
        'times': times,
        'data': data,
    }


# =============================================================
# 실행
# =============================================================
results_A = analyze_simple(raw['datasets']['set_A'])
results_B = analyze_simple(raw['datasets']['set_B'])

# =============================================================
# 시각화
# =============================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Method 1: Point-wise OLS with Pooled σ', fontsize=14, fontweight='bold')
    
    for row, (name, res) in enumerate([('Set A (3 meas)', results_A), 
                                         ('Set B (6 meas)', results_B)]):
        slopes = res['slopes']
        ci_w = res['ci_upper_pooled'] - res['ci_lower_pooled']
        
        # 기울기 맵
        im0 = axes[row, 0].imshow(slopes * 100, cmap='RdYlGn', aspect='auto', origin='lower')
        axes[row, 0].set_title(f'{name}\nSlope (Thinning Rate %/yr)')
        plt.colorbar(im0, ax=axes[row, 0], shrink=0.8)
        
        # CI 폭 맵
        im1 = axes[row, 1].imshow(ci_w * 100, cmap='Oranges', aspect='auto', origin='lower')
        axes[row, 1].set_title(f'{name}\n95% CI Width (%/yr)')
        plt.colorbar(im1, ax=axes[row, 1], shrink=0.8)
        
        # 최대 감육 포인트 시계열 + CI
        min_idx = np.unravel_index(slopes.argmin(), slopes.shape)
        i, j = min_idx
        times = res['times']
        y_obs = res['data'][:, i, j]
        y_fit = res['intercepts'][i, j] + slopes[i, j] * times
        
        ax = axes[row, 2]
        ax.plot(times, y_obs, 'ko', markersize=8, label='Measured')
        ax.plot(times, y_fit, 'r-', linewidth=2, label='OLS fit')
        
        # CI band for slope
        t_ext = np.linspace(times[0], times[-1] * 1.1, 50)
        y_upper = res['intercepts'][i, j] + res['ci_upper_pooled'][i, j] * t_ext
        y_lower = res['intercepts'][i, j] + res['ci_lower_pooled'][i, j] * t_ext
        ax.fill_between(t_ext, y_lower, y_upper, alpha=0.2, color='red', label='95% CI (slope)')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Normalized Thickness')
        ax.set_title(f'{name}\nWorst Point ({i},{j}): slope={slopes[i,j]*100:.3f}%/yr')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig('/home/claude/method1_simple.png', dpi=150, bbox_inches='tight')
    print(f"\n시각화 저장: /home/claude/method1_simple.png")

except ImportError:
    print("\nmatplotlib 미설치 - 시각화 생략")
