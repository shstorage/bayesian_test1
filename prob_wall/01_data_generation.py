"""
배관 감육 가상 데이터 생성기
=========================
- 12x13 그리드, normalized thickness (1.0 = 원래 두께)
- Set A: 3회 측정 (OH 3번)
- Set B: 6회 측정 (OH 6번)
- 감육 패턴: 타원형, 그리드의 ~30% 영역에 smooth하게 진행
- 최대 감육 ~20% (마지막 시점에서 0.80)
- 측정 노이즈 포함 (UT 특성 반영: 이상치 포함)
"""

import numpy as np
import json
import os

np.random.seed(42)

# =============================================================
# 1. 그리드 설정
# =============================================================
ROWS, COLS = 12, 13

# 그리드 좌표 (0~1로 정규화)
y_grid, x_grid = np.meshgrid(
    np.linspace(0, 1, ROWS),
    np.linspace(0, 1, COLS),
    indexing='ij'
)

# =============================================================
# 2. 타원형 감육 패턴 정의
# =============================================================
# 감육 중심: 그리드의 약간 오른쪽 아래
cx, cy = 0.6, 0.55

# 타원 반축 (그리드의 ~30% 커버하도록)
a, b = 0.30, 0.22  # x방향, y방향 반축
theta = np.radians(25)  # 타원 회전각

# 회전 적용한 타원 거리
dx = x_grid - cx
dy = y_grid - cy
x_rot = dx * np.cos(theta) + dy * np.sin(theta)
y_rot = -dx * np.sin(theta) + dy * np.cos(theta)
ellipse_dist = (x_rot / a) ** 2 + (y_rot / b) ** 2

# 감육 강도 맵: 타원 내부(dist<1)에서 smooth하게 감소
# cosine taper로 smooth edge
thinning_intensity = np.where(
    ellipse_dist < 1.0,
    0.5 * (1 + np.cos(np.pi * np.sqrt(ellipse_dist))),  # 중심에서 1, 경계에서 0
    0.0
)

# 최대 감육률 (연간): 최대 감육 포인트가 마지막 시점에서 ~20% 감육되도록
MAX_THINNING_TOTAL = 0.20  # 전체 기간 동안 최대 감육량


def generate_dataset(n_measurements: int, total_years: float, label: str):
    """
    감육 측정 데이터셋 생성
    
    Args:
        n_measurements: OH 측정 횟수
        total_years: 전체 기간 (년)
        label: 데이터셋 라벨
    
    Returns:
        dict with measurement data
    """
    # 측정 시점 (불균등 간격 - 실제 OH처럼)
    if n_measurements == 3:
        times = np.array([0.0, 3.5, 8.0])  # 년
    elif n_measurements == 6:
        times = np.array([0.0, 2.0, 4.0, 5.5, 7.0, 8.0])  # 년
    else:
        times = np.sort(np.concatenate([[0.0], 
                         np.random.uniform(1, total_years, n_measurements - 2),
                         [total_years]]))
    
    # 연간 감육률 맵 (위치별)
    annual_rate = thinning_intensity * (MAX_THINNING_TOTAL / total_years)
    
    # 측정 데이터 생성 (시점 x 행 x 열)
    data = np.zeros((n_measurements, ROWS, COLS))
    
    # UT 측정 노이즈 파라미터
    sigma_noise = 0.008   # 기본 측정 노이즈 (~0.8%)
    outlier_prob = 0.03   # 이상치 확률 3%
    outlier_scale = 0.04  # 이상치 크기 (~4%)
    
    for t_idx, t in enumerate(times):
        # 실제 두께: 1.0 - (감육률 × 시간)
        true_thickness = 1.0 - annual_rate * t
        
        # 기본 가우시안 노이즈
        noise = np.random.normal(0, sigma_noise, (ROWS, COLS))
        
        # 이상치 추가 (UT 측정 특성: 가끔 크게 벗어남)
        outlier_mask = np.random.random((ROWS, COLS)) < outlier_prob
        outlier_noise = np.random.normal(0, outlier_scale, (ROWS, COLS))
        noise = np.where(outlier_mask, outlier_noise, noise)
        
        # 약간의 체계적 편의 (측정자 간 차이 시뮬레이션)
        systematic_bias = np.random.normal(0, 0.003)
        
        data[t_idx] = true_thickness + noise + systematic_bias
    
    return {
        'label': label,
        'n_measurements': n_measurements,
        'times': times.tolist(),
        'data': data.tolist(),
        'rows': ROWS,
        'cols': COLS,
        'thinning_intensity': thinning_intensity.tolist(),
        'annual_rate': annual_rate.tolist(),
    }


# =============================================================
# 3. 데이터 생성
# =============================================================
dataset_A = generate_dataset(n_measurements=3, total_years=8.0, label='Set_A_3meas')
dataset_B = generate_dataset(n_measurements=6, total_years=8.0, label='Set_B_6meas')

# =============================================================
# 4. 저장
# =============================================================
output = {
    'description': '배관 감육 UT 측정 가상 데이터',
    'grid_size': f'{ROWS}x{COLS}',
    'max_thinning_pct': MAX_THINNING_TOTAL * 100,
    'noise_sigma_pct': 0.8,
    'outlier_prob_pct': 3.0,
    'datasets': {
        'set_A': dataset_A,
        'set_B': dataset_B,
    }
}

output_path = 'pipe_thinning_data.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

# =============================================================
# 5. 요약 출력 및 시각화
# =============================================================
print("=" * 60)
print("배관 감육 가상 데이터 생성 완료")
print("=" * 60)

for name, ds in [('Set A (3회 측정)', dataset_A), ('Set B (6회 측정)', dataset_B)]:
    data = np.array(ds['data'])
    times = np.array(ds['times'])
    print(f"\n--- {name} ---")
    print(f"  측정 시점 (년): {[f'{t:.1f}' for t in times]}")
    print(f"  그리드: {ds['rows']}x{ds['cols']} = {ds['rows']*ds['cols']} 포인트")
    print(f"  초기 평균 두께: {data[0].mean():.4f}")
    print(f"  최종 평균 두께: {data[-1].mean():.4f}")
    print(f"  최종 최소 두께: {data[-1].min():.4f}")
    print(f"  감육 영역 포인트 수: {np.sum(np.array(ds['thinning_intensity']) > 0.01)}")

# 시각화
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('Synthetic Pipe Wall Thinning Data', fontsize=14, fontweight='bold')
    
    # Row 0: Set A
    data_A = np.array(dataset_A['data'])
    times_A = dataset_A['times']
    for i in range(3):
        im = axes[0, i].imshow(data_A[i], cmap='RdYlGn', vmin=0.78, vmax=1.03,
                                aspect='auto', origin='lower')
        axes[0, i].set_title(f'Set A - t={times_A[i]:.1f}yr')
        axes[0, i].set_xlabel('Column')
        axes[0, i].set_ylabel('Row')
        plt.colorbar(im, ax=axes[0, i], shrink=0.8)
    
    # True thinning intensity
    im = axes[0, 3].imshow(thinning_intensity, cmap='Reds', aspect='auto', origin='lower')
    axes[0, 3].set_title('True Thinning Intensity')
    plt.colorbar(im, ax=axes[0, 3], shrink=0.8)
    
    # Row 1: Set B (선택적 시점)
    data_B = np.array(dataset_B['data'])
    times_B = dataset_B['times']
    show_idx = [0, 2, 5]  # 첫, 중간, 마지막
    for k, i in enumerate(show_idx):
        im = axes[1, k].imshow(data_B[i], cmap='RdYlGn', vmin=0.78, vmax=1.03,
                                aspect='auto', origin='lower')
        axes[1, k].set_title(f'Set B - t={times_B[i]:.1f}yr')
        axes[1, k].set_xlabel('Column')
        axes[1, k].set_ylabel('Row')
        plt.colorbar(im, ax=axes[1, k], shrink=0.8)
    
    # 특정 포인트 시계열
    # 감육 중심 근처 포인트
    ci, cj = 7, 8  # 감육 중심 근처
    ax = axes[1, 3]
    ax.plot(times_A, data_A[:, ci, cj], 'ro-', label=f'Set A ({ci},{cj})', markersize=8)
    ax.plot(times_B, data_B[:, ci, cj], 'bs-', label=f'Set B ({ci},{cj})', markersize=6)
    # 감육 없는 포인트
    ni, nj = 1, 1
    ax.plot(times_A, data_A[:, ni, nj], 'r^--', label=f'Set A ({ni},{nj})', markersize=8, alpha=0.5)
    ax.plot(times_B, data_B[:, ni, nj], 'bv--', label=f'Set B ({ni},{nj})', markersize=6, alpha=0.5)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Normalized Thickness')
    ax.set_title('Time Series at Selected Points')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig('data_overview.png', dpi=150, bbox_inches='tight')
    print(f"\n시각화 저장: data_overview.png")
    
except ImportError:
    print("\nmatplotlib 미설치 - 시각화 생략")

print(f"\n데이터 저장: {output_path}")
