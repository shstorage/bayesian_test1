import numpy as np
import matplotlib.pyplot as plt

def hdi_of_grid(prob_mass_vec, cred_mass=0.95):
    """
    그리드 상의 확률 질량 함수에서 최고 밀도 구간(HDI)을 계산합니다.
    """
    sorted_prob_mass = np.sort(prob_mass_vec)[::-1]
    cum_prob = np.cumsum(sorted_prob_mass)
    
    # 누적 확률이 cred_mass를 넘어서는 첫 번째 인덱스
    hdi_height_idx = np.min(np.where(cum_prob >= cred_mass))
    hdi_height = sorted_prob_mass[hdi_height_idx]
    
    # 높이가 hdi_height 이상인 모든 인덱스를 찾음
    indices = np.where(prob_mass_vec >= hdi_height)[0]
    hdi_mass = np.sum(prob_mass_vec[indices])
    
    return {'indices': indices, 'mass': hdi_mass, 'height': hdi_height}

def BernGrid(Theta, pTheta, Data, showCentTend='Mode', showHDI=True, credMass=0.95):
    """
    베르누이 가능도를 이용해 베이즈 정리를 그리드 근사로 계산하고 시각화합니다.
    """
    # 1. 입력값 검증 (Error Handling)
    if np.any((Theta > 1) | (Theta < 0)):
        raise ValueError("Theta values must be between 0 and 1")
    if np.any(pTheta < 0):
        raise ValueError("pTheta values must be non-negative")
    if not np.isclose(np.sum(pTheta), 1.0):
        raise ValueError("pTheta values must sum to 1.0")
    if not np.all(np.isin(Data, [0, 1])):
        raise ValueError("Data values must be 0 or 1")

    # 2. 데이터 요약
    z = np.sum(Data)
    N = len(Data)

    # 3. 베이즈 정리 핵심 계산 (가능도, 증거, 사후확률)
    pDataGivenTheta = (Theta**z) * ((1 - Theta)**(N - z)) # 가능도 (Likelihood)
    pData = np.sum(pDataGivenTheta * pTheta)              # 증거 (Marginal Likelihood)
    pThetaGivenData = (pDataGivenTheta * pTheta) / pData  # 사후확률 (Posterior)

    # 4. 시각화 (그래프 그리기)
    fig, axes = plt.subplots(3, 1, figsize=(6, 8))
    plt.subplots_adjust(hspace=0.6) # 그래프 간격 조정

    # 보조 함수: 막대 그래프(Bars) 그리기
    def plot_bars(ax, x, y, title, ylabel):
        ax.vlines(x, 0, y, color='skyblue', linewidth=2)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 1.1 * np.max(y) if np.max(y) > 0 else 1)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(r'$\theta$', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

    # 보조 함수: 중앙값(Mean/Mode) 텍스트 표시 (오류 수정됨)
    def add_cent_tend(ax, x, dist, title):
        if showCentTend == 'Mean':
            mean_val = np.sum(x * dist)
            text_x = 0.05 if mean_val > 0.5 else 0.95
            ha = 'left' if mean_val > 0.5 else 'right'
            ax.text(text_x, 0.8 * ax.get_ylim()[1], f"mean={mean_val:.3f}", ha=ha, fontsize=12)
        elif showCentTend == 'Mode':
            mode_val = x[np.argmax(dist)]
            text_x = 0.05 if mode_val > 0.5 else 0.95
            ha = 'left' if mode_val > 0.5 else 'right'
            ax.text(text_x, 0.8 * ax.get_ylim()[1], f"mode={mode_val:.3f}", ha=ha, fontsize=12)

    # 보조 함수: HDI 표시
    def add_hdi(ax, x, dist):
        if showHDI:
            hdi_info = hdi_of_grid(dist, credMass)
            hdi_idx = hdi_info['indices']
            hdi_height = hdi_info['height']
            
            # HDI 선 그리기
            ax.hlines(hdi_height, x[hdi_idx[0]], x[hdi_idx[-1]], color='black', linewidth=2, linestyle='--')
            ax.text(np.mean(x[hdi_idx]), hdi_height * 1.05, f"{credMass*100:.1f}% HDI", ha='center', va='bottom', fontsize=10)
            ax.text(x[hdi_idx[0]], hdi_height, f"{x[hdi_idx[0]]:.3f}", ha='center', va='top', fontsize=10)
            ax.text(x[hdi_idx[-1]], hdi_height, f"{x[hdi_idx[-1]]:.3f}", ha='center', va='top', fontsize=10)

    # [1] 사전확률 (Prior) 플롯
    plot_bars(axes[0], Theta, pTheta, "Prior", r'$p(\theta)$')
    add_cent_tend(axes[0], Theta, pTheta, "Prior")

    # [2] 가능도 (Likelihood) 플롯
    plot_bars(axes[1], Theta, pDataGivenTheta, "Likelihood", r'$p(D|\theta)$')
    add_cent_tend(axes[1], Theta, pDataGivenTheta, "Likelihood")
    # 가능도 그래프에 Data 요약 정보 추가
    text_x = 0.05 if z > 0.5 * N else 0.95
    ha = 'left' if z > 0.5 * N else 'right'
    axes[1].text(text_x, 0.9 * axes[1].get_ylim()[1], f"Data: z={z}, N={N}", ha=ha, fontsize=12, color='red')

    # [3] 사후확률 (Posterior) 플롯
    plot_bars(axes[2], Theta, pThetaGivenData, "Posterior", r'$p(\theta|D)$')
    add_cent_tend(axes[2], Theta, pThetaGivenData, "Posterior")
    add_hdi(axes[2], Theta, pThetaGivenData)

    plt.show()
    return pThetaGivenData

# ==========================================
# 실행 부분
# ==========================================

if __name__ == '__main__':
    # 1. 0부터 1까지 1001개로 쪼갠 그리드 (Theta)
    Theta = np.linspace(0, 1, 1001)

    # 2. 삼각형 모양의 사전확률(pTheta)
    pTheta = np.minimum(Theta, 1 - Theta)
    pTheta = pTheta / np.sum(pTheta) # 정규화

    # 3. 데이터: 뒷면(0) 3번, 앞면(1) 1번
    Data = np.array([0, 0, 0, 1])

    # 4. 함수 실행
    posterior = BernGrid(Theta, pTheta, Data, showCentTend="Mode", showHDI=True)