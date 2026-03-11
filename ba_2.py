import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# 설정
theta = np.linspace(0, 1, 1001)  # θ를 0~1 사이 1001개로 이산화

# 사전분포: 삼각형 모양 (θ=0.5에서 최대)
prior = 1 - np.abs(theta - 0.5) * 2
prior = prior / prior.sum()  # 정규화

# 실험 설정: z/N (앞면 횟수 / 전체 횟수)
experiments = [
    {"N": 4,  "z": 1,  "label": "N=4,  z=1"},
    {"N": 40, "z": 10, "label": "N=40, z=10"},
]

fig, axes = plt.subplots(3, 2, figsize=(13, 10))
fig.suptitle("Bayesian Coin Flip (Binomial, 1001 discrete theta values)", fontsize=13, fontweight='bold')

for col, exp in enumerate(experiments):
    N, z = exp["N"], exp["z"]

    likelihood = binom.pmf(z, N, theta)

    posterior = prior * likelihood
    posterior = posterior / posterior.sum()

    # ---- Prior ----
    ax = axes[0, col]
    ax.plot(theta, prior, color='steelblue', linewidth=1.5)
    ax.fill_between(theta, prior, alpha=0.3, color='steelblue')
    ax.set_title(f"Prior  ({exp['label']})", fontsize=11)
    ax.set_xlabel("θ")
    ax.set_ylabel("p(θ)")
    ax.set_xlim(0, 1)

    # ---- Likelihood ----
    ax = axes[1, col]
    ax.plot(theta, likelihood, color='darkorange', linewidth=1.5)
    ax.fill_between(theta, likelihood, alpha=0.3, color='darkorange')
    ax.set_title(f"Likelihood  ({exp['label']})", fontsize=11)
    ax.set_xlabel("θ")
    ax.set_ylabel("p(D|θ)")
    ax.set_xlim(0, 1)

    # ---- Posterior ----
    ax = axes[2, col]
    ax.plot(theta, posterior, color='seagreen', linewidth=1.5)
    ax.fill_between(theta, posterior, alpha=0.3, color='seagreen')
    ax.set_title(f"Posterior  ({exp['label']})", fontsize=11)
    ax.set_xlabel("θ")
    ax.set_ylabel("p(θ|D)")
    ax.set_xlim(0, 1)

    # MAP 표시
    map_theta = theta[np.argmax(posterior)]
    ax.axvline(map_theta, color='red', linestyle='--', linewidth=1.2,
               label=f"MAP={map_theta:.3f}")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.show()
