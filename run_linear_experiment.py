import numpy as np
import matplotlib.pyplot as plt

from src.inference.kalman_filter import KalmanFilter
from src.inference.path_space_filter import PathSpaceFilter


# ==============================
# 1. Define Model Dimensions
# ==============================
z_dim = 2
x_dim = 1
T = 100


# ==============================
# 2. Define System Matrices
# ==============================
A = np.array([[0.9, 0.1],
              [0.0, 0.95]])

C = np.array([[1.0, 0.0]])

Q = 0.1 * np.eye(z_dim)
R = 0.1 * np.eye(x_dim)

mu0 = np.zeros(z_dim)
P0 = np.eye(z_dim)


# ==============================
# 3. Generate NONLINEAR Data
# ==============================

true_states = np.zeros((T, z_dim))
observations = np.zeros((T, x_dim))

true_states[0] = mu0

for t in range(1, T):

    process_noise = np.random.multivariate_normal(np.zeros(z_dim), Q)

    # nonlinear dynamics
    true_states[t] = np.array([
        0.5 * true_states[t-1,0] + 0.2 * (true_states[t-1,0]**2),
        0.95 * true_states[t-1,1]
    ]) + process_noise

    # heavy-tailed observation noise
    measurement_noise = np.random.laplace(0, np.sqrt(R[0,0]), x_dim)

    observations[t] = C @ true_states[t] + measurement_noise


# ==============================
# 4. Run Kalman Filter
# ==============================

kf = KalmanFilter(A, C, Q, R, mu0, P0)

filtered_means, filtered_covs, predicted_covs, log_likelihood = kf.filter(observations)

print("Log-Likelihood:", log_likelihood)


# ==============================
# 5. Run Path-Space Filter
# ==============================

psf = PathSpaceFilter(num_particles=500)

ps_estimates = psf.filter(observations, C, Q, R)

rmse_ps = np.sqrt(np.mean((true_states[:,0] - ps_estimates)**2))

print("PathSpace RMSE:", rmse_ps)


# ==============================
# 6. Plot Results
# ==============================

plt.figure(figsize=(10,6))

# True state
plt.plot(true_states[:,0], label="True State", linewidth=3)

# Kalman mean
plt.plot(filtered_means[:,0], label="Kalman Mean")

# Kalman uncertainty
kalman_std = np.sqrt(filtered_covs[:,0,0])

plt.fill_between(
    range(T),
    filtered_means[:,0] - 2*kalman_std,
    filtered_means[:,0] + 2*kalman_std,
    alpha=0.2,
    label="Kalman ±2σ"
)

# Path-space estimate
plt.plot(ps_estimates, label="PathSpace Estimate")

plt.legend()
plt.title("State Estimation Comparison")
plt.xlabel("Time")
plt.ylabel("State")

plt.tight_layout()
plt.savefig("results/comparison.png")
plt.show()


# ==============================
# 7. Kalman RMSE
# ==============================

rmse_kalman = np.sqrt(np.mean((true_states[:,0] - filtered_means[:,0])**2))

print("Kalman RMSE:", rmse_kalman)


# ==============================
# 8. KL Divergence
# ==============================

true_mean = np.mean(true_states[:,0])
true_var = np.var(true_states[:,0])

est_mean = np.mean(filtered_means[:,0])
est_var = np.var(filtered_means[:,0])

kl_div = 0.5 * (
    np.log(est_var / true_var)
    + (true_var + (true_mean - est_mean)**2) / est_var
    - 1
)

print("KL Divergence:", kl_div)