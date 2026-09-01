import numpy as np
import matplotlib.pyplot as plt

from src.inference.kalman_filter import KalmanFilter


# ==============================
# Experiment Settings
# ==============================

T = 100
z_dim = 1

np.random.seed(42)


# ==============================
# Generate Nonlinear System
# ==============================

true_states = np.zeros(T)
observations = np.zeros(T)

true_states[0] = 0

for t in range(1, T):

    process_noise = np.random.normal(0, 0.3)

    # nonlinear dynamics
    true_states[t] = (
        0.5 * true_states[t-1]
        + 0.2 * true_states[t-1]**2
        + process_noise
    )

    # non-Gaussian noise
    measurement_noise = np.random.laplace(0, 0.3)

    observations[t] = true_states[t] + measurement_noise


observations = observations.reshape(-1,1)


# ==============================
# Kalman Filter
# ==============================

A = np.array([[0.9]])
C = np.array([[1.0]])

Q = np.array([[0.3]])
R = np.array([[0.3]])

mu0 = np.array([0.0])
P0 = np.array([[1.0]])

kf = KalmanFilter(A, C, Q, R, mu0, P0)

kf_means, _, _, log_likelihood = kf.filter(observations)

kf_estimates = kf_means[:,0]


# ==============================
# Particle Filter
# ==============================

num_particles = 500

particles = np.random.randn(num_particles)
weights = np.ones(num_particles) / num_particles

pf_estimates = np.zeros(T)

for t in range(T):

    # propagate
    particles = (
        0.5 * particles
        + 0.2 * particles**2
        + np.random.normal(0, 0.3, num_particles)
    )

    # likelihood
    likelihood = np.exp(
        -(observations[t,0] - particles)**2 / (2 * 0.3)
    )

    weights *= likelihood
    weights += 1e-12
    weights /= np.sum(weights)

    pf_estimates[t] = np.sum(weights * particles)

    # resample
    idx = np.random.choice(num_particles, num_particles, p=weights)
    particles = particles[idx]
    weights = np.ones(num_particles) / num_particles


# ==============================
# Metrics
# ==============================

rmse_kf = np.sqrt(np.mean((true_states - kf_estimates)**2))
rmse_pf = np.sqrt(np.mean((true_states - pf_estimates)**2))

print("Kalman RMSE:", rmse_kf)
print("Particle RMSE:", rmse_pf)

print("Log Likelihood:", log_likelihood)

improvement = (rmse_kf - rmse_pf) / rmse_kf * 100
print("Particle improvement over Kalman:", improvement, "%")


# ==============================
# Plot
# ==============================

plt.figure(figsize=(10,6))

plt.plot(true_states, label="True State")
plt.scatter(range(T), observations[:,0], s=10, alpha=0.4, label="Observations")

plt.plot(kf_estimates, label="Kalman")
plt.plot(pf_estimates, label="Particle Filter")

plt.legend()

plt.title("Filtering Comparison")

plt.show()