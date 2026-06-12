"""
Final project --- Gaussian-process regression and Bayesian optimization
Companion code for "A Bayesian Way of Thinking".

We design a ground-truth f on [0,1] with zero average, simulate noisy data,
estimate f with two models, then optimize f by querying it.

  Model 1  Piecewise-linear basis (a Gaussian process).
           f(x) = sum_{j=1}^p theta_j h_j(x),  h_j(x) = max(0, x - b_j),
           theta_j ~ N(0, tau^2) i.i.d.  Then f is a zero-mean GP whose covariance
           uses the kernel  K(x,x') = <h(x), h(x')> = sum_j h_j(x) h_j(x'):
               Cov(f(x), f(x')) = tau^2 K(x,x').

  Model 2  Squared-exponential (RBF) Gaussian process, with the same convention
               Cov(f(x), f(x')) = tau^2 K(x,x'),   K(x,x') = exp(-gamma (x-x')^2).

sigma^2 is KNOWN in both models; gamma is KNOWN in Model 2 (vary it to see its
effect).  In each model we choose tau^2 by a 1-D grid search on the marginal
likelihood (the evidence).  For Bayesian optimization, all of sigma^2, tau^2,
gamma are assumed known.

Dependencies: numpy, matplotlib only.
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ------------------------------------------------------------------
# 0.  the ground-truth f (designed to have zero average) and the data
# ------------------------------------------------------------------
def f_raw(x):
    """Any function you like; we will subtract its average to centre it."""
    x = np.asarray(x, dtype=float)
    return 0.5 + np.sin(2 * np.pi * x) + 0.3 * np.sin(6 * np.pi * x)

# force f to have zero average by subtracting the mean (here the 0.5 offset)
fbar = f_raw(np.linspace(0, 1, 4001)).mean()
def f(x):
    return f_raw(x) - fbar

sigma = 0.07            # KNOWN observation-noise std (a design parameter)
n     = 40              # number of training points
X = rng.uniform(0.0, 1.0, size=n)
Y = f(X) + sigma * rng.standard_normal(n)

M    = 200                              # grid resolution
grid = np.linspace(0.0, 1.0, M + 1)     # x0 in {m/M : m = 0,...,M}
ftrue = f(grid)

# ------------------------------------------------------------------
# 1.  generic GP machinery (works with the covariance Cov = tau^2 * kernel)
# ------------------------------------------------------------------
def gp_posterior(Kcov, Kstar_cov, Kss_cov, Y, sigma):
    """Posterior over the NOISELESS function values f(x0).

        Kcov      (n,n)  Cov(f(X), f(X))       = tau^2 * K(X, X)
        Kstar_cov (g,n)  Cov(f(grid), f(X))    = tau^2 * K(grid, X)
        Kss_cov   (g,)   Cov(f(x0), f(x0))     = tau^2 * K(x0, x0)

    Returns posterior mean mu (g,) and sd (g,):
        mu(x0)   = tau^2 k0^T (tau^2 K + sigma^2 I)^{-1} Y
        sd(x0)^2 = tau^2 k00 - tau^4 k0^T (tau^2 K + sigma^2 I)^{-1} k0
    """
    n = Kcov.shape[0]
    C = Kcov + sigma ** 2 * np.eye(n)
    L = np.linalg.cholesky(C)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))     # C^{-1} Y
    mu = Kstar_cov @ alpha
    V = np.linalg.solve(L, Kstar_cov.T)                     # L^{-1} (Kstar_cov)^T
    var = Kss_cov - np.sum(V * V, axis=0)
    return mu, np.sqrt(np.clip(var, 1e-12, None))

def log_evidence(Kcov, Y, sigma):
    """log marginal likelihood  log N(Y; 0, Kcov + sigma^2 I), Kcov = tau^2 * K(X,X)."""
    n = Kcov.shape[0]
    L = np.linalg.cholesky(Kcov + sigma ** 2 * np.eye(n))
    a = np.linalg.solve(L, Y)
    return -0.5 * a @ a - np.sum(np.log(np.diag(L))) - 0.5 * n * math.log(2 * math.pi)

def fit_tau2(K_XX, Y, sigma, tau2_grid):
    """1-D grid search for tau^2 maximizing the evidence; K_XX = K(X,X) without tau^2."""
    ev = [log_evidence(t * K_XX, Y, sigma) for t in tau2_grid]
    return tau2_grid[int(np.argmax(ev))]

tau2_grid = np.logspace(-3, 3, 80)

# ------------------------------------------------------------------
# 2.  Model 1 : piecewise-linear basis  ->  Gaussian process
#     kernel  K(x,x') = <h(x), h(x')>,   Cov = tau^2 K
# ------------------------------------------------------------------
def hinges(x, b):
    """hinge features (max(0,x-b_1), ..., max(0,x-b_p)).  x:(k,) -> (k, len(b))."""
    x = np.atleast_1d(np.asarray(x, float))[:, None]
    return np.maximum(0.0, x - b[None, :])

p = 60
b = np.linspace(0.0, 1.0, p + 2)[1:-1]      # p equally spaced interior breakpoints (p > n: overparametrized)
HX, HG = hinges(X, b), hinges(grid, b)
KXX_1 = HX @ HX.T                            # K(X,X) = <h(x_i), h(x_k)>
tau2_1 = fit_tau2(KXX_1, Y, sigma, tau2_grid)
mu1, sd1 = gp_posterior(tau2_1 * KXX_1, tau2_1 * (HG @ HX.T),
                        tau2_1 * np.sum(HG * HG, axis=1), Y, sigma)

# ------------------------------------------------------------------
# 3.  Model 2 : squared-exponential (RBF) GP,  gamma KNOWN
#     kernel  K(x,x') = exp(-gamma (x-x')^2),   Cov = tau^2 K
# ------------------------------------------------------------------
def rbf(A, B, gamma):
    A = np.asarray(A, float); B = np.asarray(B, float)
    return np.exp(-gamma * (A[:, None] - B[None, :]) ** 2)

gamma = 25.0                                  # KNOWN lengthscale (vary it below)
KXX_2 = rbf(X, X, gamma)
tau2_2 = fit_tau2(KXX_2, Y, sigma, tau2_grid)
mu2, sd2 = gp_posterior(tau2_2 * KXX_2, tau2_2 * rbf(grid, X, gamma),
                        tau2_2 * np.ones(len(grid)), Y, sigma)

# ------------------------------------------------------------------
# 4.  plot the two predictive mean curves with uncertainty bands
# ------------------------------------------------------------------
def plot_fit(ax, mu, sd, title):
    ax.fill_between(grid, mu - 2 * sd, mu + 2 * sd, color="0.85",
                    label=r"$\mu(x_0)\pm 2\sigma(x_0)$")
    ax.plot(grid, ftrue, "k--", lw=1.2, label=r"true $f$")
    ax.plot(grid, mu, color="C0", lw=2, label=r"posterior mean $\mu(x_0)$")
    ax.plot(X, Y, "o", color="C3", ms=4, label="data")
    ax.set_title(title); ax.set_xlabel(r"$x_0$"); ax.set_ylim(-1.5, 1.5)
    ax.legend(fontsize=8, loc="upper right")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
plot_fit(axes[0], mu1, sd1,
         fr"Model 1: piecewise-linear GP  ($p={p}$, $\tau^2$={tau2_1:.2g})")
plot_fit(axes[1], mu2, sd2,
         fr"Model 2: RBF GP  ($\gamma$={gamma:.0f} known, $\tau^2$={tau2_2:.2g})")
fig.tight_layout(); fig.savefig("fit_models.png", dpi=130)

# ------------------------------------------------------------------
# 4b. Model 1: vary the number of breakpoints p to see its effect
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, p_ in zip(axes, [5, 16, 61]):
    b_ = np.linspace(0.0, 1.0, p_ + 2)[1:-1]
    Hx, Hg = hinges(X, b_), hinges(grid, b_)
    KXX = Hx @ Hx.T
    t = fit_tau2(KXX, Y, sigma, tau2_grid)
    mu, sd = gp_posterior(t * KXX, t * (Hg @ Hx.T), t * np.sum(Hg * Hg, axis=1), Y, sigma)
    ax.fill_between(grid, mu - 2 * sd, mu + 2 * sd, color="0.85")
    ax.plot(grid, ftrue, "k--", lw=1); ax.plot(grid, mu, "C0", lw=2)
    ax.plot(X, Y, "o", color="C3", ms=3)
    ax.set_title(fr"$p={p_}$  ($\tau^2$={t:.2g})"); ax.set_xlabel(r"$x_0$")
    ax.set_ylim(-1.5, 1.5)
axes[0].set_ylabel("prediction")
fig.suptitle(r"Model 1: effect of the number of breakpoints $p$ "
             r"(more breakpoints = more flexible; $p>n$ is overparametrized)")
fig.tight_layout(); fig.savefig("vary_p.png", dpi=130)

# ------------------------------------------------------------------
# 4c. Model 2: vary the lengthscale gamma to see its effect
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, g_ in zip(axes, [6.0, 26.0, 151.0]):
    KXX = rbf(X, X, g_)
    t = fit_tau2(KXX, Y, sigma, tau2_grid)
    mu, sd = gp_posterior(t * KXX, t * rbf(grid, X, g_), t * np.ones(len(grid)), Y, sigma)
    ax.fill_between(grid, mu - 2 * sd, mu + 2 * sd, color="0.85")
    ax.plot(grid, ftrue, "k--", lw=1); ax.plot(grid, mu, "C0", lw=2)
    ax.plot(X, Y, "o", color="C3", ms=3)
    ax.set_title(fr"$\gamma={g_:.0f}$  ($\tau^2$={t:.2g})"); ax.set_xlabel(r"$x_0$")
    ax.set_ylim(-1.5, 1.5)
axes[0].set_ylabel("prediction")
fig.suptitle(r"Model 2: effect of the lengthscale $\gamma$ "
             r"(small $\gamma$ = smoother, large $\gamma$ = wigglier)")
fig.tight_layout(); fig.savefig("vary_gamma.png", dpi=130)

# ------------------------------------------------------------------
# 5.  Bayesian optimization of f  (all of sigma^2, tau^2, gamma KNOWN)
# ------------------------------------------------------------------
def Phi(z):                                   # standard normal CDF (dependency-free)
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(z) / math.sqrt(2.0)))

def surrogate(Xq, Yq, grid, tau2, gamma, sigma):
    return gp_posterior(tau2 * rbf(Xq, Xq, gamma), tau2 * rbf(grid, Xq, gamma),
                        tau2 * np.ones(len(grid)), Yq, sigma)

def ucb(mu, sd, alpha=2.0):  return mu + alpha * sd        # upper confidence bound
def pi(mu, sd, fbest):       return Phi((mu - fbest) / sd) # probability of improvement

tau2_bo, gamma_bo, alpha_ucb = 1.0, 25.0, 2.0   # all assumed KNOWN
n_init, budget = 4, 15

Xq = list(rng.uniform(0, 1, n_init))
Yq = [float(f(x) + sigma * rng.standard_normal()) for x in Xq]
best_so_far = [max(Yq)]
snapshots = []

for t in range(budget):
    mu_s, sd_s = surrogate(np.array(Xq), np.array(Yq), grid, tau2_bo, gamma_bo, sigma)
    acq = ucb(mu_s, sd_s, alpha_ucb)            # try pi(mu_s, sd_s, max(Yq)) instead
    x_next = grid[int(np.argmax(acq))]
    if t in (0, budget - 1):
        snapshots.append((t, mu_s.copy(), sd_s.copy(), acq.copy(), x_next,
                          list(Xq), list(Yq)))
    y_next = float(f(x_next) + sigma * rng.standard_normal())
    Xq.append(x_next); Yq.append(y_next)
    best_so_far.append(max(Yq))

x_star = grid[int(np.argmax(ftrue))]
print(f"true argmax = {x_star:.3f} (f = {np.max(ftrue):.3f})")
print(f"BO best query x = {Xq[int(np.argmax(Yq))]:.3f}, observed y = {max(Yq):.3f}")
print(f"Model 1: tau^2 = {tau2_1:.3g}")
print(f"Model 2: tau^2 = {tau2_2:.3g}  (gamma = {gamma:.0f})")

fig, axes = plt.subplots(2, 2, figsize=(12, 7),
                         gridspec_kw={"height_ratios": [2, 1]})
for col, snap in zip((0, 1), (snapshots[0], snapshots[-1])):
    t, mu_s, sd_s, acq, x_next, Xs, Ys = snap
    ax = axes[0, col]
    ax.fill_between(grid, mu_s - 2 * sd_s, mu_s + 2 * sd_s, color="0.85")
    ax.plot(grid, ftrue, "k--", lw=1, label="true $f$")
    ax.plot(grid, mu_s, "C0", lw=2, label="GP mean")
    ax.plot(Xs, Ys, "ro", ms=4, label="queries")
    ax.axvline(x_next, color="C2", ls=":", label="next $x$")
    ax.set_title(f"iteration {t}: GP posterior"); ax.set_ylim(-1.8, 1.8)
    ax.legend(fontsize=7, loc="upper right")
    axa = axes[1, col]
    axa.plot(grid, acq, "C1"); axa.axvline(x_next, color="C2", ls=":")
    axa.set_title(r"UCB acquisition $a(x)=\mu+2\sigma$"); axa.set_xlabel(r"$x$")
fig.tight_layout(); fig.savefig("bayesopt.png", dpi=130)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(len(best_so_far)), best_so_far, "o-")
ax.axhline(np.max(ftrue), color="k", ls="--", label="true maximum")
ax.set_xlabel("iteration"); ax.set_ylabel("best observed $y$")
ax.set_title("Bayesian optimization progress"); ax.legend()
fig.tight_layout(); fig.savefig("bo_progress.png", dpi=130)

print("saved figures: fit_models.png, vary_p.png, vary_gamma.png, bayesopt.png, bo_progress.png")
