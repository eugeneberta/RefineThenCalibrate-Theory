# Plots a learning curve for calibration and refinement errors by varying regularization strenght
# \lambda for a given set of problem parameters:
#  - Spectral distribution F parametrized by shift \epsilon and shape parameters \alpha, \beta
#  - Optimal error rate e^*
#  - Dimensions to samples ratio r=p/n
# Checks that theory matches empirical observations with empirical logistic regression on random
# samples from our data model.
# Generates paper Figure 6.

import os
import scipy
import numpy as np
import scipy.stats
from sklearn.linear_model import LogisticRegression
from utils import compute_population_metrics, solve_beta_shifted_system

import matplotlib.pyplot as plt
from tueplots import bundles, fonts, fontsizes, figsizes

plt.rcParams.update(bundles.icml2024())
plt.rcParams.update(fonts.icml2024_tex())
plt.rcParams.update(fontsizes.icml2024())

os.environ['PATH'] += ':/usr/local/texlive/2024/bin/universal-darwin'
plt.rcParams['text.usetex'] = True

# Number of step and lambda range for figure
nsteps = 20
lmbdas = np.linspace(0, -3, nsteps, endpoint=False)

# Spectral distribution parameters
a=1.
b=1.
eps=1e-3

# Problem parameters
r = .5
opterr = .1

# Computing c from e^*
c = -scipy.stats.norm.ppf(opterr)/np.sqrt(scipy.special.hyp2f1(1, a, a+b, -1/eps)/eps)

# Empirical experiment parameters
nseed = 50
n = 2000
p = int(r*n)

# Solving with mathematical model
t_cal, t_ref, t_err = np.zeros(nsteps), np.zeros(nsteps), np.zeros(nsteps)
eta, tau, gamma = None, None, None

for step in range(nsteps):
    # Solving non-linear system
    eta, gamma, tau, mean, var = solve_beta_shifted_system(
        10**lmbdas[step], r, c, a, b, eps,
        eta_init=eta, gamma_init=gamma, tau_init=tau,
        tol=1e-7, lr=min(.7, -1/min(-1e-6, lmbdas[step]))
    )
    # Resulting calibration, refinement and error rate
    cal, ref, err = compute_population_metrics(mean, var)

    # Logging
    t_cal[step] = cal
    t_ref[step] = ref
    t_err[step] = err
    print(f'[solved] lambda={lmbdas[step]:.4f} | cal={cal:.5f} | ref={ref:.5f} | err={err:.5f}')

# Solving empirically
e_cal = np.zeros((nseed, nsteps))
e_ref = np.zeros((nseed, nsteps))
e_err = np.zeros((nseed, nsteps))

for seed in range(nseed):
    # Generating n random samples
    mu = np.random.normal(0,c/np.sqrt(p),p)
    Sigma = np.random.beta(a, b, size=p) + eps

    x1 = np.random.randn(int(n/2), p)
    x1 = x1 * np.sqrt(Sigma) + mu
    y1 = np.ones(int(n/2))

    x2 = np.random.randn(int(n/2), p)
    x2 = x2 * np.sqrt(Sigma) - mu
    y2 = np.zeros(int(n/2))

    x = np.concatenate((x1, x2), axis=0)
    y = np.concatenate((y1, y2), axis=0)

    for step in range(nsteps):
        # Logistic regression
        clf = LogisticRegression(
            penalty='l2',
            C=1/(10**lmbdas[step]*n), # Converting from C to \lambda
            class_weight={0:1, 1:1},
            fit_intercept=False,
            max_iter=1000
        ).fit(x, y, sample_weight=np.ones(n))
        w_hat = clf.coef_[0] # Resulting weight vector

        # Resulting population calibration, refinement and error rate
        mean = np.dot(mu, w_hat)
        var = np.dot(w_hat, np.multiply(Sigma, w_hat))
        cal, ref, err = compute_population_metrics(mean, var)

        # Logging
        e_cal[seed, step] = cal
        e_ref[seed, step] = ref
        e_err[seed, step] = err
        print(f'[solved] seed={seed} | lambda={lmbdas[step]:.4f} | cal={cal:.5f} | ref={ref:.5f} | err={err:.5f}')

# Statistics
mean_cal = np.mean(e_cal, axis=0)
mean_ref = np.mean(e_ref, axis=0)
mean_err = np.mean(e_err, axis=0)
# 95% confidence intervals using the normal distribution (50 seeds is enough).
std_cal = 1.96*np.std(e_cal, axis=0)/np.sqrt(nseed)
std_ref = 1.96*np.std(e_ref, axis=0)/np.sqrt(nseed)
std_err = 1.96*np.std(e_err, axis=0)/np.sqrt(nseed)

# Loss = Cal + Ref
t_loss = t_cal+t_ref
e_loss = e_cal+e_ref
mean_loss = np.mean(e_loss, axis=0)
std_loss = 1.96*np.std(e_loss, axis=0)/np.sqrt(nseed)

# Plotting results
plt.rcParams.update(figsizes.icml2024_half(nrows=3, ncols=1, height_to_width_ratio=0.3))

fig, axs = plt.subplots(3, 1, sharex=True)

# Plot data on each subplot
axs[0].plot(10**lmbdas, t_loss, color='gray', label='theoretical model')
axs[0].plot(10**lmbdas, mean_loss, color='tab:green', label='empirical mean', linestyle='dotted')
axs[0].fill_between(10**lmbdas, mean_loss-std_loss, mean_loss+std_loss, color='tab:green', alpha=0.2, label=r'empirical 95\% CI')
axs[0].set_xscale('log')
axs[0].set_ylabel('cross-entropy')
axs[0].legend()

axs[1].plot(10**lmbdas, t_cal, color='gray', label='theoretical model')
axs[1].plot(10**lmbdas, mean_cal, color='tab:blue', label='empirical mean', linestyle='dotted')
axs[1].fill_between(10**lmbdas, mean_cal-std_cal, mean_cal+std_cal, color='tab:blue', alpha=0.2, label=r'empirical 95\% CI')
axs[1].set_xscale('log')
axs[1].set_ylabel('calibration error')
axs[1].legend()

axs[2].plot(10**lmbdas, t_ref, color='gray', label='theoretical model')
axs[2].plot(10**lmbdas, mean_ref, color='tab:red', label='empirical mean', linestyle='dotted')
axs[2].fill_between(10**lmbdas, mean_ref-std_ref, mean_ref+std_ref, color='tab:red', alpha=0.2, label=r'empirical 95\% CI')
axs[2].set_xscale('log')
axs[2].set_ylabel('refinement error')
axs[2].legend()

axs[2].set_xlabel(r'$\lambda$')
plt.xlim(10**-2.85, 1)
plt.gca().invert_xaxis() # Reverse x-axis

plt.savefig("figures/curves_1.pdf")

plt.show()
