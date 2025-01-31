import os
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import newton
from scipy.special import roots_hermite, expit, hyp2f1, beta
from scipy.stats import norm

def normalpdf(x, mean, var):
    return np.exp(-(x-mean)**2/(2*var))/np.sqrt(2*np.pi*var)

# Efficient solver for the non-linear system in:
# Mai, X., Liao, Z., and Couillet, R.
# A large scale analysis of logistic regression: Asymptotic performance and new insights.
# In International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2019
# Using the mathematical model that we specified for the spectral distribution F.

def derivative(x, kappa, t):
    # finding the gradient root instead of minimizing
    # the convex problem yields much faster convergence.
    return x - t - kappa/(1+np.exp(x))

def second_derivative(x, kappa, t):
    return 1 + kappa*np.exp(x)/(1+np.exp(x))**2

def g(t, kappa):
    # g in original paper
    x0 = np.maximum(t, 0.) # smart initialization scheme
    x = newton(derivative, x0, args=(kappa, t), fprime=second_derivative)
    return 1/(1+np.exp(x))

def phi_prime(x):
    # phi prime in original paper
    return -np.exp(x)/(1+np.exp(x))**2


### SOLUTION FOR NON-SHIFTED BETA DISTRIBUTION (not included in our paper).
def find_kappa(lmbda, delta, mean, var, kappa_init=1., tol=1e-6):
    # find kappa from original paper
    kappa = kappa_init
    points, weights = roots_hermite(50) # Fast gaussian integration
    while True:
        def integrand(r):
            return 1/(1/phi_prime(r+kappa*g(r,kappa)) - kappa)
        e = np.sum(weights * integrand(mean + np.sqrt(2*var) * points)) / np.sqrt(np.pi)

        kappa_ = -(delta/e)*(1 + np.log(1-e/lmbda)*lmbda/e)
        if abs(kappa-kappa_)<tol:
            return kappa_
        kappa = kappa_

def solve_system(lmbda, delta, c, eta_init, gamma_init, tau_init, tol=1e-6, lr=.7):
    # Solve system from original paper
    kappa = 1.
    if eta_init is not None:
        eta = eta_init
    else:
        eta = .5
    if gamma_init is not None:
        gamma = gamma_init
    else:
        gamma = .5
    if tau_init is not None:
        tau = tau_init
    else:
        tau = .5

    points, weights = roots_hermite(50) # For fast gaussian integration.

    while True:
        m = (eta*c/tau)*np.log(1+tau/lmbda)/2
        var = (c/2)*(eta/tau)**2*(np.log(1+tau/lmbda)-1/(1+lmbda/tau)) + (gamma*delta/tau**2)*(1 + 1/(1+tau/lmbda) - 2*(lmbda/tau)*np.log(1+tau/lmbda))
        kappa = find_kappa(lmbda, delta, m, var, kappa_init=kappa)

        points_ = m + np.sqrt(2*var) * points
        g_vals = g(points_, kappa)

        eta_diff = eta - np.sum(weights * g_vals) / np.sqrt(np.pi)
        gamma_diff = gamma - np.sum(weights * np.square(g_vals)) / np.sqrt(np.pi)
        tau_diff = tau - np.sum(weights * g_vals * (m - points_)) / (np.sqrt(np.pi) * var)

        if (abs(eta_diff)<tol) & (abs(gamma_diff)<tol) & (abs(tau_diff)<tol):
            return eta, gamma, tau, m, var

        eta -= lr*eta_diff
        gamma -= lr*gamma_diff
        tau -= lr*tau_diff

### SOLUTION FOR SHIFTED BETA DISTRIBUTION (included in our paper).
def find_beta_shifted_kappa(lmbda, r, mean, var, a, b, eps, kappa_init=1., tol=1e-6):
    kappa = kappa_init
    points, weights = roots_hermite(50)
    while True:
        def integrand(x):
            return 1/(1/phi_prime(x+kappa*g(x,kappa)) - kappa)
        e = np.sum(weights * integrand(mean + np.sqrt(2*var) * points)) / np.sqrt(np.pi)

        kappa_ = r*(beta(a+1,b)*hyp2f1(1, a+1, a+b+1, 1/(lmbda/e - eps))/beta(a,b) + eps*hyp2f1(1, a, a+b, 1/(lmbda/e - eps)))/(lmbda-e*eps)
        if abs(kappa-kappa_)<tol:
            return kappa_
        kappa = kappa_

def solve_beta_shifted_system(lmbda, r, c, a, b, eps, eta_init, gamma_init, tau_init, tol=1e-6, lr=.7):
    kappa = 1.
    if eta_init is not None:
        eta = eta_init
    else:
        eta = .5
    if gamma_init is not None:
        gamma = gamma_init
    else:
        gamma = .5
    if tau_init is not None:
        tau = tau_init
    else:
        tau = .5

    points, weights = roots_hermite(50)

    while True:
        # Using formulas from the appendix:
        m = eta*(c**2)*hyp2f1(1, a, a+b, -1/(lmbda/tau + eps))/(lmbda+tau*eps)
        var = (eta*c/(lmbda+tau*eps))**2*(eps*hyp2f1(2, a, a+b, -1/(lmbda/tau + eps)) + (beta(a+1,b)/beta(a,b))*hyp2f1(2, a+1, a+b+1, -1/(lmbda/tau + eps)))
        var += gamma*r*((beta(a+2,b)/beta(a,b))*hyp2f1(2, a+2, a+b+2, -1/(lmbda/tau + eps)) + 2*eps*(beta(a+1,b)/beta(a,b))*hyp2f1(2, a+1, a+b+1, -1/(lmbda/tau + eps)) + (eps**2)*hyp2f1(2, a, a+b, -1/(lmbda/tau + eps)))/(lmbda+tau*eps)**2
            
        kappa = find_beta_shifted_kappa(lmbda, r, m, var, a, b, eps, kappa_init=kappa)

        points_ = m + np.sqrt(2*var) * points
        g_vals = g(points_, kappa)

        eta_diff = eta - np.sum(weights * g_vals) / np.sqrt(np.pi)
        gamma_diff = gamma - np.sum(weights * np.square(g_vals)) / np.sqrt(np.pi)
        tau_diff = tau - np.sum(weights * g_vals * (m - points_)) / (np.sqrt(np.pi) * var)

        if (abs(eta_diff)<tol) & (abs(gamma_diff)<tol) & (abs(tau_diff)<tol):
            return eta, gamma, tau, m, var

        eta -= lr*eta_diff
        gamma -= lr*gamma_diff
        tau -= lr*tau_diff

def compute_population_metrics(m, var):
    # Compute calibration, refinement error using theorem 5.1 and error rate.
    norm2 = np.sqrt(var)
    inner = m*2

    def calintegrand(l):
        p = (expit((l+inner/(2*norm2))*inner/norm2)-.5)*.99+.5
        q = (expit(norm2*l + inner/2)-.5)*.99+.5
        return (p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))) * normalpdf(l, 0, 1)
    calibration, _ = quad(calintegrand, -np.inf, np.inf)

    def refintegrand(l):
        p = (expit((l+inner/(2*norm2))*inner/norm2)-.5)*.99+.5
        # return np.log(1/p) * normalpdf(l, 0, 1)
        return -(p*np.log(p)+(1-p)*np.log(1-p)) * normalpdf(l, 0, 1)
    refinement, _ = quad(refintegrand, -np.inf, np.inf)

    # def lossintegrand(l):
    #     # p = expit((l+inner/(2*norm2))*norm2)
    #     # return (1-p)**2 * normalpdf(l, 0, 1)
    #     q = (expit(norm2*l+inner/2)-.5)*.99+.5
    #     return np.log(1/q) * normalpdf(l, 0, 1)
    # loss, _ = quad(lossintegrand, -np.inf, np.inf)

    error = norm.cdf(x=-m/np.sqrt(var))

    return calibration, refinement, error
    # return calibration, refinement, error, loss

def log_experiment(file_path, new_data):
    # Helper for logging.

    # Check if the file exists
    if os.path.isfile(file_path):
        # Load the existing data
        existing_data = pd.read_csv(file_path)
        # Append the new data
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        # If file does not exist, the new data is the full DataFrame
        updated_data = new_data
    
    # Save the updated DataFrame to CSV
    updated_data.to_csv(file_path, index=False)
