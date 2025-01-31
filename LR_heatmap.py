# Run heatmap experiment

import scipy
import numpy as np
import pandas as pd
from utils import log_experiment, solve_beta_shifted_system, compute_population_metrics

logging_path = './ratios_hard.csv'

# Problem parameters: distribution shape
eps=1e-3
a=5.
b=1.

# e^* values
separabilities = [1e-3, .01, .03, .05, .1, .15, .2, .25, .3, .35, .4, .45]
# r values
ratios = np.around(np.linspace(-1,1,21), decimals=1)

for s in separabilities:
    # compute c that corresponds to e^*
    c = - scipy.stats.norm.ppf(s)/np.sqrt(scipy.special.hyp2f1(1, a, a+b, -1/eps)/eps)
    for r in ratios:
        # Starting from a large lambda
        lmbda = 3.

        # Instantiate variables for initialization with results from previous step
        eta, gamma, tau = None, None, None

        # Minimum reached, minimizers, values of other metrics at minimizer
        cal_min, cal_mzr, ref_cal_mzr, ce_cal_mzr = 1e6, None, None, None
        ref_min, ref_mzr, cal_ref_mzr, ce_ref_mzr = 1e6, None, None, None
        ce_min, ce_mzr, cal_ce_mzr, ref_ce_mzr = 1e6, None, None, None
        err_ref_mzr, err_ce_mzr = None, None

        # Wether calibration / refinement are still decreasing
        cal_improved, ref_improved = True, True
        while (cal_improved or ref_improved):
            # Solve system
            eta, gamma, tau, mean, var = solve_beta_shifted_system(
                10**lmbda, 10**r, c, a, b, eps,
                eta_init=eta, gamma_init=gamma, tau_init=tau,
                tol=1e-7, lr=min(.7, -1/min(-1e-6, lmbda))
            )
            # Compute resulting errors using Theorem 5.1
            cal, ref, err = compute_population_metrics(mean, var)
            ce = cal+ref

            # Stopping when both minimizers are reached
            if cal < cal_min:
                cal_improved = True
                cal_min = cal
                cal_mzr = lmbda
                ref_cal_mzr = ref
                ce_cal_mzr = ce
            else:
                cal_improved = False

            if ref < ref_min:
                ref_improved = True
                ref_min = ref
                ref_mzr = lmbda
                cal_ref_mzr = cal
                ce_ref_mzr = ce
                err_ref_mzr = err
            else:
                ref_improved = False

            if ce < ce_min:
                ce_min = ce
                ce_mzr = lmbda
                cal_ce_mzr = cal
                ref_ce_mzr = ref
                err_ce_mzr = err

            # Updating lambda
            lmbda -= .01

        # Logging
        log_experiment(
            file_path=logging_path,
            new_data=pd.DataFrame([{
                'sep': s, 'c': np.around(c, decimals=5), 'log_r': r,
                'loss_decrease': np.around((1-ref_min/ce_min)*100, decimals=2),
                'ce_gap':np.around(ce_min-ref_min, decimals=5),
                'err_gap':np.around(err_ce_mzr-err_ref_mzr, decimals=5),
                'cal_mzr':np.around(cal_mzr, decimals=2), 'ref_mzr':np.around(ref_mzr, decimals=2), 'ce_mzr':np.around(ce_mzr, decimals=2),
                'cal_min':np.around(cal_min, decimals=5), 'ref_cal_mzr':np.around(ref_cal_mzr, decimals=5), 'ce_cal_mzr':np.around(ce_cal_mzr, decimals=5),
                'ref_min':np.around(ref_min, decimals=5), 'cal_ref_mzr':np.around(cal_ref_mzr, decimals=5), 'ce_ref_mzr':np.around(ce_ref_mzr, decimals=5),
                'ce_min':np.around(ce_min, decimals=5), 'cal_ce_mzr':np.around(cal_ce_mzr, decimals=5), 'ref_ce_mzr':np.around(ref_ce_mzr, decimals=5),
                'err_ref_mzr':np.around(err_ref_mzr, decimals=5), 'err_ce_mzr':np.around(err_ce_mzr, decimals=5),
            }])
        )