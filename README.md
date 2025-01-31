# RefineThenCalibrate-Theory

Code to run the high-dimensional asymptotic simulations for regularized logistic regression in the paper "Rethinking Early Stopping: Refine, Then Calibrate".

## Files
- `utils.py`: Contains our solver for the non-linear system describing the weight vector of regularized-logistic regression, for our mathematical model of the spectral distribution, and functions to compute the resulting calibration and refinement errors.
- `LR_expe.py`: Compute theoretical and empirical training curve for a given set of problem parameters, plot the results as in Figure 6.
- `LR_heatmap.py`: Compute theoretical minimizers and loss decrease for a given spectral distribution, on a grid of ratios r and optimal error rate e^*, produces .csv results files.
- `figures.ipynb`: Generate heatmap Figure 7 for the paper, using csv results files.

