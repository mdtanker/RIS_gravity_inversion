from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from scipy.special import erf
from scipy.stats import pearsonr

"""

The below functions are adapted from the GitHub repository
"https://github.com/charlesrouge/SampleVis"

"""


def binning(tab, vect):
    """
    Discretizes value from tab depending on how they fall on a scale defined by vec
    Returns binned_tab, with the same shape as tab
    Example if vec = [0,1], binned_tab[i,j]=0 if tab[i,j]<=0, =1 if 0<tab[i,j]<=1,
    =2 otherwise
    """
    binned_tab = np.zeros(tab.shape)

    for i in range(len(vect)):
        binned_tab = binned_tab + 1 * (tab > vect[i] * np.ones(tab.shape))

    return binned_tab


def pearson_test_sample(sample):
    """
    Correlation Pearson test for whole sample. Outputs are:
    the Pearson statistic rho
    the p-value pval
    """
    # Local variables
    var = sample.shape[1]
    rho = np.zeros((var, var))
    pval = np.zeros((var, var))

    # Pearson test results
    for i in range(var):
        for v in np.arange(i + 1, var):
            [rho[i, v], pval[i, v]] = pearsonr(sample[:, i], sample[:, v])
            [rho[v, i], pval[v, i]] = [rho[i, v], pval[i, v]]

    return [rho, pval]


def mann_kendall_test(y, prec):
    """
    Mann-Kendall test (precision is the number of decimals)
    Outputs are the normalized statistic Z and the associated p-value
    """
    n = len(y)
    x = np.int_(y * (10**prec))

    # Sign matrix and ties
    sm = np.zeros((n - 1, n - 1))
    for i in range(n - 1):
        sm[i, i:n] = np.sign(x[i + 1 : n] - x[0 : n - 1 - i])  # E203

    # Compute MK statistic
    s = np.sum(sm)

    # Count ties and their c
    # appel Mimiontributions to variance of the MK statistic
    [val, count] = np.unique(x, return_counts=True)
    [extent, ties] = np.unique(count, return_counts=True)
    tie_contribution = np.zeros(len(ties))
    for i in range(len(ties)):
        tie_contribution[i] = (
            ties[i] * extent[i] * (extent[i] - 1) * (2 * extent[i] + 5)
        )

    # Compute the variance
    vs = (n * (n - 1) * (2 * n + 5) - np.sum(tie_contribution)) / 18
    if vs < 0:
        print("WARNING: negative variance!!!")

    # Compute standard normal statistic
    z = (s - np.sign(s)) / np.sqrt(max(vs, 1))

    # Associated p-value
    pval = 1 - erf(abs(z) / np.sqrt(2))

    return [z, pval]


def mann_kendall_test_sample(sample):
    """
    Same as above, but for whole sample
    Outputs are the normalized statistic Z and the associated p-value
    """
    # Local variables
    n = sample.shape[0]
    var = sample.shape[1]
    x = np.argsort(
        sample, axis=0
    )  # Ranks of the values in the ensemble, for each variable
    mk_res = np.zeros((var, var))
    pval = np.zeros((var, var))

    # MK test results
    for i in range(var):
        reorder_sample = np.zeros((n, var))
        for j in range(n):
            reorder_sample[j, :] = sample[x[j, i], :]
        for v in np.arange(i + 1, var):
            [mk_res[i, v], pval[i, v]] = mann_kendall_test(reorder_sample[:, v], 5)
            [mk_res[v, i], pval[v, i]] = [mk_res[i, v], pval[i, v]]

    return [mk_res, pval]


def projection_1D(sample, var_names):
    """
    Assess the uniformity of each 1D projection of the sample
    Assumes bounds of sample are [0,1]**n
    """
    [n, dim] = sample.shape
    y = np.zeros(sample.shape)

    z_int = np.linspace(0, 1, num=n + 1)
    binned_sample = binning(sample, z_int)

    for i in range(n):
        y[i, :] = 1 * (np.sum(1 * (binned_sample == i + 1), axis=0) > 0)

    proj = np.sum(y, axis=0) / n

    plt.bar(np.arange(dim), proj)
    plt.ylim(0, max(1, 1.01 * np.amax(proj)))
    plt.xticks(np.arange(dim), var_names)
    plt.ylabel("Coverage of axis")
    plt.show()
    # plt.savefig('1D_coverage_index.png')
    # plt.clf()

    # Return a single index: the average of values for all the variables
    return np.mean(proj)
