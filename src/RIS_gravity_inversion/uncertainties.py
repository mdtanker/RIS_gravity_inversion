import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from scipy.special import erf
from scipy.stats import pearsonr


"""

The below functions are adapted from the GitHub repository
"https://github.com/charlesrouge/SampleVis"

"""


def scale_normalized(sample, bounds):
    """
    Rescales the sample space into the unit hypercube, bounds = [0,1]
    """
    scaled_sample = np.zeros(sample.shape)

    for j in range(sample.shape[1]):
        scaled_sample[:, j] = (sample[:, j] - bounds[j][0]) / (
            bounds[j][1] - bounds[j][0]
        )

    return scaled_sample


# Rescales a sample defined in the unit hypercube, to its bounds
def scale_to_bounds(scaled_sample, bounds):
    sample = np.zeros(scaled_sample.shape)

    for j in range(sample.shape[1]):
        sample[:, j] = (
            scaled_sample[:, j] * (bounds[j][1] - bounds[j][0]) + bounds[j][0]
        )

    return sample


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


def correlation_plots(z_test, p_val, test_name, var_names):
    """
    Get and plot results for within-sample correlation test, based on
    1) test results z_test (Figure 1)
    2) statistical significance pval (Figure 2)
    other inputs are the test_name and var_names
    """
    # Local variables
    nvar = len(var_names)
    pval = 1 - (
        p_val + np.matlib.eye(nvar)
    )  # Transformation convenient for plotting below

    ###################################################
    # Figure 1: correlations

    # Matrix to plot
    res_mat = np.zeros((nvar, nvar + 1))
    res_mat[:, 0:-1] = z_test

    # Center the color scale on 0
    res_mat[0, nvar] = max(np.amax(z_test), -np.amin(z_test))
    res_mat[1, nvar] = -res_mat[0, nvar]

    # Plotting Pearson test results
    plt.imshow(res_mat, extent=[0, nvar + 1, 0, nvar], cmap=plt.cm.bwr)

    # Plot specifications
    ax = plt.gca()
    ax.set_xlim(0, nvar)  # Last column only to register min and max values for colorbar
    ax.set_xticks(np.linspace(0.5, nvar - 0.5, num=nvar))
    ax.set_xticklabels(var_names, rotation=20, ha="right")
    ax.set_yticks(np.linspace(0.5, nvar - 0.5, num=nvar))
    ax.set_yticklabels(var_names[::-1])
    ax.tick_params(axis="x", top=True, bottom=True, labelbottom=True, labeltop=False)
    ax.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=False)
    plt.title("Rank correlation between variables' sampled values", size=13, y=1.07)
    plt.colorbar()
    plt.show()
    # plt.savefig(test_name + '_cross_correlation.png')
    # plt.clf()

    ###################################################
    # Figure 2: correlations

    # Matrix to plot
    res_mat = np.zeros((nvar, nvar + 1))

    # Set the thresholds at +-95%, 99%, and 99.9% significance levels
    bin_thresholds = [0.9, 0.95, 0.99, 0.999]
    n_sig = len(bin_thresholds)
    res_mat[:, 0:-1] = binning(pval, bin_thresholds)

    # Set the color scale
    res_mat[0, nvar] = n_sig

    # Common color map
    cmap = plt.cm.Greys
    cmaplist = [cmap(0)]
    for i in range(n_sig):
        cmaplist.append(cmap(int(255 * (i + 1) / n_sig)))
    mycmap = cmap.from_list("Custom cmap", cmaplist, n_sig + 1)

    # Plot background mesh
    mesh_points = np.linspace(0.5, nvar - 0.5, num=nvar)
    for i in range(nvar):
        plt.plot(
            np.arange(0, nvar + 1),
            mesh_points[i] * np.ones(nvar + 1),
            c="k",
            linewidth=0.3,
            linestyle=":",
        )
        plt.plot(
            mesh_points[i] * np.ones(nvar + 1),
            np.arange(0, nvar + 1),
            c="k",
            linewidth=0.3,
            linestyle=":",
        )

    # Plotting MK test results
    plt.imshow(res_mat, extent=[0, nvar + 1, 0, nvar], cmap=mycmap)

    # Plot specifications
    ax = plt.gca()
    ax.set_xlim(0, nvar)  # Last column only to register min and max values for colorbar
    ax.set_xticks(mesh_points)
    ax.set_xticklabels(var_names, rotation=20, ha="right")
    ax.set_yticks(mesh_points)
    ax.set_yticklabels(var_names[::-1])
    ax.tick_params(axis="x", top=True, bottom=True, labelbottom=True, labeltop=False)
    ax.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=False)
    plt.title("Significance of the rank correlations", size=13, y=1.07)
    colorbar = plt.colorbar()
    colorbar.set_ticks(
        np.linspace(res_mat[0, nvar] / 10, 9 * res_mat[0, nvar] / 10, num=n_sig + 1)
    )
    cb_labels = ["None"]
    for i in range(n_sig):
        cb_labels.append(str(bin_thresholds[i] * 100) + "%")
    colorbar.set_ticklabels(cb_labels)
    plt.show()
    # plt.savefig(test_name + '_significance.png')
    # plt.clf()

    return None


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
        sm[i, i:n] = np.sign(x[i + 1 : n] - x[0 : n - 1 - i])  # noqa E203

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


def projection_2D(sample, var_names):
    """
    Plots the sample projected on each 2D plane
    """
    dim = sample.shape[1]

    for i in range(dim):
        for j in range(dim):
            plt.subplot(dim, dim, i * dim + j + 1)
            plt.scatter(
                sample[:, j],
                sample[:, i],
                s=2,
            )
            if j == 0:
                plt.ylabel(var_names[i], rotation=0, ha="right")
            if i == dim - 1:
                plt.xlabel(var_names[j], rotation=20, ha="right")

            plt.xticks([])
            plt.yticks([])
    plt.show()
    # plt.savefig('2D-projections.png')
    # plt.clf()

    return None


def space_filling_measures_discrepancy(sample):
    """
    Assumes the sample has N points (lines) in p dimensions (columns)
    Assumes sample drawn from unit hypercube of dimension p
    L2-star discrepancy formula from

    """

    [n, p] = sample.shape

    # First term of the L2-star discrepancy formula
    dl2 = 1.0 / (3.0**p)

    # Second term of the L2-star discrepancy formula
    sum_1 = 0.0
    for k in range(n):
        for i in range(n):
            prod = 1.0
            for j in range(p):
                prod = prod * (1 - max(sample[k, j], sample[i, j]))
            sum_1 = sum_1 + prod
    dl2 += sum_1 / n**2

    # Third term of the L2-star discrepancy formula
    sum_2 = 0.0
    for i in range(n):
        prod = 1
        for j in range(p):
            prod *= 1 - sample[i, j] ** 2
        sum_2 = sum_2 + prod
    dl2 -= sum_2 * (2 ** (1 - p)) / n

    return dl2


def space_filling_measures_min_distance(sample, show):
    """
    Returns the minimal distance between two points
    Assumes sample drawn from unit hypercube of dimension p
    """
    n = sample.shape[0]
    dist = np.ones(
        (n, n)
    )  # ones and not zeros because we are interested in abstracting min distance

    # Finding distance between points
    for i in range(n):
        for j in np.arange(i + 1, n):
            dist[i, j] = np.sqrt(np.sum((sample[i, :] - sample[j, :]) ** 2))
            dist[j, i] = dist[i, j]  # For the plot

    # If wanted: plots the distribution of minimal distances from all points
    if show == 1:
        plt.plot(np.arange(1, n + 1), np.sort(np.amin(dist, axis=1)))
        plt.xlim(1, n)
        plt.xlabel("Sorted ensemble members")
        plt.ylabel("Min Euclidean distance to any other point")
        plt.show()
        # plt.savefig('Distances.png')
        # plt.clf()

    # Space-filling index is minimal distance between any two points
    return np.amin(dist)
