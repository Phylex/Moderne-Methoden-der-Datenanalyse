#!/usr/bin/python3
"""This example shows that the chi2-distance of a sample (that is the sum of all the
chi_2 distances of the individual measurements in a sample) when computed for many
samples of the same experimental (or in the case of this example simulated) setup
follows the distribution given by the `chi_squared_distribution`.

.. author: Alexander Becker <nabla.becker@mailbox.org>
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def chi_squared_distribution(x, n):
    """the chi squared distribution"""
    return 1/(2**(n/2)*gamma(n/2)) * np.exp(-x/2) * x**(n/2-1)

def neg_exp(domain, theta):
    """a negative exponential function"""
    return 1/theta * np.exp(-domain/theta)

def measurements_following_curve_with_gauss_error(sigma, x_vals, curve, *params):
    """this function generates m_count measurements whos true value follows a function
    given by curve(x, *params). The measurements are samples from a gaussian distribution
    whos mean is the true value of the curve at the x coordinate of the sample and
    whos standard deviation is `sigma`, one sample is drawn per value in `x_values`"""
    deviations = np.random.randn(len(x_vals))*sigma
    true_values = [curve(x, *params) for x in x_vals]
    return true_values + deviations

def chi2_distance(distances, sigma):
    """calculates the chi2 value of a sample given the distance of the measurements
    to the (assumed) true value, given all values are sampled from a gaussian of
    constant width for all measurements"""
    return sum(distances**2/sigma**2)

if __name__ == "__main__":
    #here the goal is to show that the distribution of the total distance of all
    #measurements in a sample follows the chi squared distribution.

    #Set the parameters of the measurement including the function of the true value
    SIGMA = 1
    THETA = 3
    SAMPLE_COUNT = 10000
    CURVE = neg_exp
    MEASUREMENTS_PER_SAMPLE = 10
    UPPER_BOUND = 10
    LOWER_BOUND = 1
    #generate the x values for the measurements and the true y values corresponding to
    #the generated x values.
    X = np.linspace(LOWER_BOUND, UPPER_BOUND, MEASUREMENTS_PER_SAMPLE, endpoint=True)
    TRUE_Y = CURVE(X, THETA)

    #Create N samples of len(X) measurements
    samples = [measurements_following_curve_with_gauss_error(SIGMA, X, CURVE, THETA)
               for i in range(SAMPLE_COUNT)]
    #compute the distance of every measurement to its true value
    distances = [sample - TRUE_Y for sample in samples]
    #compute the chi2 value for all samples
    sample_chi_2 = [ chi2_distance(sample_distances, SIGMA) for sample_distances in distances ]

    #set the parameters for the plot
    BIN_COUNT = int(SAMPLE_COUNT/50)
    #plot histogram of distances overlaid with the assumed distriburion
    hist, bins, patches = plt.hist(sample_chi_2, bins=BIN_COUNT,
                                   label="Total distance of sample to true curve",
                                   color='lightsteelblue', density=True)
    PLOT_X = np.linspace(min(bins), max(bins), 1000)
    plt.plot(PLOT_X, chi_squared_distribution(PLOT_X, MEASUREMENTS_PER_SAMPLE), color='darkblue',
             label=r'Distribution of total distance of sample with $n=${}'
             .format(MEASUREMENTS_PER_SAMPLE-1))
    plt.grid(linestyle=':')
    plt.legend()
    plt.show()
