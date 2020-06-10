#!/usr/bin/python3
# coding:utf-8
'''Script that shows the connection between prior assumption, likelihood and
posterior. A series of coin tosses is simulated. A series of priors are assumed
and altered by the continuous tossing of the coin. So after each toss the
prior of the next toss is identified with the posterior of the last coin toss
As this example should show that the posterior of a large amount of tosses is
practiacally independent of the initial prior

.. author: Alexander Becker <nabla.becker@mailbox.org>
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
import statmeth as stm
import statplot as stp

def uniform_prior(x, p):
    """a uniform function as a prior"""
    return 1.

def gaussian_prior(x, mu, sigma):
    """a gaussian function as a prior"""
    return np.exp(-(x-mu)**2/sigma**2)/(np.sqrt(2*np.pi)*sigma)

def exp_prior(x, norm, tau):
    """a exponential function as a prior"""
    return 1/norm * np.exp(-x/tau)

def neg_exp_prior(x, norm, tau):
    """a mirrored exponential function as a prior"""
    return 1/norm * np.exp(-(1.-x)/tau)

def arbitrary_average_coin_tosses(average, count):
    return np.around(np.random.rand(count)/2 + (average/2.))

def binomial_coeff(n, k):
    """calculate the binomial coefficient for k out of n"""
    return factorial(n)/(factorial(k)*factorial(n-k))

def binomial_distribution(p, k, n):
    """calculate the binomial distribution"""
    return binomial_coeff(n, k) * p**k * (1-p)**(n-k)

def calc_posterior(prior, likelihood, theta_range, likelihood_params):
    """ function calculates the posterior after another coin toss """
    posterior_func = lambda theta: prior * likelihood(theta, *likelihood_params)
    integral = stm.numerical_integrate_1D(posterior_func, theta_range)
    return posterior_func(theta_range)/integral


if __name__ == "__main__":
    # toss the coin 1024 times
    tosses = arbitrary_average_coin_tosses(0.25, 1024)

    # in this case theta is the parameter p of the binomial distribution
    # so theta range can only go from 0 to 1
    x = np.linspace(0, 1, 2000)
    plot_after_tosses = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    likelihood = binomial_distribution
    palette = 'blue-green'
    priors = [np.array([1. for i in x]), exp_prior(x, 1, .1), neg_exp_prior(x, 1, .1), gaussian_prior(x, .5, .2)]
    prior_labels = ["Uniform", "Exponential", "Negative Exponential", "Gaussian"]
    # calculate the posterior iteratively
    for i, toss in enumerate(tosses):
        # calculate the prior after each toss to re-input it after each toss.
        priors = [calc_posterior(prior, likelihood, x, (toss, 1)) for prior in priors]
        # renormalize the priors
        priors = [prior/max(prior) for prior in priors]
        # if it is time to plot, plot the current probability densities.
        if i+1 in plot_after_tosses:
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111)
            for j, prior in enumerate(priors):
                ax.plot(x, prior, color=stp.PALETTS[palette][j], label=prior_labels[j])
            ax.grid()
            ax.legend()
            ax.set_ylabel("renormalized posterior")
            ax.set_xlabel(r"$\bar{ p }$ Expectation value of coin toss")
            ax.set_title("Posterior after {} tosses, {} of which Head".format(i+1, sum(tosses[:i])))
            fig.savefig("posterior_toss_"+str(i+1)+".svg")
            plt.show()
