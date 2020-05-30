#!/usr/bin/python3
# coding=utf-8
r'''
.. author: Alexander Becker <nabla.becker@mailbox.org

script to calculate an example Likelihood fit to an exponential decay
and plot the results. Also produces a plot of the likelihood scan for
theta. The exponential decay function is specified as
.. math::
    f(t,\theta) = \frac{1}{\theta} e^{\frac{-x}{\theta}}

and has the parameter :math:`\theta`.
'''
import numpy as np
import matplotlib.pyplot as plt

# define own functions
def exp_decay(x, theta):
    r"""calculates the y coordinate of an exponential decay
    function with parameter :math:`\theta`"""
    return 1/theta * np.exp(-x/theta)

def transform_to_exp(x, tau):
    """transforms a uniform distribution of numbers to an exponential one"""
    return -tau*np.log(tau*x)

def neg_log_likelyhood(x, pdf, *args):
    """computes the log likelyhood for a dataset, given a distribution pdf(x, args)"""
    return sum(-np.log(pdf(x, *args)))

def find_nearest(array, value):
    """find the value that is nearest to a given value in an array"""
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx

if __name__ == "__main__":
    # plot the hypothesis functions
    THETA = [0.5, 1.0, 1.5]
    COLOR = ['darkblue', 'blue', 'cyan']
    LINESTYLE = ['--', '-', '-.']
    x_points = np.linspace(0,10,1001, endpoint=True)
    for theta, color, linestyle in zip(THETA, COLOR, LINESTYLE):
        plt.plot(x_points, exp_decay(x_points, theta), color=color, linestyle=linestyle,
                 label=r'exp(x, $\theta$={})'.format(theta))

    #generate the random samples and plot the histogram
    N = 50
    TAU = 1
    BINS = 1000
    RANDOM_SAMPLE = transform_to_exp(np.random.rand(N), TAU)
    plt.hist(RANDOM_SAMPLE, BINS, color='lightblue', label='Random Sample')

    #make the plot nice and readable
    plt.xlabel('t')
    plt.ylabel('Probability density: p(x)')
    plt.legend()
    plt.savefig('exponential_decay.svg')
    plt.show()

    #create the diagram that shows the distribution of the maximum likelihood estimation
    #given the parameter \hat{\theta}_{ML}
    RANDOM_SAMPLES = np.array([neg_log_likelyhood(transform_to_exp(np.random.rand(N), TAU),
                                                  exp_decay, TAU)
                               for i in range(10000)])
    plt.hist(RANDOM_SAMPLES, label='10000 MC-Pseudo-Experiments', bins=50, alpha=0.5, density=True)
    plt.grid(linestyle=':')
    plt.savefig('nll_distribution.svg')
    plt.show()


    #now we do the likelihood scan
    THETA_MIN, THETA_MAX = 0.8, 1.4
    THETA_RANGE = np.linspace(THETA_MIN, THETA_MAX, 50, endpoint=True)
    NEG_LOG_L = np.array([neg_log_likelyhood(RANDOM_SAMPLE, exp_decay, theta) for theta in THETA_RANGE])
    MIN_NLL = min(NEG_LOG_L)
    MIN_NLL_IDX = NEG_LOG_L.argmin()
    #split the array into the part left of the min and right of the min
    #so that it is possible to find the closest array vals to +/- sigma
    LEFT_SIDE = NEG_LOG_L[:MIN_NLL_IDX]
    RIGHT_SIDE = NEG_LOG_L[MIN_NLL_IDX:]
    LEFT_SIGMA, LSIDX= find_nearest(LEFT_SIDE, MIN_NLL+(1./2.))
    RIGHT_SIGMA, RSIDX = find_nearest(RIGHT_SIDE, MIN_NLL+(1./2.))
    RSIDX = RSIDX+len(LEFT_SIDE)

    #plot likelihood scan
    plt.plot(THETA_RANGE, NEG_LOG_L, color='darkblue', marker='o', label='Likelihood Scan')
    #plot the lines indicating the differnt sections
    xmin, xmax, ymin, ymax = plt.axis()
    #horizontal lines for the minimum and the min+1/2 lines
    plt.hlines(y=MIN_NLL, xmin=xmin, xmax=xmax, color='blue', label='NLL minimum')
    plt.hlines(y=MIN_NLL+(1./2.), xmin=xmin, xmax=xmax, color='lightblue', linestyle='--',
               label='standard deviation')
    #vertical lines for the center and the left and right standard deviations
    plt.vlines(x=THETA_RANGE[LSIDX], ymin=NEG_LOG_L[LSIDX], ymax=ymax, color='cyan',
               label=r'$\hat{\theta} \pm \sigma$')
    plt.vlines(x=THETA_RANGE[RSIDX], ymax=ymax, ymin=NEG_LOG_L[RSIDX], color='cyan')
    plt.vlines(x=THETA_RANGE[MIN_NLL_IDX], ymin=MIN_NLL, ymax=ymax, color='gray',
               label=r'$\hat{\theta}_{ML} = $'+str(np.round(THETA_RANGE[MIN_NLL_IDX], 3)))
    #grid and legend
    plt.grid(linestyle=':')
    plt.legend()
    plt.savefig('likelihood_scan.svg')
    plt.show()
