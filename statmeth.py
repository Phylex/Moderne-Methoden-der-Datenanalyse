# coding:utf-8
"""
.. module:: statmeth
    :platform: Unix
    :synopsis: This module provides often used procedures for generating random data
        and other often used functions
.. moduleauthor:: Alexander Becker <nabla.becker@mailbox.org>
"""
import numpy as np

def generate_pdf_rnums_rejection_method(num, pdf, bounds, *args):
    """generate random numbers following the pdf given for
    pdf(*args) using the rejection method and transform the area according tho the pdf"""
    count = 0
    rgen = 0
    transformed_rnums = []
    delta = bounds[1]-bounds[0]
    maxval = max(pdf(np.linspace(bounds[0], bounds[1], int(delta*1000), endpoint=True), *args))
    while count < num:
        coordinate = np.random.rand(2) * np.array([delta, maxval+.1]) + bounds[0]
        rgen += 2
        if coordinate[1] < pdf(coordinate[0], *args):
            transformed_rnums.append(coordinate[0])
            count += 1
    print(rgen)
    return transformed_rnums


def transform_to_exp_decay(urnums, theta):
    """transforms uniformally distributed random numbers to random numbers
    with a distribution following an exponential decay"""
    return -theta * np.log(theta*urnums)
