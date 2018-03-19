# Code from scikit-MDR library
# -*- coding: utf-8 -*-

"""
Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import print_function
from collections import Counter
from scipy import stats
import numpy as np
###############################################################################
def entropy(X, base=2):
    """Calculates the entropy, H(X), in the given base

    Parameters
    ----------
    X: array-like (# samples)
        An array of values for which to compute the entropy
    base: integer (default: 2)
        The base in which to calculate entropy

    Returns
    ----------
    entropy: float
        The entropy calculated according to the equation 
        H(X) = -sum(p_x * log p_x) for all states of X

    """
    return stats.entropy(list(Counter(X).values()), base=base)

###############################################################################
def joint_entropy(X, Y, base=2):
    """Calculates the joint entropy, H(X,Y), in the given base

    Parameters
    ----------
    X: array-like (# samples)
        An array of values for which to compute the joint entropy
    Y: array-like (# samples)
        An array of values for which to compute the joint entropy
    base: integer (default: 2)
        The base in which to calculate joint entropy

    Returns
    ----------
    joint_entropy: float
        The joint entropy calculated according to the equation 
        H(X,Y) = -sum(p_xy * log p_xy) for all combined states of X and Y

    """
    X_Y = ['{}{}'.format(x, y) for x, y in zip(X, Y)]
    return entropy(X_Y, base=base)

###############################################################################
def conditional_entropy(X, Y, base=2):
    """Calculates the conditional entropy, H(X|Y), in the given base

    Parameters
    ----------
    X: array-like (# samples)
        An array of values for which to compute the conditional entropy
    Y: array-like (# samples)
        An array of values for which to compute the conditional entropy
    base: integer (default: 2)
        The base in which to calculate conditional entropy

    Returns
    ----------
    conditional_entropy: float
        The conditional entropy calculated according to the equation
        H(X|Y) = H(X,Y) - H(Y)

    """
    return joint_entropy(X, Y, base=base) - entropy(Y, base=base)

###############################################################################
def mutual_information(X, Y, base=2):
    """Calculates the mutual information between two variables, I(X;Y),
    in the given base

    Parameters
    ----------
    X: array-like (# samples)
        An array of values for which to compute the mutual information
    Y: array-like (# samples)
        An array of values for which to compute the mutual information
    base: integer (default: 2)
        The base in which to calculate mutual information

    Returns
    ----------
    mutual_information: float
        The mutual information calculated according to the equation I(X;Y)
        = H(Y) - H(Y|X)

    """
    return entropy(Y, base=base) - conditional_entropy(Y, X, base=base)

###############################################################################
def two_way_information_gain(X, Y, Z, base=2):
    """Calculates the two-way information gain between three variables,
    I(X;Y;Z), in the given base

    IG(X;Y;Z) indicates the information gained about variable Z by the
    joint variable X_Y, after removing the information that X and Y have
    about Z individually. Thus, two-way information gain measures the
    synergistic predictive value of variables X and Y about variable Z.

    Parameters
    ----------
    X: array-like (# samples)
        An array of values for which to compute the 2-way information gain
    Y: array-like (# samples)
        An array of values for which to compute the 2-way information gain
    Z: array-like (# samples)
        An array of outcome values for which to compute the 2-way
        information gain
    base: integer (default: 2)
        The base in which to calculate 2-way information

    Returns
    ----------
    mutual_information: float
        The information gain calculated according to the equation
        IG(X;Y;Z) = I(X,Y;Z) - I(X;Z) - I(Y;Z)

    """
    X_Y = ['{}{}'.format(x, y) for x, y in zip(X, Y)]
    return (mutual_information(X_Y, Z, base=base) -
            mutual_information(X, Z, base=base) -
            mutual_information(Y, Z, base=base))

###############################################################################
def three_way_information_gain(W, X, Y, Z, base=2):
    """Calculates the three-way information gain between three variables,
    I(W;X;Y;Z), in the given base

    IG(W;X;Y;Z) indicates the information gained about variable Z by
    the joint variable W_X_Y, after removing the information that W, X,
    and Y have about Z individually and jointly in pairs. Thus, 3-way
    information gain
    measures the synergistic predictive value of variables W, X, and Y
    about variable Z.

    Parameters
    ----------
    W: array-like (# samples)
        An array of values for which to compute the 3-way information gain
    X: array-like (# samples)
        An array of values for which to compute the 3-way information gain
    Y: array-like (# samples)
        An array of values for which to compute the 3-way information gain
    Z: array-like (# samples)
        An array of outcome values for which to compute the 3-way
        information gain
    base: integer (default: 2)
        The base in which to calculate 3-way information

    Returns
    ----------
    mutual_information: float
        The information gain calculated according to the equation:
            IG(W;X;Y;Z) = I(W,X,Y;Z) - IG(W;X;Z) - IG(W;Y;Z) - IG(X;Y;Z)
            - I(W;Z) - I(X;Z) - I(Y;Z)

    """
    W_X_Y = ['{}{}{}'.format(w, x, y) for w, x, y in zip(W, X, Y)]
    return (mutual_information(W_X_Y, Z, base=base) -
            two_way_information_gain(W, X, Z, base=base) -
            two_way_information_gain(W, Y, Z, base=base) -
            two_way_information_gain(X, Y, Z, base=base) -
            mutual_information(W, Z, base=base) -
            mutual_information(X, Z, base=base) -
            mutual_information(Y, Z, base=base))
