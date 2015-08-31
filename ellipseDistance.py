import numpy as np
from scipy.optimize import bisect

# Wrapper around scipy's bisect method to ignore
# incorrect search intervals caused by numerical errors.
# Assumes f is a decreasing function.
def robustDecreasingBisect(f, tMin, tMax):
    if tMax <= tMin:
        return 0.5 * (tMax + tMin)
    lastValue = f(tMax)
    if lastValue >= 0:
        return lastValue
    return bisect(f, tMin, tMax)
    firstValue = f(tMin)
    if firstValue <= 0:
        return firstValue

# Returns a unit vector r such that |Mr - c| is minimized.
# M must have shape (n, 2)
# c must have shape (2,)
# Based on:
# Distance from a Point to an Ellipse, an Ellipsoid, or a Hyperellipsoid
# David Eberly
# http://www.geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf
def closestEllipsePoint(M, c):
    U, e, V = np.linalg.svd(M, full_matrices = False)
    y = np.dot(c, U)
    eSquared = np.square(e)

    def f(t):
        denom = t + eSquared
        if np.any(denom == 0):
            return np.inf
        x = y * e / denom
        return np.sum(np.square(x)) - 1

    if y[1] == 0:
        if y[0] == 0: # This extra origin case is only necessary for circles
            x = np.zeros(2)
            x[1] = 1
        elif abs(y[0]) * e[0] >= -eSquared[1] + eSquared[0]:
            x = np.sign(y)
        else:
            x = np.empty(2)
            x[0] = y[0] * e[0] / (-eSquared[1] + eSquared[0])
            x[1] = np.sqrt(1 - np.square(x[0]))
    else:
        if y[0] == 0:
            x = np.sign(y)
        else:
            tMin = np.abs(e[1] * y[1]) - eSquared[1]
            tMax = np.linalg.norm(e * y) - eSquared[1]
            t = robustDecreasingBisect(f, tMin, tMax)
            x = y * e / (t + eSquared)

    # Because x is multiplied on the left and V is orthonormal,
    # this multiplies x by the inverse of V.
    r = np.dot(x, V)
    r /= np.linalg.norm(r)
    return r
