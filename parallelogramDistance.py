import numpy as np

# Given two vectors u and v and a point p,
# calculates coefficients a and b that minimize norm(a*u + b*v - p)
# with the restriction that abs(a) + abs(b) = 1
def closestParallelogramPoint(u, v, p):
    phase, distance = getClosestLineSegmentPhaseAndSquareDistance(u, v, p)
    a = 1 - phase
    b = phase
    
    phase2, distance2 = getClosestLineSegmentPhaseAndSquareDistance(u, -v, p)
    if distance2 < distance:
        a = 1 - phase2
        b = -phase2
        distance = distance2
    
    phase3, distance3 = getClosestLineSegmentPhaseAndSquareDistance(-u, v, p)
    if distance3 < distance:
        a = phase3 - 1
        b = phase3
        distance = distance3
    
    phase4, distance4 = getClosestLineSegmentPhaseAndSquareDistance(-u, -v, p)
    if distance4 < distance:
        a = phase4 - 1
        b = -phase4
        distance = distance4

    return a, b
    
# Returns 0 if u is closest to p, 1 if v is closest, or some phase in between.
# Also returns square of corresponding distance.
def getClosestLineSegmentPhaseAndSquareDistance(u, v, p):
    squareLength = np.sum(np.square(v - u))
    if squareLength == 0:
        phase = 0.5
    else:
        phase = np.clip(np.dot(p - u, v - u) / squareLength, 0, 1)
    squareDistance = np.sum(np.square(u + (v - u) * phase - p))
    return phase, squareDistance
