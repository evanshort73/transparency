import numpy as np
from scipy.misc import imread, imsave
#from matplotlib import pyplot as plt
from ellipseDistance import closestEllipsePoint
from checkerboard import checkerboard
from imageNormalization import imageNormalize

# Setup

def loadNormalizedImage(filename):
    im = imread(filename).astype(float) / 255
    imMean = np.mean(im, axis = (0, 1))
    im -= imMean
    return im, imMean

def initRgbScale(im1, im2):
    rgbScale = np.vstack((np.var(im1, axis = (0, 1)),
                          np.var(im2, axis = (0, 1))))
    rgbScale /= np.sum(rgbScale, axis = 0)
    np.sqrt(rgbScale, out = rgbScale)
    return rgbScale

def initHubAxis(mean1, mean2):
    meanDiff = mean2 - mean1
    hubAxisDiff = np.linalg.norm(meanDiff)
    hubAxis = meanDiff / hubAxisDiff
    opaqueDiff = -hubAxisDiff
    return hubAxis, opaqueDiff

def initOuterMatrix(im1, im2):
    originalRgbScale = initRgbScale(im1, im2)
    scale1 = 1 / originalRgbScale[0]
    scale2 = 1 / originalRgbScale[1]
    xxT = scale1[:, None] * np.einsum("ijk,ijl", im1, im1) * scale1
    xyT = scale1[:, None] * np.einsum("ijk,ijl", im1, im2) * scale2
    yyT = scale2[:, None] * np.einsum("ijk,ijl", im2, im2) * scale2
    outerMatrix = np.vstack((np.hstack((xxT, -xyT)),
                             np.hstack((-xyT.T, yyT))))
    return outerMatrix

def getScaledImages(rgbScale, im1, im2):
    originalRgbScale = initRgbScale(im1, im2)
    overallRgbScale = rgbScale / originalRgbScale
    return im1 * overallRgbScale[0], im2 * overallRgbScale[1]

def getAdjustedImages(rgbScale, hubAxis, opaqueDiff, im1, im2):
    adjustedIm1, adjustedIm2 = getScaledImages(rgbScale, im1, im2)
    adjustedIm1 += hubAxis * 0.5 * opaqueDiff
    adjustedIm2 -= hubAxis * 0.5 * opaqueDiff
    return adjustedIm1, adjustedIm2

def saveComparison(filename, rgbScale, hubAxis, opaqueDiff, im1, im2):
    adjustedIm1, adjustedIm2 = getAdjustedImages(rgbScale, hubAxis,
                                                 opaqueDiff, im1, im2)
    result = imageNormalize(checkerboard(adjustedIm1, adjustedIm2))
    imsave(filename, np.round(result * 255).astype(np.uint8))

im1, mean1 = loadNormalizedImage("in1.png")
im2, mean2 = loadNormalizedImage("in2.png")
colorSpaceCenter = -((mean1 + mean2) * 0.5 - 0.5)

rgbScale = initRgbScale(im1, im2)

# opaqueDiff is the typical value of [(im2 - im1) dot hubAxis] in opaque areas
hubAxis, opaqueDiff = initHubAxis(mean1, mean2)

outerMatrix = initOuterMatrix(im1, im2)

saveComparison("beforeColorAdjustment.png", rgbScale, hubAxis, opaqueDiff,
               im1, im2)

# Optimize rgbScale and hubAxis

def getObjectiveMatrix(outerMatrix, hubAxis):
    rainMatrix = np.diag(np.diag(outerMatrix)) \
                 + np.diag(np.diag(outerMatrix, k = -3), k = -3) \
                 + np.diag(np.diag(outerMatrix, k = 3), k = 3)
    axisTwice = np.tile(hubAxis, 2)
    objectiveMatrix = rainMatrix \
                      - axisTwice[:, None] * outerMatrix * axisTwice
    return objectiveMatrix

def scaleUpdate(i, rgbScale, hubAxis, outerMatrix):
    objectiveMatrix = getObjectiveMatrix(outerMatrix, hubAxis)
    sqrtMatrix = np.linalg.cholesky(objectiveMatrix)
    
    ellipseMatrix = sqrtMatrix[i::3].T
    goalPoint = np.dot(ellipseMatrix, rgbScale[:, i]) \
                - np.dot(np.ravel(rgbScale), sqrtMatrix)
    scale = closestEllipsePoint(ellipseMatrix, goalPoint)
    return scale

def hubAxisUpdate(rgbScale, outerMatrix):
    rgbScaleDiag = np.vstack((np.diag(rgbScale[0]), np.diag(rgbScale[1])))
    eigMatrix = np.dot(rgbScaleDiag.T, np.dot(outerMatrix, rgbScaleDiag))
    eigVals, eigVecs = np.linalg.eig(eigMatrix)
    
    eigIndex = np.argmax(eigVals)
    hubAxis = eigVecs[:, eigIndex]
    return hubAxis

def objectiveUpdate(rgbScale, hubAxis, outerMatrix):
    objectiveMatrix = getObjectiveMatrix(outerMatrix, hubAxis)
    objective = np.dot(np.ravel(rgbScale),
                       np.dot(objectiveMatrix, np.ravel(rgbScale)))
    return objective

oldObjective = np.inf
tolerance = 0.0000000001
objectiveHistory = []
for i in xrange(50):
    hubAxis = hubAxisUpdate(rgbScale, outerMatrix)
    objective = objectiveUpdate(rgbScale, hubAxis, outerMatrix)
    assert objective < oldObjective + tolerance
    objectiveHistory.append(objective)
    oldObjective = objective

    rgbScale[:, 0] = scaleUpdate(0, rgbScale, hubAxis, outerMatrix)
    objective = objectiveUpdate(rgbScale, hubAxis, outerMatrix)
    assert objective < oldObjective + tolerance
    objectiveHistory.append(objective)
    oldObjective = objective
    
    rgbScale[:, 1] = scaleUpdate(1, rgbScale, hubAxis, outerMatrix)
    objective = objectiveUpdate(rgbScale, hubAxis, outerMatrix)
    assert objective < oldObjective + tolerance
    objectiveHistory.append(objective)
    oldObjective = objective
    
    rgbScale[:, 2] = scaleUpdate(2, rgbScale, hubAxis, outerMatrix)
    objective = objectiveUpdate(rgbScale, hubAxis, outerMatrix)
    assert objective < oldObjective + tolerance
    objectiveHistory.append(objective)
    oldObjective = objective

#plt.plot(objectiveHistory)
#plt.show()

print "RGB Scale:", rgbScale
print "Hub Axis:", hubAxis
print "Flat Error:", objective

# Optimize opaqueDiff and transparentDiff

def getTransparentDiffSign(rgbScale, hubAxis, im1, im2):
    scaledIm1, scaledIm2 = getScaledImages(rgbScale, im1, im2)
    bgDiff = np.dot(scaledIm2 - scaledIm1, hubAxis) # Mean is already 0
    imMean = scaledIm1 + scaledIm2
    imMean *= 0.5
    # TODO: Do we really need to flatten the mean before computing its length?
    flatImMean = imMean - hubAxis * np.dot(imMean, hubAxis)[:, :, None]
    hubDistance = np.linalg.norm(flatImMean, axis = 2)
    hubDistance -= np.mean(hubDistance)
    covariance = np.mean(bgDiff * hubDistance)
    #bgDiffOrder = np.argsort(np.ravel(bgDiff))
    #plt.plot(np.ravel(bgDiff)[bgDiffOrder], np.ravel(hubDistance)[bgDiffOrder])
    #plt.show()
    return -np.sign(covariance)

def getStairwayError(aSorted):
    stairways = np.cumsum(aSorted)
    walls = aSorted * np.arange(len(aSorted))
    walls -= stairways
    return walls

def getDensity(aSorted, halfLife):
    k = np.log(2) / halfLife
    
    posExp = aSorted * k
    np.exp(posExp, out = posExp)
    
    negExp = aSorted * -k
    np.exp(negExp, out = negExp)
    
    leftSide = np.cumsum(posExp)
    leftSide *= negExp

    rightSide = negExp
    rightSide[:-1] = rightSide[1:]
    rightSide[-1] = 0
    np.cumsum(rightSide[::-1], axis = 0, out = rightSide[::-1])
    rightSide *= posExp

    result = leftSide
    result += rightSide
    result *= k
    return result

def denseMin(a, densityReward = 0.1, halfLife = 0.1):
    aSorted = np.sort(np.ravel(a))
    stairwayError = getStairwayError(aSorted)
    density = getDensity(aSorted, halfLife)
    density *= densityReward
    density -= stairwayError
    #plt.plot(aSorted, density * 0.000001)
    #plt.show()
    return aSorted[np.argmax(density)]

def getOpaqueDiff(rgbScale, hubAxis, im1, im2, transparentDiffSign):
    scaledIm1, scaledIm2 = getScaledImages(rgbScale, im1, im2)
    bgDiff = np.dot(scaledIm2 - scaledIm1, hubAxis)
    bgDiff *= transparentDiffSign
    opaqueDiff = denseMin(bgDiff) * transparentDiffSign
    return opaqueDiff

def getTransparentDiff(rgbScale, hubAxis, im1, im2, transparentDiffSign):
    scaledIm1, scaledIm2 = getScaledImages(rgbScale, im1, im2)
    bgDiff = np.dot(scaledIm2 - scaledIm1, hubAxis)
    # If transparent diff is positive, find max instead of min
    bgDiff *= -transparentDiffSign
    transparentDiff = denseMin(bgDiff) * -transparentDiffSign
    return transparentDiff

transparentDiffSign = getTransparentDiffSign(rgbScale, hubAxis, im1, im2)
# TODO: Consider weighting the pixels by flattened error when calculating
# opaqueDiff and transparentDiff
opaqueDiff = getOpaqueDiff(rgbScale, hubAxis, im1, im2, transparentDiffSign)
# TODO: Stop assuming transparentDiff is constant throughout the image
transparentDiff = getTransparentDiff(rgbScale, hubAxis, im1, im2,
                                     transparentDiffSign)
# TODO: Stop assuming the hub passes through the origin
adjustedTransparentDiff = transparentDiff - opaqueDiff

adjustedIm1, adjustedIm2 = getAdjustedImages(rgbScale, hubAxis,
                                             opaqueDiff, im1, im2)

print "Transparent Difference Sign:", transparentDiffSign
print "Opaque Difference:", opaqueDiff
print "Transparent Difference:", transparentDiff

saveComparison("afterColorAdjustment.png", rgbScale, hubAxis, opaqueDiff,
               im1, im2)

# Calculate alpha values

def getAdjustedColorSpaceCenter(rgbScale, im1, im2, colorSpaceCenter):
    originalRgbScale = initRgbScale(im1, im2)
    overallRgbScale = rgbScale / originalRgbScale

    adjustedColorSpaceCenter = colorSpaceCenter * np.mean(overallRgbScale,
                                                          axis = 0)
    return adjustedColorSpaceCenter

# TODO: Smooth the difference between the images before using it to find alpha
def getIdealAlpha(hubAxis, adjustedTransparentDiff, adjustedIm1, adjustedIm2):
    result = np.dot(adjustedIm2 - adjustedIm1, hubAxis)
    result -= adjustedTransparentDiff
    result /= -adjustedTransparentDiff
    np.clip(result, 0, 1, out = result)
    return result

def getMinAlpha(adjustedIm1, adjustedIm2, adjustedColorSpaceCenter):
    imMean = adjustedIm1 + adjustedIm2
    imMean *= 0.5
    # TODO: Enforce abs(perChannelCutoff) >= abs(imMean) to avoid zero division
    perChannelCutoff = np.sign(imMean)
    # TODO: Stop asserting against this perfectly normal occurence
    assert np.all(perChannelCutoff != 0)
    perChannelCutoff *= 0.5
    perChannelCutoff += adjustedColorSpaceCenter
    cutoffAlpha = imMean / perChannelCutoff
    minAlpha = np.max(cutoffAlpha, axis = 2)
    return minAlpha

def getCompromiseAlpha(idealAlpha, minAlpha):
    return np.maximum(idealAlpha, minAlpha)

idealAlpha = getIdealAlpha(hubAxis, adjustedTransparentDiff,
                           adjustedIm1, adjustedIm2)

adjustedColorSpaceCenter = getAdjustedColorSpaceCenter(rgbScale, im1, im2,
                                                       colorSpaceCenter)
minAlpha = getMinAlpha(adjustedIm1, adjustedIm2, adjustedColorSpaceCenter)

compromiseAlpha = getCompromiseAlpha(idealAlpha, minAlpha)

# Save transparent image

def getTrueColors(adjustedIm1, adjustedIm2, adjustedColorSpaceCenter,
                  compromiseAlpha):
    trueColors = adjustedIm1 + adjustedIm2
    trueColors *= 0.5
    # TODO: Even though compromiseAlpha is guaranteed to be nonzero, it could
    # still be very small. Make this more numerically stable.
    trueColors /= compromiseAlpha[:, :, None]
    colorSpaceStart = adjustedColorSpaceCenter - 0.5
    trueColors -= colorSpaceStart
    np.clip(trueColors, 0, 1, out = trueColors)
    return trueColors

# TODO: If the true color falls outside the color space, should the
# transparency value be idealAlpha, compromiseAlpha, or something in between?
def getTransparentImage(trueColors, alpha):
    transparentImage = np.concatenate((trueColors, idealAlpha[:, :, None]),
                                      axis = 2)
    return transparentImage

trueColors = getTrueColors(adjustedIm1, adjustedIm2, adjustedColorSpaceCenter,
                           compromiseAlpha)
transparentImage = getTransparentImage(trueColors, idealAlpha)

imsave("idealAlpha.png", idealAlpha)
imsave("transparentImage.png", transparentImage)
