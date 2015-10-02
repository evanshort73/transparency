import numpy as np
from scipy.misc import imread, imsave
#from matplotlib import pyplot as plt
from ellipseDistance import closestEllipsePoint
from checkerboard import checkerboard

# Setup

def initRgbScale(im1, im2):
    rgbScale = np.vstack((np.var(im1, axis = (0, 1)),
                          np.var(im2, axis = (0, 1))))
    rgbScale /= np.sum(rgbScale, axis = 0)
    np.sqrt(rgbScale, out = rgbScale)
    return rgbScale

def initHubAxis(im1, im2):
    meanDiff = np.mean(im2, axis = (0, 1)) - np.mean(im1, axis = (0, 1))
    hubAxisDiff = np.linalg.norm(meanDiff)
    hubAxis = meanDiff / hubAxisDiff
    opaqueDiff = -hubAxisDiff
    return hubAxis, opaqueDiff

def initOuterMatrix(im1, im2):
    centeredIm1 = im1 - np.mean(im1, axis = (0, 1))
    centeredIm2 = im2 - np.mean(im2, axis = (0, 1))
    
    originalRgbScale = initRgbScale(im1, im2)
    scale1 = 1 / originalRgbScale[0]
    scale2 = 1 / originalRgbScale[1]
    
    xxT = scale1[:, None] * np.einsum("ijk,ijl", centeredIm1, centeredIm1) \
          * scale1
    xyT = scale1[:, None] * np.einsum("ijk,ijl", centeredIm1, centeredIm2) \
          * scale2
    yyT = scale2[:, None] * np.einsum("ijk,ijl", centeredIm2, centeredIm2) \
          * scale2
    
    outerMatrix = np.vstack((np.hstack((xxT, -xyT)),
                             np.hstack((-xyT.T, yyT))))
    return outerMatrix

def getScaledImages(rgbScale, im1, im2):
    originalRgbScale = initRgbScale(im1, im2)
    overallRgbScale = rgbScale / originalRgbScale
    scaledIm1 = im1 * overallRgbScale[0]
    scaledIm2 = im2 * overallRgbScale[1]
    return scaledIm1, scaledIm2

def getAdjustedImages(hubAxis, opaqueDiff, scaledIm1, scaledIm2):
    oldMeanDiff = np.mean(scaledIm2, axis = (0, 1)) \
                  - np.mean(scaledIm1, axis = (0, 1))
    newMeanDiff = hubAxis * -opaqueDiff
    adjustment = 0.5 * (newMeanDiff - oldMeanDiff)
    adjustedIm1 = scaledIm1 - adjustment
    adjustedIm2 = scaledIm2 + adjustment
    return adjustedIm1, adjustedIm2

im1 = imread("in1.png").astype(float) / 255
im2 = imread("in2.png").astype(float) / 255

imsave("beforeColorAdjustment.png",
       np.round(checkerboard(im1, im2) * 255).astype(np.uint8))

rgbScale = initRgbScale(im1, im2)

# opaqueDiff is the typical value of [(im2 - im1) dot hubAxis] in opaque areas
hubAxis, opaqueDiff = initHubAxis(im1, im2)

outerMatrix = initOuterMatrix(im1, im2)

# Adjust colors

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

# TODO: Encapsulate the calculation of rgbScale into a function that returns
# values corresponding to the original images instead of pre-equalized images
print "RGB Scale:", rgbScale
print "Hub Axis:", hubAxis
print "Flat Error:", objective

im1, im2 = getScaledImages(rgbScale, im1, im2)
scaledIm1 = im1
scaledIm2 = im2

imsave("afterColorScaling.png",
       np.round(checkerboard(scaledIm1, scaledIm2) * 255).astype(np.uint8))

# Calculate transparency

def getTransparentDiffSign(hubAxis, scaledIm1, scaledIm2):
    bgDiff = np.dot(scaledIm2 - scaledIm1, hubAxis)
    bgDiff -= np.mean(bgDiff)
    imMean = scaledIm1 + scaledIm2
    imMean *= 0.5
    imMean -= np.mean(imMean, axis = (0, 1))
    hubDistance = np.linalg.norm(imMean, axis = 2)
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

def getOpaqueDiff(hubAxis, scaledIm1, scaledIm2, transparentDiffSign):
    bgDiff = np.dot(scaledIm2 - scaledIm1, hubAxis)
    bgDiff -= np.mean(bgDiff)
    bgDiff *= transparentDiffSign
    opaqueDiff = denseMin(bgDiff) * transparentDiffSign
    return opaqueDiff

def getTransparentDiff(hubAxis, scaledIm1, scaledIm2, transparentDiffSign):
    bgDiff = np.dot(scaledIm2 - scaledIm1, hubAxis)
    bgDiff -= np.mean(bgDiff)
    # If transparent diff is positive, find max instead of min
    bgDiff *= -transparentDiffSign
    transparentDiff = denseMin(bgDiff) * -transparentDiffSign
    return transparentDiff

# TODO: Smooth the difference between the images before using it to find alpha
def getIdealAlpha(hubAxis, scaledIm1, scaledIm2, opaqueDiff, transparentDiff):
    bgDiff = np.dot(scaledIm2 - scaledIm1, hubAxis)
    bgDiff -= np.mean(bgDiff)
    idealAlpha = bgDiff
    idealAlpha -= transparentDiff
    idealAlpha /= opaqueDiff - transparentDiff
    np.clip(idealAlpha, 0, 1, out = idealAlpha)
    # Remove negative zero so that we don't get negative infinity from division
    np.abs(idealAlpha, out = idealAlpha)
    return idealAlpha

transparentDiffSign = getTransparentDiffSign(hubAxis, scaledIm1, scaledIm2)
# TODO: Consider weighting the pixels by flattened error when calculating
# opaqueDiff and transparentDiff
opaqueDiff = getOpaqueDiff(hubAxis, scaledIm1, scaledIm2, transparentDiffSign)
# TODO: Stop assuming transparentDiff is constant throughout the image
transparentDiff = getTransparentDiff(hubAxis, scaledIm1, scaledIm2,
                                     transparentDiffSign)

idealAlpha = getIdealAlpha(hubAxis, scaledIm1, scaledIm2,
                           opaqueDiff, transparentDiff)

print "Transparent Difference Sign:", transparentDiffSign
print "Opaque Difference:", opaqueDiff
print "Transparent Difference:", transparentDiff

scaledIm1, scaledIm2 = getAdjustedImages(hubAxis, opaqueDiff,
                                         scaledIm1, scaledIm2)
adjustedIm1 = scaledIm1
adjustedIm2 = scaledIm2

imsave("afterColorAdjustment.png",
       np.round(checkerboard(adjustedIm1, adjustedIm2) * 255).astype(np.uint8))

# Calculate true colors

def getTrueColors(adjustedIm1, adjustedIm2, hubCenter, idealAlpha):
    imMean = adjustedIm1 + adjustedIm2
    imMean *= 0.5
    imMean -= hubCenter

    oldSettings = np.seterr(divide = "ignore")
    
    colorMultiplier = np.reciprocal(idealAlpha)
    
    perChannelCutoff = np.sign(imMean)
    perChannelCutoff += 1
    perChannelCutoff *= 0.5
    perChannelCutoff -= hubCenter
    perChannelCutoff /= imMean

    np.seterr(**oldSettings)
    
    perChannelCutoff = np.min(perChannelCutoff, axis = 2)
    colorMultiplier = np.minimum(colorMultiplier, perChannelCutoff, out = colorMultiplier)
    # TODO: Test an example where this is necessary
    colorMultiplier[np.all(imMean == 0, axis = 2)] = 1
    assert np.all(np.isfinite(colorMultiplier))
    # TODO: Test an example where this is necessary
    np.maximum(colorMultiplier, 1, out = colorMultiplier)

    trueColors = imMean
    trueColors *= colorMultiplier[:, :, None]
    trueColors += hubCenter
    # Even though we never scale colors past the color space to compensate for
    # alpha, some pixels move outside the color space during the initial color
    # adjustment
    np.clip(trueColors, 0, 1, out = trueColors)
    return trueColors

# TODO: If the true color falls outside the color space, should the
# transparency value be idealAlpha, compromiseAlpha, or something in between?
def getTransparentImage(trueColors, alpha):
    transparentImage = np.concatenate((trueColors, idealAlpha[:, :, None]),
                                      axis = 2)
    return transparentImage

# TODO: Stop assuming the hub passes through the mean of the images
hubCenter = 0.5 * (np.mean(adjustedIm1, axis = (0, 1)) \
                   + np.mean(adjustedIm2, axis = (0, 1)))

trueColors = getTrueColors(adjustedIm1, adjustedIm2, hubCenter,
                           idealAlpha)
transparentImage = getTransparentImage(trueColors, idealAlpha)

imsave("idealAlpha.png", idealAlpha)
imsave("transparentImage.png", transparentImage)
