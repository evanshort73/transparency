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

scaledIm1, scaledIm2 = getScaledImages(rgbScale, im1, im2)
del im1, im2

imsave("afterColorScaling.png",
       np.round(checkerboard(scaledIm1, scaledIm2) * 255).astype(np.uint8))

imMean = scaledIm1 + scaledIm2
imMean *= 0.5
bgDiff = np.dot(scaledIm2 - scaledIm1, hubAxis)
bgDiff -= np.mean(bgDiff)
del scaledIm1, scaledIm2

# Calculate transparency

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

def getCutoffDiff(bgDiff, direction):
    cutoffDiff = denseMin(bgDiff * -direction) * -direction
    return cutoffDiff

def getMountainWeights(bgDiff, mountainCenter, halfLife = 0.1):
    mountainWeights = bgDiff - mountainCenter
    
    k = np.log(2) / halfLife
    mountainWeights *= k
    
    np.abs(mountainWeights, out = mountainWeights)
    np.negative(mountainWeights, out = mountainWeights)
    np.exp(mountainWeights, out = mountainWeights)
    return mountainWeights

def getHubCenter(imMean, weights):
    weightedImMean = imMean * weights[:, :, None]
    hubCenter = np.sum(weightedImMean, axis=(0, 1)) / np.sum(weights)
    return hubCenter

def getHubCenterCost(imMean, hubCenter, weights):
    centeredImMean = imMean - hubCenter
    np.square(centeredImMean, out = centeredImMean)
    centeredImMeanNormSquared = np.sum(centeredImMean, axis = 2)
    del centeredImMean
    centeredImMeanNormSquared *= weights
    hubCenterCost = np.sum(centeredImMeanNormSquared) / np.sum(weights)
    return hubCenterCost

# TODO: Smooth the difference between the images before using it to find alpha
def getIdealAlpha(bgDiff, opaqueDiff, transparentDiff):
    idealAlpha = bgDiff - transparentDiff
    idealAlpha /= opaqueDiff - transparentDiff
    np.clip(idealAlpha, 0, 1, out = idealAlpha)
    # Remove negative zero so that we don't get negative infinity from division
    np.abs(idealAlpha, out = idealAlpha)
    return idealAlpha

# TODO: Consider weighting the pixels by flattened error when calculating
# opaqueDiff and transparentDiff
# TODO: Stop assuming transparentDiff is constant throughout the image
minDiff = getCutoffDiff(bgDiff, -1)
maxDiff = getCutoffDiff(bgDiff, 1)
transparentMinWeights = getMountainWeights(bgDiff, minDiff)
transparentMinHubCenter = getHubCenter(imMean, transparentMinWeights)
transparentMinCost = getHubCenterCost(imMean, transparentMinHubCenter,
                                      transparentMinWeights)
del transparentMinWeights
transparentMaxWeights = getMountainWeights(bgDiff, maxDiff)
transparentMaxHubCenter = getHubCenter(imMean, transparentMaxWeights)
transparentMaxCost = getHubCenterCost(imMean, transparentMaxHubCenter,
                                      transparentMaxWeights)
del transparentMaxWeights
if transparentMinCost <= transparentMaxCost:
    transparentDiff = minDiff
    opaqueDiff = maxDiff
    hubCenter = transparentMinHubCenter
else:
    transparentDiff = maxDiff
    opaqueDiff = minDiff
    hubCenter = transparentMaxHubCenter

idealAlpha = getIdealAlpha(bgDiff, opaqueDiff, transparentDiff)
del bgDiff

print "Transparent Min:"
print "    Hub Center:", transparentMinHubCenter
print "    Cost:", transparentMinCost
print "Transparent Max:"
print "    Hub Center:", transparentMaxHubCenter
print "    Cost:", transparentMaxCost
if transparentMinCost <= transparentMaxCost:
    print "Choosing min diff to be transparent"
else:
    print "Choosing max diff to be transparent"
print "Transparent Difference:", transparentDiff
print "Opaque Difference:", opaqueDiff

# Calculate true colors

def getTrueColors(imMean, hubCenter, idealAlpha):
    centeredImMean = imMean - hubCenter

    oldSettings = np.seterr(divide = "ignore")
    
    colorMultiplier = np.reciprocal(idealAlpha)
    
    perChannelCutoff = np.sign(centeredImMean)
    perChannelCutoff += 1
    perChannelCutoff *= 0.5
    perChannelCutoff -= hubCenter
    perChannelCutoff /= centeredImMean

    np.seterr(**oldSettings)
    
    perChannelCutoff = np.min(perChannelCutoff, axis = 2)
    colorMultiplier = np.minimum(colorMultiplier, perChannelCutoff,
                                 out = colorMultiplier)
    # TODO: Test an example where this is necessary
    colorMultiplier[np.all(centeredImMean == 0, axis = 2)] = 1
    assert np.all(np.isfinite(colorMultiplier))
    # TODO: Test an example where this is necessary
    np.maximum(colorMultiplier, 1, out = colorMultiplier)

    trueColors = centeredImMean
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

trueColors = getTrueColors(imMean, hubCenter, idealAlpha)
transparentImage = getTransparentImage(trueColors, idealAlpha)

imsave("idealAlpha.png", idealAlpha)
imsave("transparentImage.png", transparentImage)
