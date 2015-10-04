import numpy as np
from scipy.misc import imread, imsave
#from matplotlib import pyplot as plt
from rgbScale import getRgbScaleAndHubAxis, initRgbScale
from checkerboard import checkerboard

# Setup

im1 = imread("in1.png").astype(float) / 255
im2 = imread("in2.png").astype(float) / 255

imsave("beforeColorAdjustment.png",
       np.round(checkerboard(im1, im2) * 255).astype(np.uint8))

# Adjust colors

rgbScale, hubAxis, objective, objectiveHistory = getRgbScaleAndHubAxis(im1,
                                                                       im2)

#plt.plot(objectiveHistory)
#plt.show()

print "RGB Scale:", rgbScale * initRgbScale(im1, im2)
print "Hub Axis:", hubAxis
print "Flat Error:", objective

scaledIm1 = im1 * rgbScale[0]
del im1
scaledIm2 = im2 * rgbScale[1]
del im2

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
