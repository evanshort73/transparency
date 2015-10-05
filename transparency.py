import numpy as np
from scipy.misc import imread, imsave

im1 = imread("in1.png").astype(float) / 255
im2 = imread("in2.png").astype(float) / 255

# Calculate hub axis

def getHubAxis(im1, im2):
    imDiff = im2 - im1
    imDiff -= np.mean(imDiff, axis = (0, 1))
    xxT = np.einsum("ijk,ijl", imDiff, imDiff)
    eigVals, eigVecs = np.linalg.eig(xxT)
    eigIndex = np.argmax(eigVals)
    hubAxis = eigVecs[:, eigIndex]
    return hubAxis

def getFlatError(im1, im2, hubAxis):
    imDiff = im2 - im1
    imDiff -= np.mean(imDiff, axis = (0, 1))
    xxT = np.einsum("ijk,ijl", imDiff, imDiff)
    xTx = np.dot(np.ravel(imDiff), np.ravel(imDiff))
    flatError = xTx - np.dot(hubAxis, np.dot(xxT, hubAxis))
    return flatError

hubAxis = getHubAxis(im1, im2)
print "Hub Axis:", hubAxis

flatError = getFlatError(im1, im2, hubAxis)
print "Flat Error:", flatError

# Calculate hub center and cutoff values

imMean = im1 + im2
imMean *= 0.5
bgDiff = np.dot(im2 - im1, hubAxis)
bgDiff -= np.mean(bgDiff)
del im1, im2

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

# TODO: Consider weighting the pixels by flattened error when calculating
# opaqueDiff and transparentDiff
# TODO: Stop assuming transparentDiff is constant throughout the image
print "Transparent Min:"
minDiff = getCutoffDiff(bgDiff, -1)
transparentMinWeights = getMountainWeights(bgDiff, minDiff)
transparentMinHubCenter = getHubCenter(imMean, transparentMinWeights)
print "    Hub Center:", transparentMinHubCenter
transparentMinCost = getHubCenterCost(imMean, transparentMinHubCenter,
                                      transparentMinWeights)
print "    Cost:", transparentMinCost
del transparentMinWeights

print "Transparent Max:"
maxDiff = getCutoffDiff(bgDiff, 1)
transparentMaxWeights = getMountainWeights(bgDiff, maxDiff)
transparentMaxHubCenter = getHubCenter(imMean, transparentMaxWeights)
print "    Hub Center:", transparentMaxHubCenter
transparentMaxCost = getHubCenterCost(imMean, transparentMaxHubCenter,
                                      transparentMaxWeights)
print "    Cost:", transparentMaxCost
del transparentMaxWeights

if transparentMinCost <= transparentMaxCost:
    print "Choosing min diff to be transparent"
    transparentDiff = minDiff
    opaqueDiff = maxDiff
    hubCenter = transparentMinHubCenter
else:
    print "Choosing max diff to be transparent"
    transparentDiff = maxDiff
    opaqueDiff = minDiff
    hubCenter = transparentMaxHubCenter
print "Transparent Difference:", transparentDiff
print "Opaque Difference:", opaqueDiff

# Calculate transparency values

# TODO: Smooth the difference between the images before using it to find alpha
def getAlpha(bgDiff, opaqueDiff, transparentDiff):
    alpha = bgDiff - transparentDiff
    alpha /= opaqueDiff - transparentDiff
    np.clip(alpha, 0, 1, out = alpha)
    # Remove negative zero so that we don't get negative infinity from division
    np.abs(alpha, out = alpha)
    return alpha

alpha = getAlpha(bgDiff, opaqueDiff, transparentDiff)
del bgDiff

# Calculate true colors

def getTrueColors(imMean, hubCenter, alpha):
    centeredImMean = imMean - hubCenter

    oldSettings = np.seterr(divide = "ignore")
    
    colorMultiplier = np.reciprocal(alpha)
    
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

# It's possible to choose better values for color and alpha in terms of
# reproducing the original images, but that would require more complicated math
# and it wouldn't be as good at preserving fully transparent areas.
def getTransparentImage(trueColors, alpha):
    transparentImage = np.concatenate((trueColors, alpha[:, :, None]),
                                      axis = 2)
    return transparentImage

trueColors = getTrueColors(imMean, hubCenter, alpha)
transparentImage = getTransparentImage(trueColors, alpha)

imsave("alpha.png", alpha)
imsave("transparentImage.png", transparentImage)
