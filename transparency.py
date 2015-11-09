import numpy as np
from scipy.misc import imread, imsave

im1 = imread("in1.png").astype(float) / 255
im2 = imread("in2.png").astype(float) / 255

# Align images

def convolve(fixed, moving, overhang):
    fixedShape = np.asarray(fixed.shape)
    movingShape = np.asarray(moving.shape)
    overhang = np.asarray(overhang)

    wrappedShape = tuple(fixedShape + overhang)

    resultShape = tuple(fixedShape - movingShape + 1 + 2 * overhang)

    wrapped = np.fft.irfft2(np.fft.rfft2(fixed[:, :],
                                         s = wrappedShape) \
                            * np.fft.rfft2(moving[::-1, ::-1],
                                           s = wrappedShape),
                            s = wrappedShape)
    result = wrapped[-resultShape[0]:, -resultShape[1]:]

    return result

estimatedHubAxis = np.mean(im2, axis = (0, 1)) - np.mean(im1, axis = (0, 1))

hubIm1 = np.dot(im1, estimatedHubAxis)
hubIm1 -= np.mean(hubIm1)

hubIm2 = np.dot(im2, estimatedHubAxis)
hubIm2 -= np.mean(hubIm2)

smallShape = np.minimum(im1.shape[:2], im2.shape[:2])
overhang = im2.shape[:2] - (smallShape + 1) // 2
alignmentScores = convolve(hubIm1, hubIm2, overhang)
alignmentScores /= np.sqrt(convolve(np.square(hubIm1), np.square(hubIm2),
                                    overhang))
del hubIm1
del hubIm2

alignment = np.unravel_index(np.argmin(alignmentScores),
                             alignmentScores.shape) - overhang
print "Alignment:", alignment

alignmentScores -= np.min(alignmentScores)
alignmentScores *= 255 / np.max(alignmentScores)
np.round(alignmentScores, out = alignmentScores)
np.clip(alignmentScores, 0, 255, out = alignmentScores)
imsave("alignmentScores.png", alignmentScores.astype(np.uint8))
del alignmentScores

im1Start = np.maximum(alignment, 0)
im1Stop = np.minimum(alignment + im2.shape[:2], im1.shape[:2])
alignedIm1 = im1[im1Start[0]:im1Stop[0], im1Start[1]:im1Stop[1]]
del im1

im2Start = im1Start - alignment
im2Stop = im1Stop - alignment
alignedIm2 = im2[im2Start[0]:im2Stop[0], im2Start[1]:im2Stop[1]]
del im2

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

hubAxis = getHubAxis(alignedIm1, alignedIm2)
print "Hub Axis:", hubAxis

flatError = getFlatError(alignedIm1, alignedIm2, hubAxis)
print "Flat Error:", flatError

# Calculate hub center and cutoff values

imMean = alignedIm1 + alignedIm2
imMean *= 0.5
bgDiff = np.dot(alignedIm2 - alignedIm1, hubAxis)
bgDiff -= np.mean(bgDiff)
del alignedIm1, alignedIm2

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
