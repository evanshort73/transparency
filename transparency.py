import numpy as np
from scipy.misc import imread, imsave

im1 = imread("in1.png").astype(float) / 255
im2 = imread("in2.png").astype(float) / 255

# Align images

def estimateHubAxis(im1, im2):
    hubAxis = np.mean(im2, axis = (0, 1)) - np.mean(im1, axis = (0, 1))
    hubAxis /= np.linalg.norm(hubAxis)
    return hubAxis

def getTotalArea(*images):
    area = 0
    for image in images:
        area += image.shape[0] * image.shape[1]
    return area

# filterRadius = 2 * sigma
def getFilterSize(filterRadius, sigmasInFrame):
    filterSize = int(round(filterRadius * sigmasInFrame))
    return filterSize

def getGaussian(length, sigmasInFrame):
    x = np.linspace(-sigmasInFrame, sigmasInFrame, length, endpoint = True)
    np.square(x, out = x)
    x *= -0.5
    np.exp(x, out = x)
    x *= sigmasInFrame / (0.5 * length * np.sqrt(2 * np.pi))
    return x

def getSpotlight(shape, sigmasInFrame):
    column = getGaussian(shape[0], sigmasInFrame)
    row = getGaussian(shape[1], sigmasInFrame)
    spotlight = column[:, None] * row
    return spotlight

def getFilterGradient(angle, filterSize):
    x = np.cos(angle)
    y = np.sin(angle)
    row = np.linspace(-x, x, filterSize, endpoint = True)
    column = np.linspace(y, -y, filterSize, endpoint = True)
    gradient = column[:, None] + row
    return gradient

# filter orientations are counterclockwise in the interval [3:00, 9:00)
def getFilters(filterSize, sigmasInFrame, filterCount):
    miniSpotlight = getSpotlight((filterSize, filterSize), sigmasInFrame)
    filters = []
    for i in xrange(filterCount):
        angle = np.pi * i / float(filterCount)
        gradient = getFilterGradient(angle, filterSize)
        gradient *= miniSpotlight
        filters.append(gradient)
    return filters

def getHalfOverhang(fixedShape, movingShape):
    fixedShape = np.asarray(fixedShape[:2])
    movingShape = np.asarray(movingShape[:2])
    smallShape = np.minimum(fixedShape, movingShape)
    overhang = movingShape - (smallShape + 1) // 2
    return overhang

def getConvolvedShape(fixedShape, movingShape, overhang):
    fixedShape = np.asarray(fixedShape[:2])
    movingShape = np.asarray(movingShape[:2])
    overhang = np.asarray(overhang[:2])
    convolvedShape = tuple(fixedShape - movingShape + 1 + 2 * overhang)
    return convolvedShape

def convolve(fixed, moving, overhang):
    wrappedShape = tuple(np.asarray(fixed.shape) + np.asarray(overhang))
    movingReversed = moving[::-1, ::-1]
    wrapped = np.fft.irfft2(
        np.fft.rfft2(fixed, s = wrappedShape) \
        * np.fft.rfft2(moving[::-1, ::-1], s = wrappedShape),
        s = wrappedShape)

    resultShape = getConvolvedShape(fixed.shape, moving.shape, overhang)
    result = wrapped[-resultShape[0]:, -resultShape[1]:]
    return result

def getAlignment(alignmentScores, overhang):
    alignment = np.unravel_index(np.argmax(alignmentScores),
                                 alignmentScores.shape) - overhang
    return alignment

def crop(fixed, moving, alignment):
    alignment = np.asarray(alignment)

    fixedStart = np.maximum(alignment, 0)
    fixedStop = np.minimum(alignment + moving.shape[:2], fixed.shape[:2])
    croppedFixed = fixed[fixedStart[0]:fixedStop[0],
                         fixedStart[1]:fixedStop[1]]

    movingStart = fixedStart - alignment
    movingStop = fixedStop - alignment
    croppedMoving = moving[movingStart[0]:movingStop[0],
                           movingStart[1]:movingStop[1]]

    return croppedFixed, croppedMoving

hubAxis = estimateHubAxis(im1, im2)
hubIm1 = np.dot(im1, hubAxis)
hubIm2 = np.dot(im2, hubAxis)

filterRadius = getTotalArea(im1, im2) * 0.000003
print "Filter radius:", filterRadius

filters = getFilters(getFilterSize(filterRadius, sigmasInFrame = 3),
                     sigmasInFrame = 3, filterCount = 2)

spotlight1 = getSpotlight(
    getConvolvedShape(hubIm1.shape, filters[0].shape, overhang = (0, 0)),
    sigmasInFrame = 3)
spotlight2 = getSpotlight(
    getConvolvedShape(hubIm2.shape, filters[0].shape, overhang = (0, 0)),
    sigmasInFrame = 3)

overhang = getHalfOverhang(spotlight1.shape, spotlight2.shape)

alignmentScores = np.zeros(getConvolvedShape(
    spotlight1.shape, spotlight2.shape, overhang))

for i in xrange(len(filters)):
    filteredIm1 = convolve(hubIm1, filters[i], overhang = (0, 0))
    np.abs(filteredIm1, out = filteredIm1)
    filteredIm1 *= spotlight1

    filteredIm2 = convolve(hubIm2, filters[i], overhang = (0, 0))
    np.abs(filteredIm2, out = filteredIm2)
    filteredIm2 *= spotlight2

    alignmentScores += convolve(filteredIm1, filteredIm2, overhang)

del filteredIm1, filteredIm2
del spotlight1, spotlight2
del hubIm1, hubIm2

alignment = getAlignment(alignmentScores, overhang)
print "Alignment:", alignment

alignmentScores -= np.min(alignmentScores)
alignmentScores *= 255 / np.max(alignmentScores)
np.round(alignmentScores, out = alignmentScores)
np.clip(alignmentScores, 0, 255, out = alignmentScores)
imsave("alignmentScores.png", alignmentScores.astype(np.uint8))
del alignmentScores

alignedIm1, alignedIm2 = crop(im1, im2, alignment)
del im1, im2

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
