import numpy as np
from scipy.misc import imread, imsave
# from matplotlib import pyplot as plt

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

def getFilterLength(filterRadius, sigmasInFrame):
    idealLength = filterRadius * sigmasInFrame # filterRadius = 2 * sigma
    oddIntegerLength = 2 * int(round(0.5 * (idealLength + 1))) - 1
    return oddIntegerLength

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

def getEdgeFilter(filterSize, sigmasInFrame):
    assert filterSize % 2 == 1
    edgeFilter = getGaussian(filterSize, sigmasInFrame)
    edgeFilter[filterSize // 2:] *= -1
    edgeFilter[filterSize // 2] = 0
    return edgeFilter

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
    wrapped = np.fft.irfft2(
        np.fft.rfft2(fixed, s = wrappedShape) \
        * np.fft.rfft2(moving[::-1, ::-1], s = wrappedShape),
        s = wrappedShape)

    resultShape = getConvolvedShape(fixed.shape, moving.shape, overhang)
    result = wrapped[-resultShape[0]:, -resultShape[1]:]
    return result

def getConvolvedLength(fixedLength, movingLength, overhang):
    convolvedLength = fixedLength - movingLength + 1 + 2 * overhang
    return convolvedLength

def convolve1D(fixed, moving, overhang, axis):
    wrappedLength = fixed.shape[axis] + overhang
    moreDimensions = (slice(None),) + (None,) * (len(fixed.shape) - axis - 1)
    wrapped = np.fft.irfft(
        np.fft.rfft(fixed, n = wrappedLength, axis = axis) \
        * np.fft.rfft(moving[::-1], n = wrappedLength)[moreDimensions],
        n = wrappedLength, axis = axis)

    resultLength = getConvolvedLength(fixed.shape[axis], moving.size, overhang)
    result = wrapped[(slice(None),) * axis + (slice(-resultLength, None),)]
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

filterRadius = np.sqrt(getTotalArea(im1, im2)) * 0.01
print "Filter radius:", filterRadius

filterLength = getFilterLength(filterRadius, sigmasInFrame = 3)
edgeFilter = getEdgeFilter(filterLength, sigmasInFrame = 3)

spotlight1 = getSpotlight(hubIm1.shape, sigmasInFrame = 3)
spotlight2 = getSpotlight(hubIm2.shape, sigmasInFrame = 3)

overhang = getHalfOverhang(spotlight1.shape, spotlight2.shape)

alignmentScores = np.zeros(getConvolvedShape(
    spotlight1.shape, spotlight2.shape, overhang))

for axis in xrange(2):
    filteredIm1 = convolve1D(hubIm1, edgeFilter, filterLength // 2, axis)
    np.abs(filteredIm1, out = filteredIm1)
    filteredIm1 *= spotlight1

    filteredIm2 = convolve1D(hubIm2, edgeFilter, filterLength // 2, axis)
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
del alignedIm1, alignedIm2

def extrapolateVarianceFromPreviousElementsTakingEachElementAsMean(x, weights):
    middleTerm = x * weights
    np.cumsum(middleTerm, out = middleTerm)
    middleTerm *= x
    middleTerm *= -2
    variance = middleTerm
    variance += np.cumsum(np.square(x) * weights)
    variance /= np.cumsum(weights)
    variance += np.square(x)
    return variance

def getMeanOfFirstGaussian(x, weights):
    cost = extrapolateVarianceFromPreviousElementsTakingEachElementAsMean(
        x, weights)
    cost /= np.cumsum(weights)
    endOfNoise = np.argmax(cost)
    bestIndex = endOfNoise + np.argmin(cost[endOfNoise:])
    # plt.plot(x[endOfNoise + 3000:], cost[endOfNoise + 3000:] * 0.01)
    return x[bestIndex]

def getTransparentDiff(sortedBgDiff, weights):
    if np.sum(sortedBgDiff) >= 0:
        sortedBgDiff = sortedBgDiff[::-1]
        weights = weights[::-1]
    return getMeanOfFirstGaussian(sortedBgDiff, weights)

def getOpaqueDiff(sortedBgDiff, weights):
    if np.sum(sortedBgDiff) < 0:
        sortedBgDiff = sortedBgDiff[::-1]
        weights = weights[::-1]
    return getMeanOfFirstGaussian(sortedBgDiff, weights)

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

# TODO: Consider weighting the pixels by flattened error when calculating
# opaqueDiff and transparentDiff
# TODO: Stop assuming transparentDiff is constant throughout the image
bgDiffOrder = np.argsort(np.ravel(bgDiff))
spotlight = getSpotlight(bgDiff.shape, sigmasInFrame = 3)
sortedSpotlight = np.ravel(spotlight)[bgDiffOrder]
del spotlight
sortedBgDiff = np.ravel(bgDiff)[bgDiffOrder]
del bgDiffOrder
# plt.hist(sortedBgDiff, weights = sortedSpotlight, bins = 200, normed = True, rwidth = 1, linewidth = 0)
transparentDiff = getTransparentDiff(sortedBgDiff, sortedSpotlight)
print "Transparent Difference:", transparentDiff
opaqueDiff = getOpaqueDiff(sortedBgDiff, sortedSpotlight)
print "Opaque Difference:", opaqueDiff
# plt.show()
del sortedBgDiff

transparentWeights = getMountainWeights(bgDiff, transparentDiff)
hubCenter = getHubCenter(imMean, transparentWeights)
print "Hub Center:", hubCenter
del transparentWeights

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
