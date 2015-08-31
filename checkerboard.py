import numpy as np

def checkerboard(im1, im2):
    assert len(im1.shape) >= 2 and im1.shape == im2.shape
    result = np.array(im1)
    
    widthSquares = 6
    widthBoundaries = np.round(
        np.linspace(0, result.shape[1], widthSquares + 1, endpoint = True) \
        ).astype(np.int)
    widthSlices = [slice(widthBoundaries[i], widthBoundaries[i + 1]) \
                   for i in xrange(widthSquares)]
    
    averageWidth = float(result.shape[1]) / float(widthSquares)
    heightSquares = int(np.round(float(result.shape[0]) / averageWidth))
    heightBoundaries = np.round(
        np.linspace(0, result.shape[0], heightSquares + 1, endpoint = True) \
        ).astype(np.int)
    heightSlices = [slice(heightBoundaries[i], heightBoundaries[i + 1]) \
                    for i in xrange(heightSquares)]

    for y in xrange(heightSquares):
        for widthSlice in widthSlices[(y + 1) % 2::2]:
            result[heightSlices[y], widthSlice] = im2[heightSlices[y],
                                                      widthSlice]

    return result

"""
def checkerboard(im1, im2):
    assert len(im1.shape) >= 2 and im1.shape == im2.shape
    widthSquares = 6
    widthSpace = np.linspace(0, widthSquares, im1.shape[1],
                             endpoint = False, dtype = np.int)
    
    averageWidth = float(im1.shape[1]) / float(widthSquares)
    heightSquares = int(np.round(float(im1.shape[0]) / averageWidth))
    heightSpace = np.linspace(0, heightSquares, im1.shape[0],
                              endpoint = False, dtype = np.int)
    
    return np.where((widthSpace + heightSpace[:, None])[:, :, None] % 2,
                     im1, im2)
"""
