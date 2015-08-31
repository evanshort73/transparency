import numpy as np

def getImageOffset(*images):
    channelMin = np.min(np.array([np.min(i, axis = (0, 1)) for i in images]),
                        axis = 0)
    return channelMin

def getImageScale(*images):
    channelMin = getImageOffset(*images)
    channelMax = np.max(np.array([np.max(i, axis = (0, 1)) for i in images]),
                        axis = 0)
    return np.max(channelMax - channelMin)
    

def imageNormalize(*images):
    channelMin = getImageOffset(*images)
    for i in images:
        i -= channelMin
    imageMax = max([np.max(i) for i in images])
    for i in images:
        i /= imageMax
    if len(images) > 1:
        return images
    else:
        return images[0]
