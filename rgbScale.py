import numpy as np
from ellipseDistance import closestEllipsePoint

def initRgbScale(im1, im2):
    rgbScale = np.vstack((np.var(im1, axis = (0, 1)),
                          np.var(im2, axis = (0, 1))))
    rgbScale /= np.sum(rgbScale, axis = 0)
    np.sqrt(rgbScale, out = rgbScale)
    return rgbScale

def initHubAxis(im1, im2):
    meanDiff = np.mean(im2, axis = (0, 1)) - np.mean(im1, axis = (0, 1))
    hubAxis = meanDiff / np.linalg.norm(meanDiff)
    return hubAxis

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

def optimizeObjective(rgbScale, hubAxis, outerMatrix):
    rgbScale = np.array(rgbScale)
    hubAxis = np.array(hubAxis)
    outerMatrix = np.array(outerMatrix)
    
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

    return rgbScale, hubAxis, objective, objectiveHistory

def getRgbScaleAndHubAxis(im1, im2):
    originalRgbScale = initRgbScale(im1, im2)
    originalHubAxis = initHubAxis(im1, im2)
    outerMatrix = initOuterMatrix(im1, im2)
    
    rgbScale, hubAxis, objective, objectiveHistory = \
        optimizeObjective(originalRgbScale, originalHubAxis, outerMatrix)
    
    overallRgbScale = rgbScale / originalRgbScale
    return overallRgbScale, hubAxis, objective, objectiveHistory
