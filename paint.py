import pygame
from pygame.locals import *
import numpy as np
from parallelogramDistance import closestParallelogramPoint

pygame.init()
screenSize = np.array([640, 480])
screen = pygame.display.set_mode(tuple(screenSize))
bg = pygame.Surface(screenSize)

bgColor = (255, 255, 255)
ellipseColor = (0, 0, 0)
normalColor = (128, 128, 128)
skewColor = (0, 192, 0)
axisColor = (255, 0, 0)
bg.fill(bgColor)

origin = screenSize / 2
newElementPosition = 1
def vectorToSpace(screenVector):
    return np.insert(screenVector, newElementPosition, 0).astype(np.float)
def pointToSpace(screenPoint):
    return vectorToSpace(screenPoint - origin)
def toScreenVector(spaceVector):
    return np.delete(spaceVector, newElementPosition).astype(np.int)
def toScreenPoint(spaceVector):
    return toScreenVector(spaceVector) + origin

lastLine = (np.array([0, 0]), np.array([0, 0]))
skewVector1 = np.array([200, 100])
skewVector2 = np.array([0, 100])
pygame.draw.line(bg, skewColor, tuple(origin), tuple(origin + skewVector1))
pygame.draw.line(bg, skewColor, tuple(origin), tuple(origin + skewVector2))
M = np.empty((3, 2))
M[:, 0] = vectorToSpace(skewVector1)
M[:, 1] = vectorToSpace(skewVector2)
U, s, V = np.linalg.svd(M, full_matrices = False)
axes = U * s
axis1 = toScreenVector(axes[:, 0])
axis2 = toScreenVector(axes[:, 1])
pygame.draw.line(bg, axisColor, tuple(origin), tuple(origin + axis1))
pygame.draw.line(bg, axisColor, tuple(origin), tuple(origin + axis2))

running = True
while running:
    for i in pygame.event.get():
        if i.type == QUIT or (i.type == KEYDOWN and i.key == K_ESCAPE):
            running = False
        elif (i.type == pygame.MOUSEMOTION and i.buttons[0]) \
             or (i.type == MOUSEBUTTONDOWN and i.button == 1):
            mousePoint = np.array(i.pos)
            c = pointToSpace(mousePoint)
            c[newElementPosition] = np.random.random_integers(-200, 200)
            r = np.empty(2)
            r[0], r[1] = closestParallelogramPoint(M[:, 0], M[:, 1], c)
            spaceVector = np.dot(M, r.T)
            screenPoint = toScreenPoint(spaceVector)
            bg.set_at(screenPoint, ellipseColor)
            lastLine = (screenPoint, mousePoint)
    screen.blit(bg, (0, 0))
    pygame.draw.line(screen, normalColor,
                     tuple(lastLine[0]), tuple(lastLine[1]))
    pygame.display.flip()
pygame.quit()
