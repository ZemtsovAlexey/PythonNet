from math import atan2,degrees


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    x = 0
    y = 0

class MathUtils(object):

    @staticmethod
    def GetAngleOfLineBetweenTwoPoints(p1, p2):
        xDiff = p2.x - p1.x
        yDiff = p2.y - p1.y

        return degrees(atan2(yDiff, xDiff))