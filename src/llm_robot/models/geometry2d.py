from dataclasses import dataclass
from matplotlib.pylab import ArrayLike
import numpy as np
from numpy import sin, cos, tan

@dataclass
class SE2:
    """
    SE2 is a 2D pose with translation and rotation.

    x: translation in x direction
    y: translation in y direction
    theta: rotation (radians if you care about log / exp maps working)
    """
    x: float
    y: float
    theta: float
    def log(self) -> "se2":
        theta = self.theta
        halfu = 0.5 * theta + snz(theta)
        v = halfu / tan(halfu)
        return se2(v * self.x + halfu * self.y, -halfu * self.x + v * self.y, theta)

@dataclass
class se2:
    """
    se2 is a twist which can be integrated into a pose.

    dx: infinitesimal translation in x direction relative to current pose
    dy: infinitesimal translation in y direction relative to current pose
    dtheta: infinitesimal rotation relative to current pose

    """
    dx: float
    dy: float
    dtheta: float
    def exp(self) -> SE2:
        """
        Integrate the twist into a pose.

        Imagine during a time step dt you moved some amount along your immediate x axis and some rotation theta. 

        This motion over dt traces out a circular arc. 

        The exponential map of the twist is the pose that results from this motion.
        """
        angle = self.dyaw
        heading = self.dyaw
        u = angle + snz(angle)
        c = 1 - cos(u)
        s = sin(u)
        translation = Vector2d((s * self.dx - c * self.dy) / u, (c * self.dx + s * self.dy) / u)
        return SE2(translation.x, translation.y, heading)

@dataclass
class Vector2d:
    """
    Vector2d is a 2D vector with x and y components.
    """
    x: float
    y: float
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other: "Vector2d") -> "Vector2d":
        if type(other) == Vector2d:
            return Vector2d(self.x + other.x, self.y + other.y)
        else:
            raise TypeError("Unsupported type for addition with Vector2d")

    def __sub__(self, other: "Vector2d") -> "Vector2d":
        # if other implements add 
        if hasattr(other, "__add__"):
            return self + (-other)

        if type(other) == Vector2d:
            return Vector2d(self.x - other.x, self.y - other.y)
        elif type(other) == ArrayLike:
            assert len(other) == 2
            return Vector2d(self.x - other[0], self.y - other[1])
        else:
            raise TypeError("Unsupported type for subtraction with Vector2d")

    def __mul__(self, other: float) -> "Vector2d":
        return Vector2d(self.x * other, self.y * other)

    def __truediv__(self, other: float) -> "Vector2d":
        return Vector2d(self.x / other, self.y / other)
    
    def __neg__(self) -> "Vector2d":
        return self.rotate(np.pi)

    def norm(self) -> float:
        return (self.x**2 + self.y**2) ** 0.5

    def rotate(self, angle: float) -> "Vector2d":
        """
        Rotate the vector by the given angle (radians).

        This is equivalent to multiplying the vector by the rotation matrix:

        [cos(angle), -sin(angle)] 
        [sin(angle), cos(angle)]

        positive theta corresponds to counter-clockwise rotation.
        """
        return Vector2d(self.x * cos(angle) - self.y * sin(angle), self.x * sin(angle) + self.y * cos(angle))


def snz(x: float) -> float:

    EPS = 1e-9
    return EPS if x >= 0.0 else -EPS

