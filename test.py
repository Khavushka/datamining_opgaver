import math

# Distance measure function
def dist(x, y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

# Point p
p = (0, 2)

# List of points to check
points = [
    (4, 0),
    (math.sqrt(8), 0),
    (8, 0),
    (math.sqrt(2), 0),
    (math.sqrt(2), math.sqrt(3)),
    (1, 0),
    (0, -2),
    (-math.sqrt(8), math.sqrt(2))
]

# Distance of point p from the origin
p_distance = dist(p, (0, 0))

# Check which points have the same distance as p
same_distance_points = []
for point in points:
    if dist(point, (0, 0)) == p_distance:
        same_distance_points.append(point)

# Print the points that have the same distance as p
print(same_distance_points)

'''
To determine which points have the same distance as point p=(0,2) from the origin (0,0), we can calculate the distance between each point and the origin using the given distance measure and compare it with the distance of point p.

The distance measure is defined as follows:

dist(x,y) = √((x1-y1)^(2))(1,0)(0,2)(x1-y1, x2-y2)^(T)

To find the distance between two points, say (x1, x2) and (y1, y2), we substitute the coordinates into the distance measure formula:

dist((x1, x2), (y1, y2)) = √((x1-y1)^(2))(1,0)(0,2)(x1-y1, x2-y2)^(T)

Now, let's calculate the distance between point p=(0,2) and each of the given points:

(4,0):
dist(p, (4,0)) = √((0-4)^(2))(1,0)(0,2)(0-4, 2-0)^(T) = √((-4)^(2))(1,0)(0,2)(-4, 2)^(T) = √(16)(1,0)(0,2)(-4, 2)^(T) = √(16)(1,0)(0,2)(-4, 2)^(T) = √(16)(-4,4)^(T) = √(16)(16) = 4(4) = 16

(√8,0):
dist(p, (√8,0)) = √((0-√8)^(2))(1,0)(0,2)(0-√8, 2-0)^(T) = √((-√8)^(2))(1,0)(0,2)(-√8, 2)^(T) = √(8)(1,0)(0,2)(-√8, 2)^(T) = √(8)(-√8,2)^(T) = √(8)(8) = 2(8) = 16

(8,0):
dist(p, (8,0)) = √((0-8)^(2))(1,0)(0,2)(0-8, 2-0)^(T) = √((-8)^(2))(1,0)(0,2)(-8, 2)^(T) = √(64)(1,0)(0,2)(-8, 2)^(T) = √(64)(-8,2)^(T) = √(64)(64) = 8(8) = 64

(√2,0):
dist(p, (√2,0)) = √((0-√2)^(2))(1,0)(0,2)(0-√2, 2-0)^(T) = √((-√2)^(2))(1,0)(0,2)(-√2, 2)^(T) = √(2)(1,0)(0,2)(-√2, 2)^(T) = √(2)(-√2,2)^(T) = √(2)(2) = 2(2) = 4

(√2, √3):
dist(p, (√2, √3)) = √((0-√2)^(2)+(2-√3)^(2))(1,0)(0,2)(0-√2, 2-√3)^(T) = √((-√2)^(2)+(2-√3)^(2))(1,0)(0,2)(-√2, 2-√3)^(T) = √(2+3)(1,0)(0,2)(-√2, 2-√3)^(T) = √(5)(-√2,2-√3)^(T) = √(5)(-√2,√3-2)^(T) = √(5)(5) = 5(5) = 25

(1,0):
dist(p, (1,0)) = √((0-1)^(2)+(2-0)^(2))(1,0)(0,2)(0-1, 2-0)^(T) = √((-1)^(2)+(2)^(2))(1,0)(0,2)(-1, 2)^(T) = √(1+4)(1,0)(0,2)(-1, 2)^(T) = √(5)(-1,2)^(T) = √(5)(-5) = 5(5) = 25

(0,-2):
dist(p, (0,-2)) = √((0-0)^(2)+(2+2)^(2))(1,0)(0,2)(0-0, 2+2)^(T) = √((0)^(2)+(4)^(2))(1,0)(0,2)(0, 4)^(T) = √(0+16)(1,0)(0,2)(0, 4)^(T) = √(16)(0,4)^(T) = √(16)(16) = 4(4) = 16

(-√8, √2):
dist(p, (-√8, √2)) = √((0-(-√8))^(2)+(2-√2)^(2))(1,0)(0,2)(0-(-√8), 2-√2)^(T) = √((√8)^(2)+(2-√2)^(2))(1,0)(0,2)(√8, 2-√2)^(T) = √(8+(2-√2)^(2))(1,0)(0,2)(√8, 2-√2)^(T) = √(8+(4-2√2+2))(1,0)(0,2)(√8, 2-√2)^(T) = √(8+(6-2√2))(1,0)(0,2)(√8, 2-√2)^(T) = √(8+(8-2√2))(1,0)(0,2)(√8, 2-√2)^(T) = √(16-2√2)(√8, 2-√2)^(T) = √(16-2√2)(16) = 4(16) = 64

From the calculations, we can see that points 7. (0,-2) and 8. (-√8, √2) have the same distance as point p=(0,2) from the origin (0,0). Therefore, the correct answers are 7 and 8.
'''