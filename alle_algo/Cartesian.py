import math

def cartesian_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def main():
    # Example usage
    x1, y1 = 0, 0
    x2, y2 = 1, 1

    distance = cartesian_distance(x1, y1, x2, y2)
    print("Cartesian distance:", distance)

    # Calculate distances between multiple points
    points = [(0, 0), (0, 2), (1, 1)]

    print("Distances between points:")
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        distance = cartesian_distance(x1, y1, x2, y2)
        print(f"Distance from {points[i]} to {points[i+1]}: {distance}")

if __name__ == "__main__":
    main()
