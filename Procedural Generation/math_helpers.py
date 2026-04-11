import math


def snap_to_factor(value: int, target: float):
    "Returns the closest factor of the given value to the provided target"
    # Find all factors
    factors = set()
    for i in range(1, int(math.sqrt(value)) + 1):
        if value % i == 0:
            factors.add(i)
            factors.add(int(value/i))
    # Return the closest factor to the target
    return min(factors, key=lambda v: abs(target - v))