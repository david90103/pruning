import math

def ackley(arr):
    firstSum = 0.0
    secondSum = 0.0
    for c in arr:
        firstSum += c**2.0
        secondSum += math.cos(2.0*math.pi*c)
    n = float(len(arr))

    return -20.0*math.exp(-0.2*math.sqrt(firstSum/n)) - math.exp(secondSum/n) + 20 + math.e