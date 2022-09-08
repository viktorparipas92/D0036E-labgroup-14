import math

# Define functions for calculating min, max, sum, mean, variance, std, median, median absolute deviance
def my_min(arr):
    low = None

    for item in arr:
        if low is None or item < low:
            low = item

    return low

def my_max(arr):
    return -my_min([-a for a in arr])

def my_sum(arr):

    sum = 0

    for item in arr:
        sum += item

    return sum

def my_mean(arr):
    return sum(arr) / len(arr)

def my_variance(arr):

    mean = my_mean(arr)

    return my_sum([(item - mean) ** 2 for item in arr]) / len(arr)

def my_std(arr):

    return math.sqrt(my_variance(arr))

def my_median(arr):

    arr = sorted(arr)
 
    n = len(arr)

    if n % 2:
        return arr[n // 2]
 
    return (arr[(n-1) // 2] + arr[n//2]) / 2

def my_median_abs_dev(arr):

    m = my_median(arr)

    return my_median([abs(item - m) for item in arr])