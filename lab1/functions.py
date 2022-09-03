import math


def min_(array):
    minimum = None
    for element in array:
        if minimum is None or element < minimum:
            minimum = element
    return minimum


def max_(array):
    maximum = None
    for element in array:
        if maximum is None or element > maximum:
            maximum = element
    return maximum


def spread(array):
    return max_(array) - min_(array)


def mean(array):
    return sum(array) / len(array)


def mean_2(array):
    sum_ = 0
    len_ = 0
    for element in array:
        sum_ += element
        len_ += 1
    return sum_ / len_


def variance(array):
    mean_value = mean(array)
    numerator = sum((element - mean_value)**2 for element in array)
    return numerator / (len(array) - 1)


def variance_2(array):
    mean_value = mean_2(array)
    sum_numerator = 0
    sum_denominator = -1
    for element in array:
        sum_numerator += (element - mean_value)**2
        sum_denominator += 1
    return sum_numerator / sum_denominator


def standard_deviation(array):
    return math.sqrt(variance(array))


def median(array):
    sorted_array = sorted(array)
    length = len(array)
    if length % 2:  # odd
        indices = [(length - 1) // 2]
    else:  # even
        indices = [length//2 - 1, length // 2]

    return mean([sorted_array[i] for i in indices])


def median_absolute_deviation(array):
    median_of_array = median(array)
    absolute_deviations_from_median = [
        abs(element - median_of_array) for element in array
    ]
    return median(absolute_deviations_from_median)
