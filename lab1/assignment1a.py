import statistics

from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import median_abs_deviation

from functions import (
    min_, max_, spread, mean, mean_2, variance, variance_2, standard_deviation,
    median, median_absolute_deviation,
)


grades = [8, 6, 1, 7, 8, 9, 8, 7, 10, 7, 6, 9, 7]


def task_1():
    min_value = min_(grades)
    assert(min_value == min(grades))
    print(f'The minimum grade is: {min_value}')

    max_value = max_(grades)
    assert(max_value == max(grades))
    print(f'The maximum grade is: {max_value}')

    mean_value = mean(grades)
    assert(mean_value == mean_2(grades))
    assert(mean_value == statistics.mean(grades))
    print(f'The mean grade is: {mean_value:.2f}')

    spread_grades = spread(grades)
    assert(spread_grades == (max(grades) - min(grades)))
    print(f'The spread (range) of the grades is: {spread_grades}')


def task_2():
    variance_value = variance(grades)
    assert (variance_value == variance_2(grades))
    assert (variance_value == statistics.variance(grades))
    print(f'The variance of the grades is: {variance_value:.2f}')

    standard_deviation_grades = standard_deviation(grades)
    assert(standard_deviation_grades == statistics.stdev(grades))
    print(
        f'The standard deviation of the grades is: '
        f'{standard_deviation_grades:.2f}'
    )


def task_3():
    median_grade = median(grades)
    assert(median_grade == statistics.median(grades))
    print(f'The median of the grades is {median_grade}')

    mad_grade = median_absolute_deviation(grades)
    grades_df = pd.DataFrame(grades, columns=['Grades'])
    assert(
        mad_grade == grades_df[['Grades']].apply(median_abs_deviation).Grades
    )
    print(f'The median absolute deviation of the grades is {mad_grade}')


def task_4():
    plt.hist(grades)
    plt.title('Histogram')
    plt.show()

    mode_grade = statistics.mode(grades)
    print(f'The most frequent grade is: {mode_grade}')

    print(
        f'According to the plot, there is one outlier in the sample data, '
        f'which is a grade of 1. All the other grades are >= 6.'
    )


if __name__ == '__main__':
    task_1()
    task_2()
    task_3()
    task_4()
