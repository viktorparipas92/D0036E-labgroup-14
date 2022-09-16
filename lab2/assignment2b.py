from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from lab1 import functions
from lab2.functions import groupby_aggregate, groupby, my_csv_loader


def assert_approx_equal(value1, value2, relative_threshold=1e-5):
    absolute_threshold = relative_threshold * max(abs(value1), abs(value2))
    assert(abs(value1 - value2) <= absolute_threshold)


def convert_data_to_numpy_arrays(
        features: pd.DataFrame, target: pd.DataFrame
) -> Tuple[np.array, np.array]:
    features_nparray = features.to_numpy()
    X_b = np.c_[np.ones(len(features_nparray)), features_nparray]
    target_nparray = target.to_numpy()
    return X_b, target_nparray


def perform_linear_regression(x, y):
    X_b, y = convert_data_to_numpy_arrays(x, y)
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    print(
        'According to the linear regression estimator, the '
        f'slope of the function is {theta_best[1]:.3f}, while the '
        f'0th-order term is {theta_best[0]:.1f}'
    )

    # Check with sklearn linear regressor
    linear_regressor = LinearRegression()
    linear_regressor.fit(x.to_frame(), y)
    assert_approx_equal(theta_best[1], linear_regressor.coef_[0])
    assert_approx_equal(theta_best[0], linear_regressor.intercept_)
    return X_b, y, theta_best


def plot_linear_regression(x, y, theta_best):
    plt.scatter(x.to_frame(), y)
    axes = plt.gca()
    regression_x = np.array(axes.get_xlim())
    regression_y = theta_best[0] + theta_best[1] * regression_x
    plt.plot(regression_x, regression_y, 'r-')
    plt.show()


def make_predictions_with_linear_regression(ages_to_predict, theta_best):
    predicted_incomes = np.c_[np.ones((2, 1)), ages_to_predict].dot(theta_best)
    for age in ages_to_predict:
        print(
            f"The predicted income for {age}-year-olds is: "
            f"{predicted_incomes[np.where(ages_to_predict == age)][0]:.1f}."
        )
    return predicted_incomes


def check_mean_squared_error(X_b, y, theta_best):
    all_predicted_incomes = X_b.dot(theta_best)
    my_mean_squared_error = ((all_predicted_incomes - y) ** 2).mean()
    print(
        "The amount of the mean squared error (MSE) between the predicted "
        f"incomes and the actual incomes is {my_mean_squared_error:.2f}. "
    )
    sklearn_mean_squared_error = mean_squared_error(y, all_predicted_incomes)
    assert_approx_equal(my_mean_squared_error, sklearn_mean_squared_error)
    return my_mean_squared_error


def run_pipeline(features, target):
    X_b, y, theta_best = perform_linear_regression(features, target)
    plot_linear_regression(features, target, theta_best)
    _ = make_predictions_with_linear_regression(
        np.array([35, 80]), theta_best
    )
    _ = check_mean_squared_error(X_b, y, theta_best)


DATASET_PATH = 'datasets/income_data.csv'


if __name__ == '__main__':
    encoding = 'unicode_escape'
    column_names, rows = my_csv_loader(
        DATASET_PATH, encoding=encoding, type_map={'2020': float}
    )
    income_data = pd.DataFrame(rows, columns=column_names)
    assert(
        income_data.shape == pd.read_csv(DATASET_PATH, encoding=encoding).shape
    )
    print(income_data.head())

    print("""
    A debugger is more convenient when understanding control flow, 
    e.g. which if-statements/branches get executed
    A printout could be more efficient, for example, in a loop that executes a 
    large number of times if the debug information is only needed on a certain
    criterion. e.g. every X iterations then a value should be printed
    """)

    mean_income_by_region_per_age_group = groupby_aggregate(
        groupby(income_data, 'age'),
        cols=['2020'],
        fn_aggregate=functions.mean,
        group_column_name='age'
    )
    assert(
        mean_income_by_region_per_age_group.shape ==
        income_data.groupby('age')[
            ['region', '2020']
        ].mean().reset_index().shape
    )
    print(mean_income_by_region_per_age_group.head())

    ages = mean_income_by_region_per_age_group.age
    ages_numeric = pd.to_numeric(ages.str.strip('+ years'))
    mean_incomes_2020 = mean_income_by_region_per_age_group['2020']

    run_pipeline(ages_numeric, mean_incomes_2020)

    print("Now repeat for 30+-year-olds only")
    ages_30_and_above = ages_numeric[ages_numeric > 30]
    ages_30_and_above_indices = ages_30_and_above.index
    mean_incomes_for_30_and_above = mean_incomes_2020[ages_30_and_above_indices]

    run_pipeline(ages_30_and_above, mean_incomes_for_30_and_above)
    print(
        "The newly calculated MSE value is about 21% of that of the original "
        "model. That means the linear regression is a better fit for this "
        "truncated dataset."
        "When we fit the linear regressor to only the incomes of people above "
        "30, the resulting slope is higher (in an absolute sense), i.e. the "
        "modified model predicts a much higher income for a 35-year-old, "
        "and a slightly lower income for an 80-year-old. "
        "Looking at the graph, our model seems to have less variance and more "
        "bias, that is to say it is underfitting the data. One possible "
        "improvement could be to try to fit a higher-order polynomial instead "
        "of a straight line."
    )






