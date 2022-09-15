from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


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
    # Calculate the coefficients the old-fashioned way
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


if __name__ == '__main__':
    income_data = pd.read_csv(
        'datasets/income_data.csv', encoding='unicode_escape'
    )
    print(income_data)

    print("""
    When is it more appropriate to use the debugger for inspecting variable 
    values, and when would you prefer to use a print function? Identify:

    One case when it is more convenient/efficient to use the debugger. 
    One case when it is more convenient/efficient to use a printout.
    From Kristofer:

    A debugger is more convenient when understanding control flow, 
    e.g. which if-statements/branches that gets executed
    A printout could be more efficient, for example, in a loop that executes a 
    large amount of times and it is only for a certain criteria that debug 
    information is needed, 
    e.g. every X iterations then a value should be printed
    """)

    mean_income_by_region = income_data.groupby('age')[
        ['region', '2020']
    ].mean().reset_index()
    ages = mean_income_by_region.age
    ages_numeric = pd.to_numeric(ages.str.strip('+ years'))
    mean_incomes_2020 = mean_income_by_region['2020']

    X_b, y, theta_best = perform_linear_regression(
        ages_numeric, mean_incomes_2020
    )
    plot_linear_regression(ages_numeric, mean_incomes_2020, theta_best)
    predicted_incomes = make_predictions_with_linear_regression(
        np.array([35, 80]), theta_best
    )
    my_mean_squared_error = check_mean_squared_error(X_b, y, theta_best)

    print("Now repeat for 30+-year-olds only")
    ages_30_and_above = ages_numeric[ages_numeric > 30]
    ages_30_and_above_indices = ages_30_and_above.index
    mean_incomes_for_30_and_above = mean_incomes_2020[ages_30_and_above_indices]

    X_b, y, theta_best = perform_linear_regression(
        ages_30_and_above, mean_incomes_for_30_and_above
    )
    plot_linear_regression(
        ages_30_and_above, mean_incomes_for_30_and_above, theta_best
    )
    predicted_incomes = make_predictions_with_linear_regression(
        np.array([35, 80]), theta_best
    )
    my_mean_squared_error = check_mean_squared_error(X_b, y, theta_best)






