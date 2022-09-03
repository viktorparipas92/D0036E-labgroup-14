import matplotlib.pyplot as plt
import pandas as pd

from download import fetch_data_from_tgz
from functions import mean


HOUSING_PATH = 'datasets/housing'
HISTOGRAM_COLUMNS = {
    'households': '# households',
    'median_income': 'Med. income',
    'housing_median_age': 'Med. house age [y]',
    'median_house_value': 'Med. value [$]',
}


def task_5(housing_data):
    task_5a(housing_data)
    task_5b(housing_data)
    task_5c(housing_data)
    task_5d(housing_data)
    task_5e(housing_data)
    task_5f(housing_data)


def task_5a(housing_data):
    number_of_districts = housing_data.shape[0]
    print(
        f'The number of districts (rows) in the sample is {number_of_districts}'
    )


def task_5b(housing_data):
    median_house_value_column = housing_data['median_house_value']
    mean_of_median_house_values = mean(list(median_house_value_column))
    assert(
        mean_of_median_house_values
        == housing_data['median_house_value'].mean()
    )
    print(
        f'The mean of the median house values across all districts is '
        f'${mean_of_median_house_values:,.0f}.'
    )


def task_5c(housing_data):
    fig, axes = plt.subplots(4, 1)
    fig.subplots_adjust(hspace=1)
    for i, (key, title) in enumerate(HISTOGRAM_COLUMNS.items()):
        axes[i].hist(housing_data[key])
        axes[i].set_title(title)

    plt.show()


def task_5d(housing_data):
    print(
        '''
        There is an uptick at the tail end of both the housing_median_age and
        the median_house_value histograms, meaning there is a relatively high 
        number of districts with a median house value/age that is close to the
        highest median house value/age across the sample. This indicates that 
        these values are capped.
        
        The fact that there are no outliers in the distribution is mostly thanks
         to handling median values which tend to even out even if individual 
         properties in districts are very old or expensive.
        '''
    )


def task_5e(housing_data):
    print(
        '''
        The magnitude of the values in the median_house_value column
        seem perfectly reasonable (between $100k and $500k.).
        
        If the question is about the median_income column, that range is very 
        low and the values look either pre-processed or at least presented in a 
        unit different than USD.
        
        The Hands-On Machine Learning book suggests that 1 unit corresponds to 
        roughly $10,000 which seems reasonable looking at the range.
        '''
    )


def task_5f(housing_data):
    ocean_proximity_values = list(
        housing_data['ocean_proximity'].value_counts().index
    )

    fig = plt.figure(constrained_layout=True, figsize=(6.4, 9.6))
    fig.suptitle('Histograms')

    subfigs = fig.subfigures(len(ocean_proximity_values), 1)
    for i, choice in enumerate(ocean_proximity_values):
        selected_districts = housing_data[
            housing_data['ocean_proximity'] == choice
        ]
        print(
            f'The mean house value for houses {choice.lower()} is '
            f'${mean(list(selected_districts["median_house_value"])):,.0f}'
        )
        subfigs[i].suptitle(f'{choice}')

        axes = subfigs[i].subplots(1, len(HISTOGRAM_COLUMNS), sharey=True)
        for j, (key, title) in enumerate(HISTOGRAM_COLUMNS.items()):
            axes[j].hist(selected_districts[key])
            axes[j].set_title(f'{title}', fontsize='small')

    plt.show()


if __name__ == '__main__':
    fetch_data_from_tgz(HOUSING_PATH)
    housing_data = pd.read_csv(f'{HOUSING_PATH}/housing.csv')

    task_5(housing_data)
    task_5f(housing_data)
