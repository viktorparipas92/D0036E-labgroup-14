"""
Lab 2: Data aggregation and regression
with SCB databases
Lab 1. The data to be used is available in Canvas and has been downloaded from
the SCB website
(www.scb.se).
For this lab you will load 2 tables. The first table includes the
total population by level of education by
region and year. The second table includes mean income by region, age and year.
The goal of this Lab is to load the data, preprocess it and apply aggregation
functions. Finally, you will create a function to perform a multiple linear
regressions on the 2nd dataset and compare results.

Task 1: Load and inspect the data
Create a function to load data from a csv file based on the information
provided in the lectures. Using that function, load the file pop_year_trim.csv
that is available in Canvas. Inspect the values of the loaded data elements
using the debugger and print them on screen (you should use both the debugger
and the printout to verify that the csv loader behaves as expected).
The output should look something
like this:
Tip: Save the loaded rows in a Pandas Dataframe to help with the analysis.

Task 2: Data aggregation:
Now when you have loaded the data successfully you will re-use some of the
functions from Lab 1 to perform data aggregation.
• Calculate the mean population with post-secondary education in the Norrbotten
region in the last 5 years. Do this by filtering all the values in the education
level to only be those of post-secondary education.
• The table produced by this task should look similar to this one:
•
o Using that value calculate the standard deviation as well. What does this
value mean?
• Calculate the mean population for each region in Sweden in the last 5 years.
• Tip: To perform this task you will need to create a function that groups
and sums all the levels of education per region, check how pandas group_by
function works for inspiration.
The average population in Norrbotten should be around 182,250 people.
This value may differ from reality as this data only considers population
between 16-75 years
• Finally let's visualize the data, for 2020 create a histogram with the
regions and their populations.
• The histogram for all regions in 2020 should look like the following graph.

Task 3: Load a second data set
With the load function created, use it to load the second table income_data.csv that is available in
Canvas. This dataset contains information regarding the average income by age.
Check that the data has been loaded successfully using the debugger and a printout on screen. The
output should look something like this:

Discussion: Discuss with your lab partner. When is it more appropriate to use the debugger for
inspecting variable values, and when would you prefer to use a print function? Identify:
• One case when it is more convenient/efficient to use the debugger.
• One case when it is more convenient/efficient to use a printout.

Task 4: Linear regression
For this task you will perform a linear regression on the average yearly income and age data of the year
2020. Based on the example shown in Lecture 8 and chapter 4 of the ML course book.
• Perform the linear regression on the provided dataset
• You will need to group the data once again and provide the mean income per region based on
every age value.
• Create a scatter plot for your data and plot the regression line.
• Checkpoint: A graph with age on the X axis should look like the following.
•
• What are the predicted y values for the following population points (35, 80)
• Evaluate the model using MSE, implement the function that evaluates each of the provided data
points against the predicted value (do not use sklearn MSE functions). (Often while creating
regression models, we would split our data into test and train samples but for the sake of this
exercise we will evaluate against the complete data)
• Explain the obtained MSE value and what does it mean.
• Now take only into consideration the values from people above 30 years and perform the linear
regression again along with the graph.
• Checkpoint: A graph for people above 30 years should look like the following.

•
•
• Calculate once again the predicted y values for the following population points (35, 80)
• Calculate once again the MSE for this new regression and explain the obtained value.
• What differences can you see from the graphs, predicted values and MSE scores from both
linear regressions?
• How do you think this analysis can be improved considered the obtained results.

You have now completed Lab 2.

"""

from matplotlib import pyplot as plt
import pandas as pd

from lab1 import functions


LAST_FIVE_YEARS = ['2016', '2017', '2018', '2019', '2020']


if __name__ == '__main__':
    regional_populations_by_year = pd.read_csv('datasets/pop_year_trim.csv')
    # Only show populations with post-secondary education
    college_graduates = regional_populations_by_year[
        regional_populations_by_year['level of education']
        == 'post secondary education'
    ]
    norrbotten_college_graduate_populations = college_graduates[
        college_graduates.region == '25 Norrbotten county'
    ][LAST_FIVE_YEARS].values[0]
    my_mean = functions.mean(norrbotten_college_graduate_populations)
    assert(my_mean == norrbotten_college_graduate_populations.mean())
    print('Mean population with post-secondary education in Norrbotten '
          f'region in the last 5 years: {my_mean:.1f}')

    my_stdev = functions.standard_deviation(
        norrbotten_college_graduate_populations, biased=True
    )
    assert(my_stdev == norrbotten_college_graduate_populations.std())
    print(
        'Standard deviation of the populations with post-secondary education '
        f'in Norrbotten region in the last 5 years: {my_stdev:.1f}'
    )

    data_with_sum_by_region = regional_populations_by_year.groupby('region')[
        LAST_FIVE_YEARS
    ].sum().reset_index()
    data_with_sum_by_region['mean population'] = [
        functions.mean(p) for p
        in data_with_sum_by_region[LAST_FIVE_YEARS].values
    ]
    print(data_with_sum_by_region)

    # data_with_sum_by_region[
    #     ['region', 'mean population']
    # ].set_index('region').plot.bar()
    # plt.show()

    data_with_sum_by_region[
        ['region', '2020']
    ].set_index('region').plot.bar()
    plt.show()


