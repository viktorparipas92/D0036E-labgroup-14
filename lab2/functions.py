from typing import Generator, List, Dict, Callable, Union, Optional, Tuple

import pandas as pd
import re


def load_csv(
    filename: str,
    type_map: Dict = None,
    debug: bool = False,
    encoding=None,
):
    def split_line(line: str) -> List:
        """
        Split a comma separated line observing quotation
        """
        line = line.rstrip('\n')
        line_split = re.split(
            r",(?=(?:[^\"']*[\"'][^\"']*[\"'])*[^\"']*$)",
            line
        )
        return [l.strip('"').strip("'") for l in line_split]

    column_names = []
    column_names_to_index = {}
    rows = []
    type_map = type_map or {}

    with open(filename, encoding=encoding) as file:
        column_names.extend(split_line(next(file)))
        if debug:
            print(column_names)

        for index, column_name in enumerate(column_names):
            column_names_to_index[column_name] = index

        for line in file:
            values = split_line(line)

            for column_name, type_ in type_map.items():
                column_index = column_names_to_index[column_name]
                values[column_index] = type_(values[column_index])

            rows.append(values)
            if debug and len(rows) < 5:
                print(rows)

    return column_names, rows


def groupby(dataframe: pd.DataFrame, column_name: str) -> Generator[Tuple, None, None]:
    dataframe = dataframe.sort_values(column_name)

    current_value = None
    group = []
    for _, row in dataframe.iterrows():
        if row[column_name] != current_value and len(group) > 0:
            yield current_value, group
            group = []

        current_value = row[column_name]
        group.append(row)

    if group:
        yield current_value, group


def groupby_aggregate(
    groups: Generator[Tuple, None, None],
    columns: List,
    aggregate_function: Callable,
    group_column_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Function to perform an aggregate operation on columns in an existing
    grouping and return a new data frame
    """
    group_names = []
    column_aggregates = [[] for _ in columns]

    for group_name, group in groups:
        for i, column in enumerate(columns):
            aggregate = aggregate_function([row[column] for row in group])
            column_aggregates[i].append(aggregate)

        group_names.append(group_name)

    result: Dict = {
        'group' if group_column_name is None
        else group_column_name: group_names
    }
    for column, aggregate in zip(columns, column_aggregates):
        result[column] = aggregate

    return pd.DataFrame(result)