import pandas as pd
import re


def my_csv_loader(filename, type_map=None, debug=False, encoding=None):
    def split_line(line):
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
        # First line contains column names
        column_names.extend(split_line(next(file)))

        if debug:
            print(column_names)

        # Map column name to zero-based column index
        for index, column_name in enumerate(column_names):
            column_names_to_index[column_name] = index

        # Now load all the rows
        for line in file:
            values = split_line(line)

            # All the values are now in string, perform type mapping (if any)
            for column_name, type_ in type_map.items():
                column_index = column_names_to_index[column_name]
                values[column_index] = type_(values[column_index])

            rows.append(values)
            if debug and len(rows) < 5:
                print(rows)

    return column_names, rows


def groupby(df, groupby):
    """
    Function to group by a given column
    """
    # We need to first sort on the grouping column
    df = df.sort_values(groupby)

    last = None
    group = []
    # Iterate through the dataset by row.
    # Whenever a new value is found for the grouping column, that means
    # a new group has started and we emit/yield the last group
    for _, row in df.iterrows():
        # Check if we are on a new group
        if row[groupby] != last and len(group) > 0:
            # Return the group
            yield last, group
            group = []

        last = row[groupby]
        group.append(row)

    if len(group) > 0:
        yield last, group


def groupby_aggregate(df_group, cols, fn_aggregate, group_column_name=None):
    """
    Function to perform an aggregate operation on columns in an existing
    grouping and return a new data frame
    """
    groupnames = []
    col_sums = [[] for col in cols]

    # Iterate through all the gruops and apply the aggregate function to it
    for groupname, group in df_group:
        # Apply aggregate to each supplied column
        for i, col in enumerate(cols):
            sum = fn_aggregate([row[col] for row in group])
            col_sums[i].append(sum)

        groupnames.append(groupname)

    # Re-format the resulting data so that we can easily load it
    # into a Pandas Data Frame
    d = {
        'group' if group_column_name is None else group_column_name: groupnames
    }

    for col, sums in zip(cols, col_sums):
        d[col] = sums

    return pd.DataFrame(d)