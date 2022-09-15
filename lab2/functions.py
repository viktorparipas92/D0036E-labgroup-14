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
