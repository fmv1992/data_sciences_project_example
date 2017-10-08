"""Provide data processing functions."""
# pylama: ignore=D103


def main():
    pass


def remove_nulls(dataframe, null_value=-1):
    return dataframe.fillna(null_value)


if __name__ == '__main__':
    main()
