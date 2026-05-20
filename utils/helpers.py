import pandas as pd


def validate_dataframe(df):

    if df is None:
        return False

    if df.empty:
        return False

    return True