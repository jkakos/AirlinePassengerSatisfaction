import pandas as pd


def convert_to_categorical(data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Convert the specified columns to categorical dtype.

    """
    df = data.copy()
    for col in cols:
        df[col] = df[col].astype('category')

    return df


def add_business_class_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reformat the 'Class' column to be a binary indicator for whether a
    passenger is in business class.

    """
    df = df.copy()
    df['business_class'] = df['Class'] == 'Business'
    df = df.drop(columns=['Class'])

    return df


def add_cabin_rating_feature(df: pd.DataFrame, cabin_cols: list[str]) -> pd.DataFrame:
    """
    Create a new feature that combines Seat comfort, Inflight entertainment,
    Cleanliness, and Leg room service into an overall cabin experience rating.
    The new rating is calculated as the mean of the individual ratings.

    """
    df['cabin_rating'] = df[cabin_cols].mean(axis=1)

    return df


def add_service_rating_feature(
    df: pd.DataFrame, service_cols: list[str]
) -> pd.DataFrame:
    """
    Create a new feature that combines Baggage handling, Inflight service,
    Checkin service, and On-board service into an overall cabin experience
    rating. The new rating is calculated as the mean of the individual
    ratings.

    """
    df['service_rating'] = df[service_cols].mean(axis=1)

    return df
