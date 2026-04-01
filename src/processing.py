import pandas as pd


def reformat_data(
    data: pd.DataFrame, categorical_cols: list[str] | None = None
) -> pd.DataFrame:
    """
    Data modifications to be applied to both training and test sets.

    """
    df = data.copy()
    df = df.drop(columns=['Unnamed: 0', 'id'])

    # Convert satisfaction to bool
    df['satisfaction'] = df['satisfaction'] == 'satisfied'

    if categorical_cols is not None:
        for col in categorical_cols:
            df[col] = df[col].astype('category')

    return df


def impute_arrival_delays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in 'Arrival Delay in Minutes' with the corresponding
    value in 'Departure Delay in Minutes'.

    """
    missing_arrival_delay = df['Arrival Delay in Minutes'].isna()
    df.loc[missing_arrival_delay, 'Arrival Delay in Minutes'] = df.loc[
        missing_arrival_delay, 'Departure Delay in Minutes'
    ]

    return df


def replace_survey_zeros(df: pd.DataFrame, rated_cols: list[str]) -> pd.DataFrame:
    """
    Replace 0 ratings in survey responses with 3s, the midpoint value of the
    1-5 rating scale. These 0s likely represent "N/A" responses.

    """

    for col in rated_cols:
        df[col] = df[col].replace(0, 3)

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
