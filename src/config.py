PROJECT_ID = 'project-e18491ab-d50e-4c36-a7d'

DTYPE_MAP = {
    'id': 'int64',
    'gender': 'str',
    'is_loyal_customer': 'int64',
    'age': 'int64',
    'is_personal_travel': 'int64',
    'class': 'str',
    'flight_distance': 'int64',
    'wifi_service': 'int64',
    'convenient_time': 'int64',
    'online_booking_ease': 'int64',
    'gate_location': 'int64',
    'food_drink': 'int64',
    'online_boarding': 'int64',
    'seat_comfort': 'int64',
    'inflight_entertainment': 'int64',
    'onboard_service': 'int64',
    'leg_room': 'int64',
    'baggage_handling': 'int64',
    'checkin_service': 'int64',
    'inflight_service': 'int64',
    'cleanliness': 'int64',
    'departure_delay': 'int64',
    'arrival_delay': 'int64',
    # Engineered features ---------
    'is_business_class': 'int64',
    'cabin_rating': 'float64',
    'service_rating': 'float64',
    'satisfied': 'int64',
}

NUMERIC_COLS = ['age', 'flight_distance', 'departure_delay', 'arrival_delay']
CATEGORICAL_COLS = ['gender', 'is_loyal_customer', 'is_personal_travel', 'class']

# These columns are all ratings from 0-5 or 1-5
RATED_COLS = [
    'wifi_service',
    'convenient_time',
    'online_booking_ease',
    'gate_location',
    'food_drink',
    'online_boarding',
    'seat_comfort',
    'inflight_entertainment',
    'onboard_service',
    'leg_room',
    'baggage_handling',
    'checkin_service',
    'inflight_service',
    'cleanliness',
]

TARGET = 'satisfied'


def set_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set the data dtypes according to the DTYPE_MAP. Only applies to columns
    that are present in both the DTYPE_MAP and the dataframe.

    """
    active_dtypes = {
        col: dtype for (col, dtype) in DTYPE_MAP.items() if col in df.columns
    }
    return df.astype(active_dtypes)
