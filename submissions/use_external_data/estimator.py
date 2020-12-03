
import os
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor


def _merge_airport_data(X):
    filepath = os.path.join(os.path.dirname(__file__), "external_data.csv")

    X = X.copy()  # to avoid raising SettingOnCopyWarning
    # Make sure that DateOfDeparture is of dtype datetime
    data_airport = pd.read_csv(filepath, sep=";")

    X_airport = data_airport[
        [
            "AirportCode",
            "CityPopulation2012",
            "DomesticEnplanedPassengers2012(millions)",
            "AverageDomesticFlightFare2012($)",
        ]
    ]
    X_airport = X_airport.rename(
        columns={
            "AirportCode": "Departure",
            "CityPopulation2012": "PopulationDeparture",
            "DomesticEnplanedPassengers2012(millions)": "NbPassengersDeparture",
            "AverageDomesticFlightFare2012($)": "AverageFlightFareDeparture",
        }
    )
    X_merged = pd.merge(X, X_airport, how="left", on=["Departure"], sort=False)
    X_airport = X_airport.rename(
        columns={
            "Departure": "Arrival",
            "PopulationDeparture": "PopulationArrival",
            "NbPassengersDeparture": "NbPassengersArrival",
            "AverageFlightFareDeparture": "AverageFlightFareArrival",
        }
    )
    X_merged = pd.merge(X_merged, X_airport, how="left", on=["Arrival"], sort=False)

    return X_merged


data_merger = FunctionTransformer(_merge_airport_data)
def _encode_dates(X):
    
    X.loc[:, 'DateOfDeparture'] = pd.to_datetime(X['DateOfDeparture'])
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, 'year'] = X['DateOfDeparture'].dt.year
    X.loc[:, 'month'] = X['DateOfDeparture'].dt.month
    X.loc[:, 'day'] = X['DateOfDeparture'].dt.day
    X.loc[:, 'weekday'] = X['DateOfDeparture'].dt.weekday
    X.loc[:, 'week'] = X['DateOfDeparture'].dt.week
    X.loc[:, 'n_days'] = X['DateOfDeparture'].apply(
        lambda date: (date - pd.to_datetime("1970-01-01")).days
    )
    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["DateOfDeparture"])


def get_estimator():
    data_merger = FunctionTransformer(_merge_airport_data)

    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ["DateOfDeparture"]

    categorical_encoder = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OrdinalEncoder()
    )
    categorical_cols = ['Arrival', 'Departure']

    preprocessor = make_column_transformer(
        (date_encoder, date_cols),
        (categorical_encoder, categorical_cols),
        remainder='passthrough',  # passthrough numerical columns as they are
    )

    regressor = RandomForestRegressor(
        n_estimators=10, max_depth=4, max_features=10, n_jobs=4
    )

    return make_pipeline(data_merger, preprocessor, regressor)
