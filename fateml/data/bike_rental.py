import pandas as pd
from sklearn.model_selection import train_test_split
from fateml.data.utils import DataSplits
import statsmodels.api as sm
import numpy as np
from .fishmarket import _standardize
from sklearn.preprocessing import OneHotEncoder


def load_dataset(standardize: bool, statsmodels_format: bool):
    df = pd.read_csv("https://raw.githubusercontent.com/ficstamas/FateML/1d501349b56c8c65e601b6ce40c56cca2a8050a5/notebooks/data/bike_rental_daily.csv")
    splits = DataSplits()
    df = df.drop(['instant', 'dteday', 'casual', 'registered'], axis=1)
    features_boolean = ['holiday', 'workingday', 'yr']
    target = 'cnt'

    wsit = df['weathersit']

    encoder = OneHotEncoder()
    data = encoder.fit_transform(wsit.values.reshape(-1, 1) - 1).toarray()
    data = np.concatenate([data, np.zeros((len(data), 1))], axis=1)
    categories = ["clear", "foggy", "rainy", "storm"]
    index = df.index
    weather_state = pd.DataFrame(data=data, index=index, columns=categories)
    features_boolean = features_boolean + categories

    df = pd.concat([df[df.columns.difference(['weathersit'])], weather_state], axis=1)

    season = df['season']

    encoder = OneHotEncoder()
    data = encoder.fit_transform(season.values.reshape(-1, 1) - 1).toarray()
    categories = ["winter", "spring", "summer", "fall"]
    index = df.index
    season = pd.DataFrame(data=data, index=index, columns=categories)
    features_boolean = features_boolean + categories

    df = pd.concat([df[df.columns.difference(['season'])], season], axis=1)

    months = df['mnth']

    encoder = OneHotEncoder()
    data = encoder.fit_transform(months.values.reshape(-1, 1) - 1).toarray()
    categories = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                  "November", "December"]
    index = df.index
    months = pd.DataFrame(data=data, index=index, columns=categories)
    features_boolean = features_boolean + categories

    df = pd.concat([df[df.columns.difference(['mnth'])], months], axis=1)

    weekday = df['weekday']

    encoder = OneHotEncoder()
    data = encoder.fit_transform(weekday.values.reshape(-1, 1)).toarray()
    categories = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    index = df.index
    weekday = pd.DataFrame(data=data, index=index, columns=categories)
    features_boolean = features_boolean + categories

    df = pd.concat([df[df.columns.difference(['weekday'])], weekday], axis=1)

    features_numeric = df.columns.difference(features_boolean)

    train, test = train_test_split(df, train_size=0.8, random_state=0)
    train, dev = train_test_split(train, train_size=int(len(df) * 0.7), random_state=0)

    if standardize:
        _, test_, other = _standardize(train[train.columns.difference(features_boolean + [target, ])],
                                       test[test.columns.difference(features_boolean + [target, ])])
        train_, dev_, other = _standardize(train[train.columns.difference(features_boolean + [target, ])],
                                           dev[dev.columns.difference(features_boolean + [target, ])])
        splits.other["normalizer"] = other
        train = pd.concat([train_, train[features_boolean + [target, ]]], axis=1)
        test = pd.concat([test_, test[features_boolean + [target, ]]], axis=1)
        dev = pd.concat([dev_, dev[features_boolean + [target, ]]], axis=1)

    if statsmodels_format:
        train = sm.add_constant(train)
        test = sm.add_constant(test)
        dev = sm.add_constant(dev)

    splits.train_x = train[train.columns.difference([target])]
    splits.train_y = train[[target]]
    splits.test_x = test[test.columns.difference([target])]
    splits.test_y = test[[target]]
    splits.dev_x = dev[dev.columns.difference([target])]
    splits.dev_y = dev[[target]]
    splits.features = {
        "target": target,
        "numeric": features_numeric.tolist(),
        "categorical": features_boolean
    }
    return splits
