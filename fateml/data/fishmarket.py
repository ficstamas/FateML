import pandas as pd
from sklearn.model_selection import train_test_split
from fateml.data.utils import DataSplits
from sklearn.preprocessing import OneHotEncoder, Normalizer, LabelEncoder
import statsmodels.api as sm
from typing import Dict


def prepare_for_classification(df: pd.DataFrame, normalize=False, statsmodels_format=False) -> DataSplits:
    """
    Prepares the fish market dataset for classification
    :param df: fish market dataset
    :param normalize: Normalize numeric parameters
    :param statsmodels_format: adds an extra const column to the dataset
    :return:
    """
    dataset = DataSplits()

    encoder = OneHotEncoder()
    data = encoder.fit_transform(df['Species'].values.reshape(-1, 1)).toarray()

    categories = df['Species'].unique()
    index = df.index
    encoded_species = pd.DataFrame(data=data, index=index, columns=categories)

    preprocessed_df = pd.concat([df, encoded_species], axis=1)

    train: pd.DataFrame
    test: pd.DataFrame
    train, test = train_test_split(preprocessed_df, train_size=0.7, random_state=0)

    # encode labels
    label = LabelEncoder()
    # collect possible labels
    label.fit(df[['Species']])
    # transform labels on train
    labels_ = label.transform(train['Species'])
    train = pd.concat(
        [
            train[train.columns.difference(['Species'])],
            pd.DataFrame(data=labels_, columns=['Species'], index=train.index)
        ],
        axis=1
    )
    # transform labels on test
    labels_ = label.transform(test['Species'])
    test = pd.concat(
        [
            train[train.columns.difference(['Species'])],
            pd.DataFrame(data=labels_, columns=['Species'], index=test.index)
        ],
        axis=1
    )
    if normalize:
        train_, test_, other = _normalize(train[train.columns.difference(['Species'])],
                                          test[test.columns.difference(['Species'])])
        dataset.other["normalizer"] = other
        train = train_
        test = test_

    if statsmodels_format:
        train = sm.add_constant(train)
        test = sm.add_constant(test)

    dataset.train_x = train[train.columns.difference(['Species'])]
    dataset.train_y = train[['Species']]
    dataset.test_x = test[test.columns.difference(['Species'])]
    dataset.test_y = test[['Species']]

    return dataset


def prepare_for_regression(df: pd.DataFrame, normalize=False, statsmodels_format=False) -> DataSplits:
    """
    Prepares the fish market dataset for regression
    :param df: fish market dataset
    :param normalize: Normalize numeric parameters
    :param statsmodels_format: adds an extra const column to the dataset
    :return:
    """
    dataset = DataSplits()

    encoder = OneHotEncoder()
    data = encoder.fit_transform(df['Species'].values.reshape(-1, 1)).toarray()

    categories = df['Species'].unique().tolist()
    index = df.index
    encoded_species = pd.DataFrame(data=data, index=index, columns=categories)

    preprocessed_df = pd.concat([df, encoded_species], axis=1)

    train: pd.DataFrame
    test: pd.DataFrame
    train, test = train_test_split(preprocessed_df, train_size=0.7, random_state=0)

    if normalize:
        train_, test_, other = _normalize(train[train.columns.difference(categories + ['Weight', 'Species'])],
                                          test[test.columns.difference(categories + ['Weight', 'Species'])])
        dataset.other["normalizer"] = other
        train = pd.concat([train_, train[categories + ['Weight']]], axis=1)
        test = pd.concat([test_, test[categories + ['Weight']]], axis=1)

    if statsmodels_format:
        train = sm.add_constant(train)
        test = sm.add_constant(test)

    dataset.train_x = train[train.columns.difference(['Weight'])]
    dataset.train_y = train[['Weight']]
    dataset.test_x = test[test.columns.difference(['Weight'])]
    dataset.test_y = test[['Weight']]

    return dataset


def _normalize(train: pd.DataFrame, test: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, Dict):
    norm = Normalizer()

    mean = train.mean()
    train = train.subtract(mean)
    test = test.subtract(mean)

    normalized = norm.fit_transform(train)
    train = pd.DataFrame(data=normalized, columns=train.columns, index=train.index)
    normalized = norm.transform(test)
    test = pd.DataFrame(data=normalized, columns=test.columns, index=test.index)
    return train, test, {"normalizer": norm, "mean": mean}

