import pandas as pd
from sklearn.model_selection import train_test_split
from fateml.data.utils import DataSplits, enforce_dtypes
import statsmodels.api as sm
import numpy as np
from sklearn.impute import SimpleImputer
from .fishmarket import _standardize


def load_dataset(standardize: bool, statsmodels_format: bool, impute_strategy='most_frequent'):
    df = pd.read_csv("https://raw.githubusercontent.com/ficstamas/FateML/1d501349b56c8c65e601b6ce40c56cca2a8050a5/notebooks/data/risk_factors_cervical_cancer.csv")
    splits = DataSplits()

    features_numeric = ['Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies',
                        'Smokes (years)', 'Hormonal Contraceptives (years)', 'IUD (years)', 'STDs (number)',
                        'STDs: Number of diagnosis', 'STDs: Time since first diagnosis',
                        'STDs: Time since last diagnosis']
    features_boolean = ['Smokes', 'Hormonal Contraceptives', 'IUD', 'STDs']
    target = 'Biopsy'
    features = features_numeric + features_boolean

    df = df[features + [target, ]]

    for column in df.columns:
        df.loc[df[column] == '?', column] = np.nan
        # Let's change the boolean values to boolean from string
        if column in features_boolean + [target]:
            df.loc[df[column].astype('float').astype('bool'), column] = True
            df.loc[~df[column].astype('float').astype('bool'), column] = False

    enforce_dtypes(df, features_numeric, float)
    enforce_dtypes(df, features_boolean + [target], bool)

    train, test = train_test_split(df, train_size=0.8, random_state=0)
    train, dev = train_test_split(train, train_size=int(len(df) * 0.7), random_state=0)

    imp_mean = SimpleImputer(strategy=impute_strategy, missing_values=np.nan)
    splits.other["imputer"] = imp_mean
    train_imputed = imp_mean.fit_transform(train)
    dev_imputed = imp_mean.transform(dev)
    test_imputed = imp_mean.transform(test)

    # sadly, sklearn returns a numpy array and not a DataFrame, so we have to fix that
    train = pd.DataFrame(data=train_imputed, columns=train.columns, index=train.index)
    dev = pd.DataFrame(data=dev_imputed, columns=dev.columns, index=dev.index)
    test = pd.DataFrame(data=test_imputed, columns=test.columns, index=test.index)

    enforce_dtypes(train, features_boolean + [target], int)
    enforce_dtypes(dev, features_boolean + [target], int)
    enforce_dtypes(test, features_boolean + [target], int)

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
        "numeric": features_numeric,
        "categorical": features_boolean
    }

    return splits
