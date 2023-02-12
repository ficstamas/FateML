from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from fateml.data.utils import DataSplits
import statsmodels.api as sm
from .fishmarket import _standardize
import pandas as pd


def load_dataset(standardize=True, statsmodels_format=True, scaled=True):
    df = load_diabetes(as_frame=True, scaled=scaled)
    df = df.frame
    splits = DataSplits()
    splits.features = {
        "numeric": df.columns.difference(["target", "sex"]).tolist(),
        "categorical": ["sex"],
        "target": "target"
    }

    train, test = train_test_split(df, train_size=0.8, random_state=0)
    train, dev = train_test_split(train, train_size=int(len(df) * 0.7), random_state=0)

    if standardize:
        _, test_, other = _standardize(train[splits.features["numeric"]],
                                       test[splits.features["numeric"]])
        train_, dev_, other = _standardize(train[splits.features["numeric"]],
                                           dev[splits.features["numeric"]])
        splits.other["preprocessor"] = other
        train = pd.concat([train_, train[splits.features["categorical"] + ["target"]]], axis=1)
        test = pd.concat([test_, test[splits.features["categorical"] + ["target"]]], axis=1)
        dev = pd.concat([dev_, dev[splits.features["categorical"] + ["target"]]], axis=1)

    if statsmodels_format:
        train = sm.add_constant(train)
        test = sm.add_constant(test)
        dev = sm.add_constant(dev)

    splits.train_x = train[train.columns.difference(["target"])]
    splits.train_y = train[["target"]]
    splits.test_x = test[test.columns.difference(["target"])]
    splits.test_y = test[["target"]]
    splits.dev_x = dev[dev.columns.difference(["target"])]
    splits.dev_y = dev[["target"]]
    return splits
