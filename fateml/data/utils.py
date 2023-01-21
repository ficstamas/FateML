import dataclasses
import pandas as pd
from typing import Optional, Any
from sklearn.preprocessing import LabelEncoder
import copy
import numpy as np


@dataclasses.dataclass
class DataSplits:
    train_x: Optional[pd.DataFrame] = None
    train_y: Optional[pd.DataFrame] = None
    dev_x: Optional[pd.DataFrame] = None
    dev_y: Optional[pd.DataFrame] = None
    test_x: Optional[pd.DataFrame] = None
    test_y: Optional[pd.DataFrame] = None
    other: Optional[Any] = dataclasses.field(default_factory=lambda: {})


def binarize_labels_in_splits(split: DataSplits, target_label: str) -> DataSplits:
    split = copy.deepcopy(split)
    if "label_encoder" not in split.other:
        raise KeyError("'label_encoder' is missing from split.other")

    if split.train_y is not None:
        split.train_y = binarize_label(split.train_y, target_label, split.other["label_encoder"])
    if split.dev_y is not None:
        split.dev_y = binarize_label(split.dev_y, target_label, split.other["label_encoder"])
    if split.test_y is not None:
        split.test_y = binarize_label(split.test_y, target_label, split.other["label_encoder"])
    return split


def binarize_label(df_: pd.DataFrame, target_label: str, label_encoder: LabelEncoder) -> pd.DataFrame:
    classes = label_encoder.classes_
    id_ = np.where(classes == target_label)[0]
    if len(id_) == 0:
        raise ValueError(f"Label is not in {classes}")
    id_ = int(id_[0])
    series = df_[df_.columns[0]].copy()
    series[series != id_] = 0
    series[series == id_] = 1
    series.name = target_label
    return series.to_frame()
