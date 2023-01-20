import dataclasses
import pandas as pd
from typing import Optional, Any


@dataclasses.dataclass
class DataSplits:
    train_x: Optional[pd.DataFrame] = None
    train_y: Optional[pd.Series] = None
    dev_x: Optional[pd.DataFrame] = None
    dev_y: Optional[pd.Series] = None
    test_x: Optional[pd.DataFrame] = None
    test_y: Optional[pd.Series] = None
    other: Optional[Any] = dataclasses.field(default_factory=lambda: {})
