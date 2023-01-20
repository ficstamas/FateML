import pandas as pd
import numpy as np
import statsmodels.iolib.summary


def summary2pandas(summary):
    """
    Transforms summary objects from statsmodels to pandas DataFrames
    """
    results_as_html = summary.tables[1].as_html()
    return pd.read_html(results_as_html, header=0, index_col=0)[0]


def feature_importance(summary: statsmodels.iolib.summary.Summary) -> pd.DataFrame:
    """
    Calculates feature importance from statsmodels summary tables
    :param summary:
    :return:
    """
    df = summary2pandas(summary)

    return pd.DataFrame(data={
        "features": df.index,
        "standard error": df['std err'],
        "beta hat": df['coef'],
        "t statistics": df['t'],
        "feature importance": np.abs(df['t'])
    })