import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import PoissonRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def plot_poisson_regressor_demo(
    df: pd.DataFrame,
    target: str,
    feature_x: str,
    group: str | None = None,
    alphas=(0.0, 0.5, 5.0),
    *,
    n_grid: int = 300,
    dropna: bool = True,
    max_groups: int = 6,
    add_coef_shrinkage_plot: bool = True,
    random_state: int = 0,
):
    """
    Fit PoissonRegressor on an arbitrary pandas DataFrame and produce:
      1) Scatter of observed counts vs feature_x (optionally colored by group)
      2) Predicted mean curves across feature_x for each alpha
         - If group is provided, draw one curve per group level (up to max_groups)
      3) Optional coefficient shrinkage plot vs alpha (in transformed feature space)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target : str
        Name of count target column (non-negative integers are ideal).
    feature_x : str
        Feature to sweep on the x-axis for partial dependence-style curves.
    group : str | None
        Optional grouping feature. If provided, curves are drawn per group level.
        Can be categorical or numeric; numeric will be treated as categorical levels.
    alphas : iterable[float]
        L2 regularization strengths to compare.
    n_grid : int
        Number of points in the sweep grid for feature_x.
    dropna : bool
        Drop rows with NA in selected columns.
    max_groups : int
        Max number of group levels to plot (most frequent levels kept).
    add_coef_shrinkage_plot : bool
        Whether to plot coefficient magnitude vs alpha (transformed space).
    random_state : int
        Used only for reproducible subsampling if needed in the future.

    Returns
    -------
    dict
        Contains fitted pipelines and metadata.
    """
    if target not in df.columns:
        raise ValueError(f"target='{target}' not found in df columns.")
    if feature_x not in df.columns:
        raise ValueError(f"feature_x='{feature_x}' not found in df columns.")
    if group is not None and group not in df.columns:
        raise ValueError(f"group='{group}' not found in df columns.")

    # Keep only needed columns for modeling
    cols = [target, feature_x] + ([group] if group is not None else [])
    d = df[cols].copy()

    if dropna:
        d = d.dropna()

    # Target checks (PoissonRegressor expects non-negative y; counts are best)
    y = d[target].to_numpy()
    if np.any(y < 0):
        raise ValueError("Target contains negative values; PoissonRegressor requires y >= 0.")

    # Build feature matrix using all columns except target
    feature_cols = [c for c in d.columns if c != target]
    X = d[feature_cols]

    # Identify numeric vs categorical columns
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    # Preprocess:
    # - numeric: standardize (helpful for L2 regularization)
    # - categorical: one-hot
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=True, with_std=True), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Fit one model per alpha
    pipes = []
    for a in alphas:
        pipe = Pipeline(
            steps=[
                ("pre", pre),
                ("poisson", PoissonRegressor(alpha=float(a), max_iter=1000)),
            ]
        )
        pipe.fit(X, y)
        pipes.append(pipe)

    # ---- Build sweep grid for feature_x ----
    x_vals = d[feature_x].to_numpy()
    if not np.issubdtype(d[feature_x].dtype, np.number):
        raise ValueError(
            f"feature_x='{feature_x}' is non-numeric. "
            "For a sweep plot it must be numeric."
        )
    x_grid = np.linspace(np.nanmin(x_vals), np.nanmax(x_vals), n_grid)

    # Choose "baseline" values for other features when sweeping:
    # - numeric: median
    # - categorical: mode
    baselines = {}
    for c in feature_cols:
        if c == feature_x:
            continue
        if c in numeric_cols:
            baselines[c] = float(d[c].median())
        else:
            baselines[c] = d[c].mode(dropna=True).iloc[0] if len(d[c].mode(dropna=True)) else d[c].iloc[0]

    # If group is provided, decide which group levels to plot
    group_levels = [None]
    if group is not None:
        # Treat group as categorical for plotting levels
        vc = d[group].astype("object").value_counts(dropna=False)
        group_levels = vc.index.tolist()[:max_groups]

    # Helper to build a grid DataFrame for predictions
    def make_pred_df(group_level):
        pred_dict = {}
        for c in feature_cols:
            if c == feature_x:
                pred_dict[c] = x_grid
            elif group is not None and c == group:
                pred_dict[c] = np.array([group_level] * len(x_grid), dtype=object)
            else:
                pred_dict[c] = np.array([baselines[c]] * len(x_grid))
        return pd.DataFrame(pred_dict)

    # ---- Plot 1: Data + fitted mean curves ----
    plt.figure(figsize=(11, 6))

    # Scatter data (optionally by group)
    if group is None:
        plt.scatter(d[feature_x], d[target], s=18, alpha=0.5, label="Observed counts")
    else:
        # Plot only the top group levels (others omitted for clarity)
        dg = d.copy()
        dg[group] = dg[group].astype("object")
        keep = set(group_levels)
        for lvl in group_levels:
            mask = (dg[group] == lvl)
            plt.scatter(
                dg.loc[mask, feature_x],
                dg.loc[mask, target],
                s=18,
                alpha=0.5,
                label=f"Observed (group={lvl})",
            )

    # Predicted curves
    for a, pipe in zip(alphas, pipes):
        for lvl in group_levels:
            pred_df = make_pred_df(lvl)
            lam_hat = pipe.predict(pred_df)
            if group is None:
                lbl = f"Predicted mean (alpha={a})"
            else:
                lbl = f"Predicted mean (alpha={a}, {group}={lvl})"
            plt.plot(x_grid, lam_hat, linestyle="--", linewidth=2, label=lbl)

    title = "PoissonRegressor partial-dependence-style sweep"
    subtitle = rf"Sweep: {feature_x}   |   Target: {target}"
    if group is not None:
        subtitle += rf"   |   Curves by: {group} (top {len(group_levels)} levels)"
    plt.title(title + "\n" + subtitle)

    plt.xlabel(feature_x)
    plt.ylabel(f"{target} / Predicted mean λ")
    plt.grid(True, alpha=0.25)

    # Avoid an unreadable legend if many curves
    # If too many entries, put legend outside
    n_entries = (1 if group is None else len(group_levels)) * len(alphas) + (1 if group is None else len(group_levels))
    if n_entries > 10:
        plt.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
        plt.tight_layout(rect=[0, 0, 0.78, 1])
    else:
        plt.legend(fontsize=9, loc="best")
        plt.tight_layout()
    plt.show()

    # ---- Plot 2: Coefficient shrinkage vs alpha (transformed space) ----
    shrinkage = None
    if add_coef_shrinkage_plot:
        # Get feature names after preprocessing
        feature_names = pipes[0].named_steps["pre"].get_feature_names_out()
        coefs = np.vstack([p.named_steps["poisson"].coef_ for p in pipes])  # shape: (len(alphas), n_features)
        l2norm = np.linalg.norm(coefs, axis=1)

        # Also track coefficients associated with feature_x (could be multiple if one-hot, etc.)
        # We'll show the sum of absolute values for any transformed feature that contains feature_x in its name.
        fx_mask = np.array([feature_x in name for name in feature_names])
        fx_abs_sum = np.sum(np.abs(coefs[:, fx_mask]), axis=1) if np.any(fx_mask) else None

        plt.figure(figsize=(9, 4.8))
        plt.plot(alphas, l2norm, marker="o", label="||coef||₂ (all transformed features)")
        if fx_abs_sum is not None:
            plt.plot(alphas, fx_abs_sum, marker="o", label=f"sum|coef| for '{feature_x}' features")

        plt.xscale("symlog", linthresh=0.1)  # keeps alpha=0 visible while showing large alphas
        plt.xlabel("alpha (L2 regularization strength)")
        plt.ylabel("Coefficient magnitude (transformed space)")
        plt.title("Regularization effect: coefficients shrink as alpha increases")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.show()

        shrinkage = {
            "feature_names": feature_names,
            "coefs": coefs,
            "l2norm": l2norm,
            "feature_x_abs_sum": fx_abs_sum,
            "alphas": np.array(alphas, dtype=float),
        }

    return {
        "pipelines": pipes,
        "alphas": tuple(float(a) for a in alphas),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "baselines": baselines,
        "group_levels": group_levels,
        "shrinkage": shrinkage,
    }