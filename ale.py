# Credits to https://github.com/blent-ai/ALEPython
import numpy as np
from sklearn.utils._indexing import _get_column_indices, _safe_assign, _safe_indexing


def accumulated_local_effects(
    X,
    predict_function,
    feature,
    breaks=19,
    trim=(0.01, 0.99),
    center=True,
    weights=None,
):
    """Accumulated Local Effects (ALE)

    Calculates accumulated local effects for a continuous feature,
    see Apley & Zhu (2016). The resulting values are mean centered by default to 0.

    Parameters
    ----------
    X : Array-like or DataFrame
        The input data for which to compute the ALE.
    predict_function : callable
        A function that takes a DataFrame and returns numeric predictions
        of any dimension.
    feature : str or int
        The position or name of the continuous feature for which to compute the ALE.
    breaks : int or array-like, optional
        If an integer, specifies the number of quantiles to use for breaks.
        If an array-like, specifies the break points directly.
    trim : tuple of float
        A tuple of two floats specifying the lower and upper quantiles to trim the data.
        Only used if `breaks` is an integer. Default is (0.01, 0.99).
    center : bool, optional
        If True, the ALE is centered by subtracting the average effect.
    sample_weights : array-like or None, optional
        Optional sample weights for the observations in X. Note that standard deviations
        use ddof = 0 in the presence of weights.

    Returns
    -------
    Dictionary, with these components.
        breaks : np.ndarray
            The break points used for the feature.
        ale : np.ndarray
            The accumulated local effects for the feature.
        bin_sizes : np.ndarray
            The number of observations (or the weight sum) of observations falling
            between two breaks (right closed). The first value is 0.
        standard_deviations : np.ndarray
            The standard deviations of the local effects within each bin.
            The first value is NaN.

    References
    ----------
    .. [1] Apley, D. W., & Zhu, J. (2016). Visualizing the Effects of Predictor
           Variables in Black Box Supervised Learning Models. arXiv:1612.08468.
           https://arxiv.org/abs/1612.08468

    Notes
    -----

    Additionally, the function returns the standard deviations of the local effects
    within each bin.
    This helps to assess (A) presence of interactions, and (B) the uncertainty
    of the ALE estimate (divide by square root of corresponding bin sizes
    to get standard errors).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.compose import ColumnTransformer
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import SplineTransformer
    >>>
    >>> rng = np.random.default_rng(1)
    >>>
    >>> age = rng.uniform(0, 40, size=1000)
    >>> living_area = rng.uniform(40, 150, size=1000)
    >>> X = pd.DataFrame({"age": age, "living_area": living_area})
    >>>
    >>> rent = 20 * living_area - 20 * age + (age - 20) ** 2 + rng.normal(0, 10, size=1000)
    >>>
    >>> preprocess_glm = ColumnTransformer(
    ...     transformers=[
    ...         ("spline", SplineTransformer(include_bias=False, knots="quantile"), ["age"]),
    ...         ("linear", "passthrough", ["living_area"]),
    ...     ],
    ...     verbose_feature_names_out=False,
    ... ).set_output(transform="pandas")
    >>> model = Pipeline(
    ...     steps=[
    ...         ("preprocessor", preprocess_glm),
    ...         ("model", LinearRegression()),
    ...     ]
    ... )
    >>> model.fit(X, y=rent)
    >>>
    >>> fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    >>>
    >>> for feature, ax in zip(("age", "living_area"), axes):
    ...     ale = accumulated_local_effects(X, model.predict, feature=feature)
    ...     ax.plot(ale["breaks"], ale["ale"])
    ...     ax.set_title(f"{feature}")
    >>> plt.show()
    """
    has_weights = weights is not None

    if hasattr(X, "copy") and callable(X.copy):
        X = X.copy()

    # Turn feature name into column index
    if isinstance(feature, str):
        feature = _get_column_indices(X, feature)[0]

    x = _safe_indexing(X, feature, axis=1)
    x = np.asarray(x).ravel()  # Ensure x is a 1D numpy array

    bad = np.isnan(x)
    if bad.any():
        X = _safe_indexing(X, ~bad, axis=0)
        x = _safe_indexing(x, ~bad, axis=0)
        if has_weights:
            weights = _safe_indexing(weights, ~bad, axis=0)

    # Prepare breaks
    if isinstance(breaks, int):
        r = np.linspace(*trim, num=breaks, endpoint=True)
        breaks = np.quantile(x, r)
    breaks = np.unique(breaks)
    n_breaks = len(breaks)
    n_bins = n_breaks - 1

    # Get bin IDs from 0 to n_bins - 1
    # Values outside the range are placed in the first or last bin
    bin_ids = np.searchsorted(breaks, x).clip(1, n_bins) - 1

    # Calculate prediction differences
    preds = []
    for j in range(2):
        _safe_assign(X, breaks[bin_ids + j], column_indexer=feature)
        preds.append(predict_function(X))
    ind_local_effects = preds[1] - preds[0]

    # Aggregate per bin_id
    bin_sizes = np.zeros(n_breaks)
    effects = np.zeros((n_breaks, *ind_local_effects.shape[1:]))
    standard_deviations = np.full_like(effects, fill_value=np.nan)

    # Grouped aggregation in numpy
    for i in np.arange(n_bins):
        mask = bin_ids == i
        if mask.any():
            y = ind_local_effects[mask]
            w = weights[mask] if has_weights else None
            bin_sizes[i + 1] = np.sum(w) if has_weights else np.count_nonzero(mask)
            effects[i + 1] = np.average(y, weights=w, axis=0)
            if not has_weights:
                standard_deviations[i + 1] = np.std(y, axis=0)
            else:
                variance = np.average((y - effects[i + 1]) ** 2, weights=w, axis=0)
                standard_deviations[i + 1] = np.sqrt(variance)

    ale = effects.cumsum(axis=0)

    if center:
        avg_ale_per_bin = (ale[:-1] + ale[1:]) / 2
        ale -= np.average(avg_ale_per_bin, weights=bin_sizes[1:], axis=0)

    if len(ale.shape) == 1:
        ale = ale.flatten()
        standard_deviations = standard_deviations.flatten()

    return {
        "breaks": breaks,
        "ale": ale,
        "bin_sizes": bin_sizes,
        "standard_deviations": standard_deviations,
    }
