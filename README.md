# Python implementation of Accumulated Local Effects (ALE)

The code is heavily inspired by https://github.com/blent-ai/ALEPython

Differences: 

- Our implementation works only for the easy case of one continuous predictor.
- It supports prediction functions of any dimension, i.e., also probabilitic classification.
- The code is intended to work with numpy, pandas, and polars.
- It supports case weights.
- It returns standard deviations of local effects. This is useful to assess presence/strength of interaction effects. Furthermore, by dividing by root bin size, it provides a measure of estimation accuracy.

See the [example](example.ipynb) with this code:

## Example

```py
import matplotlib.pyplot as plt
import numpy as np
from ale import accumulated_local_effects
from sklearn.datasets import load_diabetes
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

# Load data and fit model
X, y = load_diabetes(return_X_y=True, as_frame=True)

est = HistGradientBoostingRegressor(max_iter=50, max_depth=4).fit(X, y)

# Get the top 4 most important features
m = 4
imp = permutation_importance(est, X, y, random_state=0)
top_m = X.columns[np.argsort(imp.importances_mean)[-m:]]

# Plot results for top 4 predictors
fig, axes = plt.subplots(2, 2, figsize=(6, 6))

for feature, ax in zip(top_m, axes.flatten()):
    ale = accumulated_local_effects(X, est.predict, feature=feature)
    ax.plot(ale["breaks"], ale["ale"], "b-", label="ALE")
    standard_errors = ale["standard_deviations"] / np.sqrt(ale["bin_sizes"])
    upper, lower = (ale["ale"] + ind * standard_errors for ind in (1, -1))
    ax.fill_between(ale["breaks"], lower, upper, alpha=0.2, color="blue", label="±1 se")
    ax.set_title(f"{feature}")

    if feature == top_m[0]:
        ax.legend()

plt.tight_layout()
plt.show()
```

![image](example.png)