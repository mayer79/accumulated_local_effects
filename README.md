# Python implementation of Accumulated Local Effects (ALE)

The code is heavily inspired by https://github.com/blent-ai/ALEPython

Differences: 

- Our implementation works only for the easy case of one continuous predictor.
- It supports prediction functions of any dimension, i.e., also probabilitic classification.
- The code is intended to work with numpy, pandas, and polars.
- It supports case weights.
- It returns standard deviations of local effects. This is useful to assess presence/strength of interaction effects. Furthermore, by dividing by root bin size, it provides a measure of estimation accuracy.

See the [example](example.ipynb).