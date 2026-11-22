# dicer

![dicer logo](https://raw.githubusercontent.com/Ajoo/dicer/main/logo.png)

**Preprocessing that remembers what your data means.**

dicer is a Python library for disciplined data preprocessing. Every column carries
a *domain*: its bounds, its precision, and its set of possible values. Every
preprocessing step is a transformation between domains. dicer can also take a probability distribution that a model
predicts in *transformed* space and map it back to the original units, with
the probability mass accounted for correctly.

## At a glance

Train a plain classifier on 32 quantile bins of income, then get a full
predictive **density over dollars** for every row. The density is tail-aware,
so you can sample from it and score it against held-out data.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from dicer import FrameTracker
from dicer.stats import FiniteSupportDistribution

# Toy data: income (dollars and cents) depends on city and occupancy.
rng = np.random.default_rng(0)
n = 4000
city = rng.choice(["lisbon", "porto", "faro"], n)
occupancy = rng.beta(4, 2, n)
log_mu = 10.2 + 0.4 * (city == "lisbon") + 0.8 * occupancy
df = pd.DataFrame({
    "income": np.round(rng.lognormal(log_mu, 0.5), 2),
    "occupancy": occupancy,
    "city": city,
})
train, test = df.iloc[:3000], df.iloc[3000:]

# 1. Fit a preprocessing pipeline: declare what you know, learn what you don't.
fit = (
    FrameTracker.from_dataframe(train)
    .assume["income"].nonnegative()
    .infer["income"].fixed_point()                       # detects cents: 2 decimals
    .infer["income"].quantile_discretize(32, right_tail_distribution="genpareto")
    .assume["occupancy"].ge(0.0)
    .assume["occupancy"].le(1.0)
    .transform["occupancy"].logit()
    .assume["city"].has_support(["lisbon", "porto", "faro"])
    .transform["city"].one_hot_encode()
    .finish()
)
pipeline = fit.transformation

# 2. Replay it on new data. Nothing is refit; validation is explicit.
pipeline.validate(test)
test_t = pipeline.apply(test)

# 3. Any classifier over the 32 income bins...
features = ["occupancy", "city.0", "city.1", "city.2"]
clf = LogisticRegression(max_iter=1000).fit(fit.data[features], fit.data["income"])
probs = clf.predict_proba(test_t[features])                     # shape (1000, 32)

# ...becomes a per-row probability density over the original income scale.
bins = FiniteSupportDistribution(pipeline.output_domain["income"], probs)
income = pipeline["income"].pullback_distribution(bins, fixed_point=False)

income.logpdf(test["income"].to_numpy()).mean()   # -11.67  (true generating model: -11.62)
income.ppf(0.5)[:3]                               # per-row medians: [66459, 54065, 50060]
income.ppf(0.95)[:3]                              # [149542, 124287, 110704]
income.rvs(random_state=0)                        # one plausible income per row, in dollars
```

The transformed training frame is ordinary model input:

```text
   income  occupancy  city.0  city.1  city.2
0      31   2.056405       0       0       1
1      10   0.943785       0       1       0
2       2   1.635333       0       1       0
```

In this example:

- **One pipeline goes both ways.** The same fitted object turns raw data into
  features and turns predictions back into distributions over raw values.
- **The likelihoods are in the original units.** A 32-bin classifier scores
  −11.67 nats per row on held-out incomes, close to the true data-generating density's −11.62.
  Because every score is a density over dollars, you can compare models that used completely different
  preprocessing.
- **Discretization is reversible in distribution.** Within each bin, mass is spread
  back out across the bin. In the top bin, mass follows a generalized Pareto
  tail fitted to the training data, so the density does not stop at the
  largest income seen in training.
- **The data's precision is part of the model.** dicer noticed that incomes are recorded
  in cents. By default the pulled-back distribution puts its mass on that
  0.01 grid. `fixed_point=False` asks for the continuous density instead.

## Installation

```bash
pip install dicerpy
```

The package is published as `dicerpy` but imported as `dicer`.

To install the latest development version instead:

```bash
pip install git+https://github.com/Ajoo/dicer.git
```


## Philosophy

Most preprocessing libraries treat a pipeline as a function from arrays to
arrays. dicer treats it as a map between **domains**, the sets of values a column
can take. That changes three things.

**1. What a column can be is stated explicitly.** A float64 column could hold any real
number. Your income column cannot be negative and is recorded in cents. dicer
starts from the weakest domain the dtype allows. It only narrows that domain when you
say so (`assume`) or when you ask it to learn from data (`infer`). An observed
minimum is evidence for a bound, not a guarantee. dicer makes you choose which
one you mean.

**2. Steps are transformations between domains, not just functions.** Each step knows its
input and output domains. Where it can, it also knows its inverse and how probability mass
moves through it:

| Kind of step | Examples | Inverse |
|---|---|---|
| Invertible | `affine`, `log`, `logit`, `probit`, `tanh`, piecewise affine | exact, with Jacobian |
| Lossy | `digitize`, `round`, `clip`, `to_int` | **stochastic**: reconstruct a value from a distribution over its preimage |
| Recoding | `encode`, `one_hot_encode`, `to_categorical` | exact relabelling |

Lossy steps are where dicer differs most from ordinary pipelines. Quantile binning
throws information away. Instead of pretending it doesn't, dicer gives every
bin a distribution for "where inside the bin." The tails can have parametric
shapes (uniform, normal, generalized Pareto). Pulling back through a lossy
step gives a proper distribution, not a point estimate.

**3. Fitting and applying are separate, and fitting happens once.** A pipeline is
fitted on training data with a fluent tracker, then frozen into an immutable
transformation. That transformation is replayed on validation or test data
without re-estimating anything. Checking new data against the fitted domains is
a separate, explicit call.

The fluent API has three verbs:

- **`assume`** narrows a domain without changing any values. Examples are bounds, finite
  precision and categorical support. Assumptions are checked against the tracked
  data. Use them for things you believe hold *beyond* the training set.
- **`transform`** changes values and their domain, using parameters you supply,
  such as cut points, clip bounds, a scale or an encoding.
- **`infer`** learns an assumption or a transformation's parameters from the
  training data. Examples are bounds, precision, standardization, quantile bins and tail
  shapes. Once fitted, the result is fixed.

## A walkthrough

This section takes the opening example apart step by step.

### Starting from a DataFrame

```python
tracker = FrameTracker.from_dataframe(train)
```

`from_dataframe` copies the frame and infers a minimal domain for every column
from its dtype. Here that is `float: (-inf, inf)` for both numeric columns and a
generic object domain for `city`. Nothing has been assumed yet.

If you have no data, `FrameTracker.from_dtypes({...})` builds the same
structure without data. You can then reason about domains and transformations
abstractly.

### Selecting columns

Each verb can be scoped to a subset of columns:

```python
tracker.assume["income"]                     # one column by name
tracker.infer(["Slope", "Distance_*"])       # names and shell-style globs
tracker.infer(continuous=True)               # columns whose domain is continuous
tracker.transform(bounded=True, max_decimals=0)
```

The available property filters are `continuous`, `ordered`, `bounded`,
`bounded_left`, `bounded_right`, `finite`, `min_cardinality`/`max_cardinality`
and `min_decimals`/`max_decimals`. They match on the column's *current* domain.
This lets you write rules like "convert every column with no decimal places to integers"
after an inference step has found out which columns those are.

### Assumptions

```python
.assume["income"].nonnegative()
.assume["occupancy"].ge(0.0)
.assume["occupancy"].le(1.0)
.assume["city"].has_support(["lisbon", "porto", "faro"])
```

Each assumption returns a new tracker with a narrower domain. Assumptions are
validated against the tracked data immediately. If you assume
`occupancy <= 0.5` and the training data disagrees, you get an error at that
line, not a silent clip. `has_support` turns the open-ended object
column into a finite, unordered domain with exactly three values.

### Inference

```python
.infer["income"].fixed_point()
.infer["income"].quantile_discretize(32, right_tail_distribution="genpareto")
```

`fixed_point()` looks for the smallest number of decimal places that explains
the data. Here it finds 2, and `income` becomes a fixed-point domain. From then on,
validation rejects values like `1.234`, and pulled-back distributions put their mass on the
0.01 grid.

`quantile_discretize(32, ...)` puts the interior bin edges at empirical
quantiles. It then chooses a reconstruction distribution for each end:

- The **left** side is bounded at 0 by the earlier assumption, so the default is a
  uniform tail.
- The **right** side is unbounded, and here we ask for a generalized Pareto tail.
  It is fitted to the training data and gets at least 5% of the probability mass
  by default (`min_prob_right_tail`).

The output is an integer column in `[0..31]`. The fitted `Digitize`
transformation keeps the bin edges and the tail models.

Other inferences include `lb`/`ub` (empirical or Pareto-extrapolated bounds),
`standardize`, `min_max_scale`, `clip`, `to_quantiles`, and categorical encoders
such as `frequency_encode` and `conditional_rates` (target-rate encoding).

### Transformations

```python
.transform["occupancy"].logit()
.transform["city"].one_hot_encode()
```

`logit` needs its input inside `[0, 1]`. Because of the assumptions above, the domain
already is, so the transformation is valid by construction. It maps the
domain to `(-inf, inf)`. `one_hot_encode` turns the
logical column `city` into three physical columns, `city.0`, `city.1` and
`city.2`, in the order of the declared support.

To derive a new column and keep the original, use `into=`:

```python
.infer("income", into="income_bin").quantile_discretize(8)
```

### Finishing

```python
fit = tracker.finish()
fit.data            # the transformed training DataFrame
fit.transformation  # an immutable, replayable FrameTransformation
```

`FitResult` is a named tuple, so `data, pipeline = tracker.finish()` also works.

Frame trackers are *linear*. Each operation consumes the tracker it was called
on and returns a new one. This lets dicer transform its private copy of the
training data in place without surprises. If you reuse an old tracker, you get a
clear `RuntimeError` instead of a silently inconsistent branch. Chaining, as in the
examples above, is the intended style.

### Replaying on new data

```python
pipeline.validate(test)          # raises if test violates the fitted domains
test_t = pipeline.apply(test)    # never mutates `test`
```

Validation is a separate call, so you decide when to pay for it. Errors name
the column and the broken assumption:

```text
ValueError: Column 'city' does not satisfy unordered finite: {'lisbon', 'porto', 'faro'}:
            Values must belong to the finite domain.
ValueError: Column 'income' does not satisfy fixed-point (2 digits): (0.0, inf):
            Values must have at most 2 decimal places.
```

`apply` keeps the index and declared column order. Steps with randomness, such as
dequantization or stochastic inverses, accept a `random_state`.

You can inspect the fitted domains at either end:

```python
pipeline.input_domain
# income                  fixed-point (2 digits): (0.0, inf)
# occupancy                                float: (0.0, 1.0)
# city         unordered finite: {'lisbon', 'porto', 'faro'}

pipeline.output_domain
# income             int: [0..31]
# occupancy    float: (-inf, inf)
# city                 one-hot[3]
```

### Moving distributions through the pipeline

Indexing a fitted pipeline by output column gives that column's composed
transformation, from the raw input to the final output:

```python
pipeline["income"]       # the fitted Digitize (bin edges + tail models)
pipeline["occupancy"]    # the logit map
```

Every transformation supports two operations on distributions:

- **`pushforward_distribution(p)`** takes a distribution over *inputs* and returns the
  distribution of the *outputs*.
- **`pullback_distribution(q)`** takes a distribution over *outputs* and returns a distribution
  over *inputs*. For lossy steps this uses the stochastic inverse.

Distributions can be SciPy frozen distributions or dicer's own types, and both can be batched:

```python
from scipy.stats import lognorm, norm

# A Gaussian predicted in logit space becomes a density on (0, 1).
occ = pipeline["occupancy"].pullback_distribution(norm(0.7, 0.8))
occ.pdf([0.25, 0.5, 0.75])    # [0.212, 1.360, 2.349]
occ.cdf(0.5)                  # 0.191

# A continuous belief about income becomes probabilities over the 32 bins.
prior = pipeline["income"].pushforward_distribution(lognorm(s=0.5, scale=60_000))
prior.probabilities           # array of 32 bin probabilities, summing to 1

# Batched categorical predictions -> batched densities (one per row).
bins = FiniteSupportDistribution(pipeline.output_domain["income"], probs)  # (n, 32)
income = pipeline["income"].pullback_distribution(bins, fixed_point=False)
```

Pulled-back distributions support `pdf`/`logpdf`, `pmf`, `cdf`, `ppf`, `rvs`
and `mean`. Mass is conserved exactly. For a fixed-point column, the default
(lattice) pullback and the `fixed_point=False` (continuous) pullback differ
by the grid spacing: the `pmf` of the first equals the `pdf` of the second
times 0.01.

To reconstruct individual values rather than whole distributions, use a lossy step's
stochastic inverse directly:

```python
pipeline["income"].inverse.apply(np.array([0, 15, 31]), random_state=0)
# array([ 11939.71,  49731.26, 142877.5 ])   # one draw per code, on the cents grid
```

### Single columns and bare domains

Everything above also works on one array with `ColumnTracker`:

```python
from dicer import ColumnTracker

fit = (
    ColumnTracker.from_series(train["occupancy"])
    .assume.ge(0.0)
    .assume.le(1.0)
    .infer.standardize()
    .finish()
)
```

It also works on domains with no data at all, if you only want to work with
transformations and distributions:

```python
from dicer import Column

domain, logit = (
    Column.from_dtype(np.float64)
    .assume.ge(0.0)
    .assume.le(1.0)
    .transform.logit()
    .unpack()
)
logit.pullback_distribution(norm(0, 3))   # a logit-normal on (0, 1)
```

## What's available

Methods are registered per domain. A column only offers the operations that make
sense for its current domain.

| Domain | `assume` | `transform` | `infer` |
|---|---|---|---|
| Floating point | `ge`, `le`, `nonnegative`, `nonpositive`, `fixed_point` | `affine`, `map`, `log`, `exp`, `logit`, `expit`, `probit`, `invprobit`, `softsign`, `tanh`, `atanh`, `piecewise_affine`, `generalized_piecewise_affine`, `clip`, `round`, `digitize` | `nonnegative`, `lb`, `ub`, `fixed_point`, `standardize`, `min_max_scale`, `clip`, `to_quantiles`, `quantile_discretize` |
| Fixed point | as floating point | as floating point, plus `to_int`, `dequantize` | as floating point |
| Integer | `ge`, `gt`, `le`, `lt`, `nonnegative`, `nonpositive` | `affine`, `clip`, `digitize`, `encode`, `to_fixed_point`, `as_unordered` | `nonnegative`, `lb`, `ub`, `clip`, `quantile_discretize` |
| Generic (object) | `has_support` | — | `to_categorical` |
| Finite unordered | — | `encode`, `one_hot_encode`, `to_categorical` | `conditional_rates` |
| Unordered integer codes | — | as finite unordered, plus `permute`, `as_ordered` | `frequency_encode`, `rank_by_frequency`, `conditional_rates` |
| pandas categorical | — | `encode`, `one_hot_encode` | — |

At the frame level, `assume(...).one_hot()`, `.simplex()` and `.vector()` group
several physical columns into one logical column, for example a dataset that
arrives already one-hot encoded:

```python
.assume("Soil_Type_*", into="Soil_Type").one_hot()
```

## Status

dicer is **pre-alpha**. The API may still change. Known limitations:

- Distributions are handled **per column**. There is no joint or multivariate
  distribution over a whole frame yet.
- Not every transformation supports every distribution operation. Unsupported
  combinations raise `NotImplementedError`.
- Missing values are not modelled. Clean or impute them before using dicer.

## Development

```bash
git clone https://github.com/Ajoo/dicer.git
cd dicer
pip install -e ".[test]"
pytest tests
```

## License

MIT. See [LICENSE](https://github.com/Ajoo/dicer/blob/main/LICENSE).
