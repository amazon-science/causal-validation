# Installation

## Stable version

The latest stable release of `causal-validation` can be installed via `pip`:

``` bash
pip install causal-validation
```

We recommend you check your installation version:

``` bash
python -c 'import causal_validation; print(causal_validation.__version__)'
```

## Development version

!!! warning
    This version is possibly unstable and may contain bugs.


The latest development version of `causal_validation` can be installed via running following:

``` bash
git clone git@github.com:amazon-science/causal-validation.git;
cd causal-validation;
hatch shell create
```

We advise you create virtual environment before installing:

``` bash
conda create -n causal-validation python=3.11.0;
conda activate causal-validation
```

and recommend you check your installation passes the supplied unit tests:

``` bash
hatch run dev:test
```