# laby
Physics lab stuff. A small convenience wrapper around `scipy.odr`.
Some graphing code inspired by [huji_lab](https://github.com/stormage2/huji_lab).
## Contents
### Fitting
#### `parse_data(data, y_first=True)`
Parse data formatted in columns. Accepts 1-4 columns of data: (y), (y, x), (y, y_error, x), (y, y_error, x, x_error).
Can also optionally parse the title line as names for the samples, and extracts units from brackets.
Returns a `Data` instance.

#### `*Model`
Generic models used for fitting, e.g Linear, Quadratic. 

`model.fcn(a, x)` is the model function (accepts `a` - the vector of parameters).

`model.fit(data, guess=None, simple=False` - Tries to fit the data and return an Output object. If an initial guess is not provided, it is assumed to be zero. If `simple` is set to True, errors are ignored and a simple least squares fit is attempted.

#### Output
`output.pprint()` - Prints the fit values.

`output.plot(...)` - Plot the fit graph.

### Miscellaneous
#### `fmt_err(x, err, sig=2, exp_display=5)`
Format a value with an uncertainty. Returns the values formatted as e.g "0.0434 +-0.0022", with `sig` determining how many significant digits of error value are included, and `exp_display` determines the minimum (negative) exponent for displaying the value in scientific notation (e.g 34e-10).

## Usage example
```python
>>> from laby import parse_data, QuadraticModel
>>> data = parse_data("""v[m/s]	sv	h[m]	sh
1.076666667	0.005317059 0.349	0.000408248	
1.008333333	0.004250523 0.305	0.000408248	
0.941166667	0.002750152 0.265	0.000408248	
0.866333333	0.002514182 0.225	0.000408248	
0.789333333	0.002048132 0.185	0.000408248	
0.697	0.004140694 0.145	0.000408248	
0.589166667	0.002238551 0.105	0.000408248	
0.4585	0.002203412 0.065	0.000408248""", y_first=False)
>>> output = QuadraticModel.fit(data)
>>> output.pprint()
```
```
Fit parameters:
  a1 = 0.3241 +-0.0087
  a2 = -0.039 +-0.012
  a3 = 0.0151 +-0.0040
X^2 = 1.7
X^2_reduced = 0.34 (< 1)
p_value = 0.89 (OK)
```
```python
>>> output.plot()
```
TODO Image goes here...