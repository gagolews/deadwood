# kneedle: Knee/Elbow Point Detection

## Description

Finds the most significant knee/elbow using the Kneedle algorithm with exponential smoothing.

## Usage

``` r
kneedle_increasing(x, convex = TRUE, dt = 0.01)
```

## Arguments

|  |  |
|----|----|
| `x` | data vector (increasing) |
| `convex` | whether the data in `x` are convex-ish (elbow detection) or not (knee lookup) |
| `dt` | controls the smoothing parameter $\alpha = 1-\exp(-dt)$ of the exponential moving average, $y_i = \alpha x_i + (1-\alpha) y_{i-1}$, $y_1 = x_1$ |

## Value

Returns the index of the knee/elbow point; 1 if not found.

## Author(s)

[Marek Gagolewski](https://www.gagolewski.com/)

## References

V. Satopaa, J. Albrecht, D. Irwin, B. Raghavan, Finding a \"Kneedle\" in a haystack: Detecting knee points in system behavior, In: 31st Intl. Conf. Distributed Computing Systems Workshops, 2011, 166-171, [doi:10.1109/ICDCSW.2011.20](https://doi.org/10.1109/ICDCSW.2011.20)

## See Also

The official online manual of <span class="pkg">deadwood</span> at <https://deadwood.gagolewski.com/>
