use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const N_STATS: usize = 12;
const N_UNARY_STATS: usize = 10;
const N_PAIR_STATS: usize = 10;
const N_LAGGED_PAIR_STATS: usize = 4;

#[pyfunction]
fn daily_return_stats<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    day_offsets: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let close = close.as_array();
    let offsets = day_offsets.as_slice()?;
    if offsets.len() < 2 || offsets[0] != 0 || offsets[offsets.len() - 1] != close.nrows() as i64 {
        return Err(PyValueError::new_err(
            "day_offsets must start at zero and end at close.shape[0]",
        ));
    }
    if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(PyValueError::new_err("day_offsets must be monotonic"));
    }

    let n_days = offsets.len() - 1;
    let n_symbols = close.ncols();
    let mut output = Array3::from_elem((n_days, n_symbols, N_STATS), f64::NAN);

    for day in 0..n_days {
        let start = offsets[day] as usize;
        let end = offsets[day + 1] as usize;
        for symbol in 0..n_symbols {
            // Match DataFrame.pct_change(fill_method=None): only adjacent rows
            // form a return, while the first row of a later day still compares
            // with the immediately preceding row from the prior day.
            let mut previous = if start > 0 {
                close[[start - 1, symbol]]
            } else {
                f64::NAN
            };
            let mut count = 0usize;
            let mut sum = 0.0;
            let mut abs_sum = 0.0;
            let mut max_return = f64::NEG_INFINITY;
            let mut positive = 0usize;
            let mut negative = 0usize;
            let mut raw_m2 = 0.0;
            let mut raw_m4 = 0.0;
            for row in start..end {
                let value = close[[row, symbol]];
                if !value.is_nan() && !previous.is_nan() {
                    let ret = value / previous - 1.0;
                    if !ret.is_nan() {
                        count += 1;
                        sum += ret;
                        abs_sum += ret.abs();
                        max_return = max_return.max(ret);
                        positive += usize::from(ret > 0.0);
                        negative += usize::from(ret < 0.0);
                        let square = ret * ret;
                        raw_m2 += square;
                        raw_m4 += square * square;
                    }
                }
                previous = value;
            }
            if count == 0 {
                continue;
            }
            let mean = sum / count as f64;
            let mut m2 = 0.0;
            let mut m3 = 0.0;
            let mut m4 = 0.0;
            previous = if start > 0 {
                close[[start - 1, symbol]]
            } else {
                f64::NAN
            };
            for row in start..end {
                let value = close[[row, symbol]];
                if !value.is_nan() && !previous.is_nan() {
                    let ret = value / previous - 1.0;
                    if !ret.is_nan() {
                        let centered = ret - mean;
                        let square = centered * centered;
                        m2 += square;
                        m3 += square * centered;
                        m4 += square * square;
                    }
                }
                previous = value;
            }
            let variance = m2 / count as f64;
            let std = variance.sqrt();
            output[[day, symbol, 0]] = count as f64;
            output[[day, symbol, 1]] = sum;
            output[[day, symbol, 2]] = mean;
            output[[day, symbol, 3]] = std;
            output[[day, symbol, 4]] = abs_sum / count as f64;
            output[[day, symbol, 5]] = max_return;
            output[[day, symbol, 6]] = positive as f64;
            output[[day, symbol, 7]] = negative as f64;
            output[[day, symbol, 8]] = raw_m2 / count as f64;
            output[[day, symbol, 9]] = raw_m4 / count as f64;
            if std > 0.0 && count >= 3 {
                let biased_skew = (m3 / count as f64) / std.powi(3);
                output[[day, symbol, 10]] =
                    ((count * (count - 1)) as f64).sqrt() / (count - 2) as f64 * biased_skew;
                output[[day, symbol, 11]] = (m4 / count as f64) / std.powi(4) - 3.0;
            } else if std == 0.0 && count >= 3 {
                output[[day, symbol, 10]] = 0.0;
                output[[day, symbol, 11]] = 0.0;
            }
        }
    }
    Ok(output.into_pyarray(py))
}

#[pyfunction]
fn daily_unary_stats<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    day_offsets: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let values = values.as_array();
    let offsets = day_offsets.as_slice()?;
    if offsets.len() < 2 || offsets[0] != 0 || offsets[offsets.len() - 1] != values.nrows() as i64 {
        return Err(PyValueError::new_err("invalid day_offsets"));
    }
    let mut output =
        Array3::from_elem((offsets.len() - 1, values.ncols(), N_UNARY_STATS), f64::NAN);
    for day in 0..offsets.len() - 1 {
        let start = offsets[day] as usize;
        let end = offsets[day + 1] as usize;
        for symbol in 0..values.ncols() {
            let observed: Vec<f64> = (start..end)
                .map(|row| values[[row, symbol]])
                .filter(|value| !value.is_nan())
                .collect();
            if observed.is_empty() {
                continue;
            }
            let count = observed.len();
            let sum = observed.iter().sum::<f64>();
            let mean = sum / count as f64;
            let mut m2 = 0.0;
            let mut m3 = 0.0;
            for value in &observed {
                let centered = value - mean;
                m2 += centered * centered;
                m3 += centered * centered * centered;
            }
            let variance = m2 / count as f64;
            let std = variance.sqrt();
            output[[day, symbol, 0]] = count as f64;
            output[[day, symbol, 1]] = sum;
            output[[day, symbol, 2]] = mean;
            output[[day, symbol, 3]] = std;
            if count >= 2 {
                output[[day, symbol, 4]] = (m2 / (count - 1) as f64).sqrt();
            }
            output[[day, symbol, 5]] = observed.iter().copied().fold(f64::INFINITY, f64::min);
            output[[day, symbol, 6]] = observed.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            output[[day, symbol, 7]] = observed[0];
            output[[day, symbol, 8]] = observed[count - 1];
            if count >= 3 {
                output[[day, symbol, 9]] = if std == 0.0 {
                    0.0
                } else {
                    ((count * (count - 1)) as f64).sqrt() / (count - 2) as f64 * (m3 / count as f64)
                        / std.powi(3)
                };
            }
        }
    }
    Ok(output.into_pyarray(py))
}

#[pyfunction]
fn daily_pair_stats<'py>(
    py: Python<'py>,
    left: PyReadonlyArray2<'py, f64>,
    right: PyReadonlyArray2<'py, f64>,
    day_offsets: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let left = left.as_array();
    let right = right.as_array();
    if left.dim() != right.dim() {
        return Err(PyValueError::new_err(
            "pair inputs must have the same shape",
        ));
    }
    let offsets = day_offsets.as_slice()?;
    if offsets.len() < 2 || offsets[0] != 0 || offsets[offsets.len() - 1] != left.nrows() as i64 {
        return Err(PyValueError::new_err("invalid day_offsets"));
    }
    let mut output = Array3::from_elem((offsets.len() - 1, left.ncols(), N_PAIR_STATS), f64::NAN);
    for day in 0..offsets.len() - 1 {
        let start = offsets[day] as usize;
        let end = offsets[day + 1] as usize;
        for symbol in 0..left.ncols() {
            let pairs: Vec<(f64, f64)> = (start..end)
                .map(|row| (left[[row, symbol]], right[[row, symbol]]))
                .filter(|(x, y)| !x.is_nan() && !y.is_nan())
                .collect();
            if pairs.is_empty() {
                continue;
            }
            let count = pairs.len();
            let x_mean = pairs.iter().map(|pair| pair.0).sum::<f64>() / count as f64;
            let y_mean = pairs.iter().map(|pair| pair.1).sum::<f64>() / count as f64;
            let mut x_m2 = 0.0;
            let mut y_m2 = 0.0;
            let mut cross = 0.0;
            let mut product = 0.0;
            for &(x, y) in &pairs {
                let x_centered = x - x_mean;
                let y_centered = y - y_mean;
                x_m2 += x_centered * x_centered;
                y_m2 += y_centered * y_centered;
                cross += x_centered * y_centered;
                product += x * y;
            }
            let x_std = (x_m2 / count as f64).sqrt();
            let y_std = (y_m2 / count as f64).sqrt();
            let covariance = cross / count as f64;
            output[[day, symbol, 0]] = count as f64;
            output[[day, symbol, 1]] = x_mean;
            output[[day, symbol, 2]] = y_mean;
            output[[day, symbol, 3]] = x_std;
            output[[day, symbol, 4]] = y_std;
            output[[day, symbol, 5]] = covariance;
            if x_std > 0.0 && y_std > 0.0 {
                output[[day, symbol, 6]] = covariance / (x_std * y_std);
            }
            if x_m2 > 0.0 {
                let slope = cross / x_m2;
                output[[day, symbol, 7]] = slope;
                output[[day, symbol, 8]] = (pairs
                    .iter()
                    .map(|(x, y)| (y - y_mean - slope * (x - x_mean)).powi(2))
                    .sum::<f64>()
                    / count as f64)
                    .sqrt();
            }
            output[[day, symbol, 9]] = product / count as f64;
        }
    }
    Ok(output.into_pyarray(py))
}

#[pyfunction]
fn daily_lagged_pair_stats<'py>(
    py: Python<'py>,
    left: PyReadonlyArray2<'py, f64>,
    right: PyReadonlyArray2<'py, f64>,
    day_offsets: PyReadonlyArray1<'py, i64>,
    lag: usize,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let left = left.as_array();
    let right = right.as_array();
    if left.dim() != right.dim() || lag == 0 {
        return Err(PyValueError::new_err(
            "pair inputs must have the same shape and lag must be positive",
        ));
    }
    let offsets = day_offsets.as_slice()?;
    if offsets.len() < 2 || offsets[0] != 0 || offsets[offsets.len() - 1] != left.nrows() as i64 {
        return Err(PyValueError::new_err("invalid day_offsets"));
    }
    let mut output = Array3::from_elem(
        (offsets.len() - 1, left.ncols(), N_LAGGED_PAIR_STATS),
        f64::NAN,
    );
    for day in 0..offsets.len() - 1 {
        let start = offsets[day] as usize;
        let end = offsets[day + 1] as usize;
        for symbol in 0..left.ncols() {
            let pairs: Vec<(f64, f64)> = (start..end)
                .map(|row| (left[[row, symbol]], right[[row, symbol]]))
                .filter(|(x, y)| !x.is_nan() && !y.is_nan())
                .collect();
            if pairs.len() <= lag {
                continue;
            }
            let count = pairs.len() - lag;
            let x_mean = pairs[..count].iter().map(|pair| pair.0).sum::<f64>() / count as f64;
            let y_mean = pairs[lag..].iter().map(|pair| pair.1).sum::<f64>() / count as f64;
            let mut x_m2 = 0.0;
            let mut y_m2 = 0.0;
            let mut cross = 0.0;
            for index in 0..count {
                let x = pairs[index].0 - x_mean;
                let y = pairs[index + lag].1 - y_mean;
                x_m2 += x * x;
                y_m2 += y * y;
                cross += x * y;
            }
            output[[day, symbol, 0]] = count as f64;
            output[[day, symbol, 1]] = (x_m2 / count as f64).sqrt();
            output[[day, symbol, 2]] = (y_m2 / count as f64).sqrt();
            if x_m2 > 0.0 && y_m2 > 0.0 {
                output[[day, symbol, 3]] = cross / (x_m2 * y_m2).sqrt();
            }
        }
    }
    Ok(output.into_pyarray(py))
}

#[pyfunction]
fn daily_tail_means<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    day_offsets: PyReadonlyArray1<'py, i64>,
    windows: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let values = values.as_array();
    let offsets = day_offsets.as_slice()?;
    let windows = windows.as_slice()?;
    if offsets.len() < 2 || offsets[0] != 0 || offsets[offsets.len() - 1] != values.nrows() as i64 {
        return Err(PyValueError::new_err("invalid day_offsets"));
    }
    if windows.iter().any(|window| *window < 1) {
        return Err(PyValueError::new_err("windows must be positive"));
    }
    let mut output = Array3::from_elem(
        (offsets.len() - 1, values.ncols(), windows.len() + 2),
        f64::NAN,
    );
    for day in 0..offsets.len() - 1 {
        let start = offsets[day] as usize;
        let end = offsets[day + 1] as usize;
        for symbol in 0..values.ncols() {
            let observed: Vec<f64> = (start..end)
                .map(|row| values[[row, symbol]])
                .filter(|value| !value.is_nan())
                .collect();
            if observed.is_empty() {
                continue;
            }
            output[[day, symbol, 0]] = observed.len() as f64;
            output[[day, symbol, 1]] = observed[observed.len() - 1];
            for (index, &window) in windows.iter().enumerate() {
                let window = window as usize;
                if observed.len() >= window + 1 {
                    output[[day, symbol, index + 2]] = observed
                        [observed.len() - window - 1..observed.len() - 1]
                        .iter()
                        .sum::<f64>()
                        / window as f64;
                }
            }
        }
    }
    Ok(output.into_pyarray(py))
}

#[pyfunction]
fn daily_breakout_features<'py>(
    py: Python<'py>,
    high: PyReadonlyArray2<'py, f64>,
    low: PyReadonlyArray2<'py, f64>,
    close: PyReadonlyArray2<'py, f64>,
    day_offsets: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let high = high.as_array();
    let low = low.as_array();
    let close = close.as_array();
    if high.dim() != low.dim() || high.dim() != close.dim() {
        return Err(PyValueError::new_err(
            "OHLC inputs must have the same shape",
        ));
    }
    let offsets = day_offsets.as_slice()?;
    if offsets.len() < 2 || offsets[0] != 0 || offsets[offsets.len() - 1] != close.nrows() as i64 {
        return Err(PyValueError::new_err("invalid day_offsets"));
    }
    let mut output = Array3::from_elem((offsets.len() - 1, close.ncols(), 2), f64::NAN);
    for day in 0..offsets.len() - 1 {
        let start = offsets[day] as usize;
        let end = offsets[day + 1] as usize;
        if end - start < 30 {
            continue;
        }
        for symbol in 0..close.ncols() {
            let observed: Vec<(f64, f64, f64)> = (start..end)
                .map(|row| {
                    (
                        high[[row, symbol]],
                        low[[row, symbol]],
                        close[[row, symbol]],
                    )
                })
                .filter(|(h, l, c)| !h.is_nan() && !l.is_nan() && !c.is_nan())
                .collect();
            if observed.len() < 30 {
                continue;
            }
            let atr = observed[observed.len() - 20..]
                .iter()
                .map(|(h, l, _)| h - l)
                .sum::<f64>()
                / 20.0;
            if atr < 1e-12 {
                output[[day, symbol, 0]] = 0.0;
                output[[day, symbol, 1]] = 0.0;
                continue;
            }
            let mut retrace_sum = 0.0;
            let mut retrace_count = 0usize;
            let mut break_count = 0usize;
            let mut hold_count = 0usize;
            for index in 20..observed.len() {
                let high_max = observed[index - 20..index]
                    .iter()
                    .map(|value| value.0)
                    .fold(f64::NEG_INFINITY, f64::max);
                let low_min = observed[index - 20..index]
                    .iter()
                    .map(|value| value.1)
                    .fold(f64::INFINITY, f64::min);
                let price = observed[index].2;
                let after = (index + 4).min(observed.len() - 1);
                let close_after = observed[after].2;
                if price >= high_max {
                    let future_high = observed[index..(index + 5).min(observed.len())]
                        .iter()
                        .map(|value| value.0)
                        .fold(f64::NEG_INFINITY, f64::max);
                    retrace_sum += (future_high - close_after) / atr;
                    retrace_count += 1;
                } else if price <= low_min {
                    let future_low = observed[index..(index + 5).min(observed.len())]
                        .iter()
                        .map(|value| value.1)
                        .fold(f64::INFINITY, f64::min);
                    retrace_sum += (close_after - future_low) / atr;
                    retrace_count += 1;
                }
                if price >= high_max && high_max - observed[index - 1].0 > 0.0 {
                    break_count += 1;
                    hold_count += usize::from(close_after >= price - 0.5 * atr);
                } else if price <= low_min && observed[index - 1].1 - low_min > 0.0 {
                    break_count += 1;
                    hold_count += usize::from(close_after <= price + 0.5 * atr);
                }
            }
            output[[day, symbol, 0]] = if retrace_count > 0 {
                retrace_sum / retrace_count as f64
            } else {
                0.0
            };
            output[[day, symbol, 1]] = if break_count > 0 {
                hold_count as f64 / break_count as f64
            } else {
                0.5
            };
        }
    }
    Ok(output.into_pyarray(py))
}

fn population_std(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

fn markov_anomaly(prices: &[f64]) -> (Vec<f64>, Vec<bool>) {
    let mut returns = Vec::with_capacity(prices.len());
    returns.push(0.0);
    returns.extend(prices.windows(2).map(|pair| pair[1] / pair[0] - 1.0));
    let states: Vec<usize> = returns
        .iter()
        .map(|value| usize::from(*value > 0.0))
        .collect();
    let mut transitions = [[0.0f64; 2]; 2];
    for pair in states.windows(2) {
        transitions[pair[0]][pair[1]] += 1.0;
    }
    for row in &mut transitions {
        let total = row[0] + row[1];
        let denominator = if total == 0.0 { 1.0 } else { total };
        row[0] /= denominator;
        row[1] /= denominator;
    }
    let mut probabilities = Vec::with_capacity(states.len());
    let mut state_probability = [0.5, 0.5];
    for (index, &state) in states.iter().enumerate() {
        if index > 0 {
            state_probability = [
                state_probability[0] * transitions[0][0] + state_probability[1] * transitions[1][0],
                state_probability[0] * transitions[0][1] + state_probability[1] * transitions[1][1],
            ];
        }
        probabilities.push(state_probability[state]);
    }
    let mut ordered = probabilities.clone();
    ordered.sort_by(f64::total_cmp);
    let location = 0.05 * (ordered.len() - 1) as f64;
    let lower = location.floor() as usize;
    let upper = location.ceil() as usize;
    let threshold = ordered[lower] + (ordered[upper] - ordered[lower]) * location.fract();
    let anomaly = probabilities
        .iter()
        .map(|value| *value < threshold)
        .collect();
    (returns, anomaly)
}

#[pyfunction]
fn daily_smart_money_v4<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    volume: PyReadonlyArray2<'py, f64>,
    day_offsets: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let close = close.as_array();
    let volume = volume.as_array();
    if close.dim() != volume.dim() {
        return Err(PyValueError::new_err(
            "close and volume must have the same shape",
        ));
    }
    let offsets = day_offsets.as_slice()?;
    if offsets.len() < 2 || offsets[0] != 0 || offsets[offsets.len() - 1] != close.nrows() as i64 {
        return Err(PyValueError::new_err("invalid day_offsets"));
    }
    let mut output = Array3::from_elem((offsets.len() - 1, close.ncols(), 2), f64::NAN);
    for day in 0..offsets.len() - 1 {
        let start = offsets[day] as usize;
        let end = offsets[day + 1] as usize;
        if end - start < 30 {
            continue;
        }
        for symbol in 0..close.ncols() {
            let pairs: Vec<(f64, f64)> = (start..end)
                .map(|row| (close[[row, symbol]], volume[[row, symbol]]))
                .filter(|(price, quantity)| !price.is_nan() && !quantity.is_nan())
                .collect();
            if pairs.len() >= 30 {
                let prices: Vec<f64> = pairs.iter().map(|pair| pair.0).collect();
                let (_, anomaly) = markov_anomaly(&prices);
                let anomaly_count = anomaly.iter().filter(|value| **value).count();
                let anomaly_volume = pairs
                    .iter()
                    .zip(&anomaly)
                    .filter(|(_, flag)| **flag)
                    .map(|(pair, _)| pair.1)
                    .sum::<f64>();
                let rest_volume = pairs
                    .iter()
                    .zip(&anomaly)
                    .filter(|(_, flag)| !**flag)
                    .map(|(pair, _)| pair.1)
                    .sum::<f64>();
                if anomaly_count >= 3 && rest_volume >= 1e-12 {
                    let anomaly_value = pairs
                        .iter()
                        .zip(&anomaly)
                        .filter(|(_, flag)| **flag)
                        .map(|(pair, _)| pair.0 * pair.1)
                        .sum::<f64>()
                        / anomaly_volume;
                    let rest_value = pairs
                        .iter()
                        .zip(&anomaly)
                        .filter(|(_, flag)| !**flag)
                        .map(|(pair, _)| pair.0 * pair.1)
                        .sum::<f64>()
                        / rest_volume;
                    if rest_value >= 1e-12 {
                        output[[day, symbol, 0]] = anomaly_value / rest_value;
                    }
                }
            }
            let prices: Vec<f64> = (start..end)
                .map(|row| close[[row, symbol]])
                .filter(|value| !value.is_nan())
                .collect();
            if prices.len() >= 30 {
                let (returns, anomaly) = markov_anomaly(&prices);
                let all_std = population_std(&returns);
                let anomaly_count = anomaly.iter().filter(|value| **value).count();
                output[[day, symbol, 1]] = if all_std < 1e-12 || anomaly_count < 3 {
                    0.0
                } else {
                    let anomaly_returns: Vec<f64> = returns
                        .iter()
                        .zip(anomaly)
                        .filter(|(_, flag)| *flag)
                        .map(|(value, _)| *value)
                        .collect();
                    population_std(&anomaly_returns) / all_std
                };
            }
        }
    }
    Ok(output.into_pyarray(py))
}

fn direction(value: f64) -> f64 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[pyfunction]
fn daily_oi_features<'py>(
    py: Python<'py>,
    high: PyReadonlyArray2<'py, f64>,
    low: PyReadonlyArray2<'py, f64>,
    close: PyReadonlyArray2<'py, f64>,
    volume: PyReadonlyArray2<'py, f64>,
    position: PyReadonlyArray2<'py, f64>,
    day_offsets: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let high = high.as_array();
    let low = low.as_array();
    let close = close.as_array();
    let volume = volume.as_array();
    let position = position.as_array();
    if high.dim() != low.dim()
        || high.dim() != close.dim()
        || high.dim() != volume.dim()
        || high.dim() != position.dim()
    {
        return Err(PyValueError::new_err(
            "OHLCV and position inputs must have the same shape",
        ));
    }
    let offsets = day_offsets.as_slice()?;
    if offsets.len() < 2 || offsets[0] != 0 || offsets[offsets.len() - 1] != close.nrows() as i64 {
        return Err(PyValueError::new_err("invalid day_offsets"));
    }
    let mut output = Array3::from_elem((offsets.len() - 1, close.ncols(), 7), f64::NAN);
    for day in 0..offsets.len() - 1 {
        let start = offsets[day] as usize;
        let end = offsets[day + 1] as usize;
        let rows = end - start;
        for symbol in 0..close.ncols() {
            let previous_close = if start > 0 {
                close[[start - 1, symbol]]
            } else {
                f64::NAN
            };
            let previous_position = if start > 0 {
                position[[start - 1, symbol]]
            } else {
                f64::NAN
            };
            let mut last_close = previous_close;
            let mut last_position = previous_position;
            let mut shape = Vec::new();
            let mut torrent = Vec::new();
            let mut herding = Vec::new();
            let mut peaks = Vec::new();
            for row in start..end {
                let price = close[[row, symbol]];
                let oi = position[[row, symbol]];
                let ret = if !price.is_nan() && !last_close.is_nan() {
                    price / last_close - 1.0
                } else {
                    f64::NAN
                };
                let oi_change = if !oi.is_nan() && !last_position.is_nan() {
                    oi - last_position
                } else {
                    f64::NAN
                };
                let h = high[[row, symbol]];
                let l = low[[row, symbol]];
                let quantity = volume[[row, symbol]];
                if !h.is_nan() && !l.is_nan() && !price.is_nan() && !oi_change.is_nan() {
                    shape.push((h, l, price, oi_change));
                }
                if !ret.is_nan() && !quantity.is_nan() && !oi_change.is_nan() {
                    torrent.push((ret, quantity, oi_change));
                }
                if !ret.is_nan() && !oi_change.is_nan() {
                    herding.push((ret, oi_change));
                }
                if !oi_change.is_nan() {
                    peaks.push(oi_change.abs());
                }
                last_close = price;
                last_position = oi;
            }
            if rows >= 30 && shape.len() >= 30 {
                let high_max = shape
                    .iter()
                    .map(|value| value.0)
                    .fold(f64::NEG_INFINITY, f64::max);
                let low_min = shape
                    .iter()
                    .map(|value| value.1)
                    .fold(f64::INFINITY, f64::min);
                let range = high_max - low_min;
                let absolute_changes: Vec<f64> = shape.iter().map(|value| value.3.abs()).collect();
                let sigma = population_std(&absolute_changes);
                if range < 1e-12 || sigma < 1e-12 {
                    output[[day, symbol, 0]] = 0.5;
                } else {
                    let mean = absolute_changes.iter().sum::<f64>() / absolute_changes.len() as f64;
                    let selected: Vec<f64> = shape
                        .iter()
                        .zip(&absolute_changes)
                        .filter(|(_, change)| **change > mean + 2.0 * sigma)
                        .map(|(value, _)| (value.2 - low_min) / range)
                        .collect();
                    output[[day, symbol, 0]] = if selected.len() < 2 {
                        0.5
                    } else {
                        selected.iter().sum::<f64>() / selected.len() as f64
                    };
                }
                if range < 1e-12 {
                    output[[day, symbol, 4]] = 0.5;
                } else {
                    let additions: Vec<&(f64, f64, f64, f64)> =
                        shape.iter().filter(|value| value.3 > 0.0).collect();
                    output[[day, symbol, 4]] = if additions.len() < 5 {
                        0.5
                    } else {
                        let weight = additions.iter().map(|value| value.3).sum::<f64>();
                        if weight > 1e-12 {
                            additions
                                .iter()
                                .map(|value| ((value.2 - low_min) / range) * value.3)
                                .sum::<f64>()
                                / weight
                        } else {
                            0.5
                        }
                    };
                }
            }
            if rows >= 30 && torrent.len() >= 30 {
                let volume_mean =
                    torrent.iter().map(|value| value.1).sum::<f64>() / torrent.len() as f64;
                if volume_mean >= 1e-12 {
                    let selected: Vec<f64> = torrent
                        .iter()
                        .filter(|value| value.0 < 0.0 && value.1 > volume_mean && value.2 > 0.0)
                        .map(|value| value.0)
                        .collect();
                    output[[day, symbol, 1]] = if selected.len() < 2 {
                        0.0
                    } else {
                        -selected.iter().sum::<f64>() / selected.len() as f64
                    };
                }
            }
            if rows >= 20 && herding.len() >= 20 {
                let valid: Vec<&(f64, f64)> = herding
                    .iter()
                    .filter(|value| value.0 != 0.0 && value.1 != 0.0)
                    .collect();
                output[[day, symbol, 2]] = if valid.len() < 10 {
                    0.0
                } else {
                    valid
                        .iter()
                        .filter(|value| (value.0 > 0.0) == (value.1 > 0.0))
                        .count() as f64
                        / valid.len() as f64
                };
            }
            if rows >= 30 && peaks.len() >= 30 {
                let mean = peaks.iter().sum::<f64>() / peaks.len() as f64;
                let sigma = population_std(&peaks);
                output[[day, symbol, 3]] = if sigma < 1e-12 {
                    0.0
                } else {
                    let jumps: Vec<bool> =
                        peaks.iter().map(|value| *value > mean + sigma).collect();
                    (1..jumps.len() - 1)
                        .filter(|index| jumps[*index] && !(jumps[*index - 1] && jumps[*index + 1]))
                        .count() as f64
                };
            }

            let pairs: Vec<(f64, f64)> = (start..end)
                .map(|row| (position[[row, symbol]], volume[[row, symbol]]))
                .filter(|(oi, quantity)| !oi.is_nan() && !quantity.is_nan())
                .collect();
            if rows >= 20 && pairs.len() >= 20 {
                let mut accumulate = 0usize;
                let mut sell_off = 0usize;
                for pair in pairs.windows(2) {
                    let oi_change = pair[1].0 - pair[0].0;
                    let volume_change = pair[1].1 - pair[0].1;
                    accumulate += usize::from(volume_change < 0.0 && oi_change > 0.0);
                    sell_off += usize::from(volume_change > 0.0 && oi_change < 0.0);
                }
                if pairs.len() - 1 >= 10 {
                    output[[day, symbol, 5]] =
                        (accumulate as f64 - sell_off as f64) / (pairs.len() - 1) as f64;
                }
            }
            let triples: Vec<(f64, f64, f64)> = (start..end)
                .map(|row| {
                    (
                        position[[row, symbol]],
                        volume[[row, symbol]],
                        close[[row, symbol]],
                    )
                })
                .filter(|(oi, quantity, price)| {
                    !oi.is_nan() && !quantity.is_nan() && !price.is_nan()
                })
                .collect();
            if rows >= 20 && triples.len() >= 20 {
                let scores: Vec<f64> = triples
                    .windows(2)
                    .map(|pair| {
                        direction(pair[1].0 - pair[0].0)
                            * direction(pair[1].1 - pair[0].1)
                            * direction(pair[1].2 - pair[0].2)
                    })
                    .filter(|score| *score != 0.0)
                    .collect();
                output[[day, symbol, 6]] = if scores.len() < 5 {
                    0.0
                } else {
                    scores.iter().sum::<f64>() / scores.len() as f64
                };
            }
        }
    }
    Ok(output.into_pyarray(py))
}

#[pyfunction]
fn daily_return_path_features<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    day_offsets: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let close = close.as_array();
    let offsets = day_offsets.as_slice()?;
    if offsets.len() < 2 || offsets[0] != 0 || offsets[offsets.len() - 1] != close.nrows() as i64 {
        return Err(PyValueError::new_err("invalid day_offsets"));
    }
    let mut output = Array3::from_elem((offsets.len() - 1, close.ncols(), 2), f64::NAN);
    for day in 0..offsets.len() - 1 {
        let start = offsets[day] as usize;
        let end = offsets[day + 1] as usize;
        if end - start < 30 {
            continue;
        }
        for symbol in 0..close.ncols() {
            let mut previous = if start > 0 {
                close[[start - 1, symbol]]
            } else {
                f64::NAN
            };
            let mut returns = Vec::new();
            for row in start..end {
                let value = close[[row, symbol]];
                if !value.is_nan() && !previous.is_nan() {
                    let ret = value / previous - 1.0;
                    if !ret.is_nan() {
                        returns.push(ret);
                    }
                }
                previous = value;
            }
            if returns.len() >= 30 {
                let mut patterns = [0usize; 27];
                for window in returns.windows(3) {
                    let mut order = [0usize, 1, 2];
                    order.sort_by(|left, right| window[*left].total_cmp(&window[*right]));
                    patterns[order[0] * 9 + order[1] * 3 + order[2]] += 1;
                }
                let total = (returns.len() - 2) as f64;
                let entropy = patterns
                    .iter()
                    .filter(|count| **count > 0)
                    .map(|count| {
                        let probability = *count as f64 / total;
                        -probability * probability.ln()
                    })
                    .sum::<f64>();
                output[[day, symbol, 0]] = -entropy / 6.0f64.ln();
            }
            if end - start >= 60 && returns.len() >= 60 {
                let vol5: Vec<f64> = returns.chunks_exact(5).map(population_std).collect();
                let vol30: Vec<f64> = returns.chunks_exact(30).map(population_std).collect();
                let mean5 = vol5.iter().sum::<f64>() / vol5.len() as f64;
                let mean30 = vol30.iter().sum::<f64>() / vol30.len() as f64;
                output[[day, symbol, 1]] = if mean30 > 1e-12 { mean5 / mean30 } else { 1.0 };
            }
        }
    }
    Ok(output.into_pyarray(py))
}

#[pyfunction]
fn daily_volume_shock_features<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    volume: PyReadonlyArray2<'py, f64>,
    amount: PyReadonlyArray2<'py, f64>,
    day_offsets: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let close = close.as_array();
    let volume = volume.as_array();
    let amount = amount.as_array();
    if close.dim() != volume.dim() || close.dim() != amount.dim() {
        return Err(PyValueError::new_err(
            "close, volume, and amount must have the same shape",
        ));
    }
    let offsets = day_offsets.as_slice()?;
    if offsets.len() < 2 || offsets[0] != 0 || offsets[offsets.len() - 1] != close.nrows() as i64 {
        return Err(PyValueError::new_err("invalid day_offsets"));
    }
    let mut output = Array3::from_elem((offsets.len() - 1, close.ncols(), 3), f64::NAN);
    for day in 0..offsets.len() - 1 {
        let start = offsets[day] as usize;
        let end = offsets[day + 1] as usize;
        if end - start < 30 {
            continue;
        }
        for symbol in 0..close.ncols() {
            let mut previous = if start > 0 {
                close[[start - 1, symbol]]
            } else {
                f64::NAN
            };
            let mut global_pairs = Vec::new();
            let mut surge_pairs = Vec::new();
            for row in start..end {
                let price = close[[row, symbol]];
                let ret = if !price.is_nan() && !previous.is_nan() {
                    price / previous - 1.0
                } else {
                    f64::NAN
                };
                let quantity = volume[[row, symbol]];
                let turnover = amount[[row, symbol]];
                if !ret.is_nan() && !quantity.is_nan() {
                    global_pairs.push((ret, quantity));
                }
                if !ret.is_nan() && !turnover.is_nan() {
                    surge_pairs.push((ret.abs(), ret.abs() / (turnover + 1e-12)));
                }
                previous = price;
            }
            if global_pairs.len() >= 30 {
                let quantities: Vec<f64> = global_pairs.iter().map(|pair| pair.1).collect();
                let mean = quantities.iter().sum::<f64>() / quantities.len() as f64;
                let threshold = mean + 2.0 * population_std(&quantities);
                let spikes: Vec<usize> = quantities
                    .iter()
                    .enumerate()
                    .filter(|(_, value)| **value > threshold)
                    .map(|(index, _)| index)
                    .collect();
                if spikes.is_empty() {
                    output[[day, symbol, 0]] = 0.0;
                } else {
                    let impacts: Vec<f64> = spikes
                        .iter()
                        .filter_map(|index| {
                            let forward = 5usize.min(global_pairs.len() - index - 1);
                            if forward == 0 {
                                None
                            } else {
                                Some(direction(
                                    global_pairs[index + 1..index + 1 + forward]
                                        .iter()
                                        .map(|pair| pair.0)
                                        .sum::<f64>(),
                                ))
                            }
                        })
                        .collect();
                    output[[day, symbol, 0]] = if impacts.is_empty() {
                        0.0
                    } else {
                        impacts.iter().sum::<f64>() / impacts.len() as f64
                    };
                }
            }

            let price_volume: Vec<(f64, f64)> = (start..end)
                .map(|row| (close[[row, symbol]], volume[[row, symbol]]))
                .filter(|(price, quantity)| !price.is_nan() && !quantity.is_nan())
                .collect();
            if price_volume.len() >= 30 {
                let quantities: Vec<f64> = price_volume.iter().map(|pair| pair.1).collect();
                let mean = quantities.iter().sum::<f64>() / quantities.len() as f64;
                let threshold = mean + 2.0 * population_std(&quantities);
                let spikes: Vec<usize> = quantities
                    .iter()
                    .enumerate()
                    .filter(|(_, value)| **value > threshold)
                    .map(|(index, _)| index)
                    .collect();
                if !spikes.is_empty() {
                    let returns: Vec<f64> = price_volume
                        .windows(2)
                        .map(|pair| pair[1].0 / pair[0].0 - 1.0)
                        .collect();
                    let shocks: Vec<f64> = spikes
                        .iter()
                        .filter(|index| **index > 0)
                        .map(|index| returns[index - 1].abs())
                        .collect();
                    let recoveries: Vec<f64> = spikes
                        .iter()
                        .filter(|index| **index + 5 < price_volume.len())
                        .map(|index| returns[*index..*index + 5].iter().sum::<f64>().abs())
                        .collect();
                    if !recoveries.is_empty() && !shocks.is_empty() {
                        let shock_mean = shocks.iter().sum::<f64>() / shocks.len() as f64;
                        output[[day, symbol, 1]] = recoveries.iter().sum::<f64>()
                            / recoveries.len() as f64
                            / shock_mean.max(1e-9);
                    }
                }
            }
            if end - start >= 60 && surge_pairs.len() >= 60 {
                let returns: Vec<f64> = surge_pairs.iter().map(|pair| pair.0).collect();
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                let sigma = population_std(&returns);
                if sigma > 0.0 {
                    let before: Vec<f64> = returns
                        .iter()
                        .enumerate()
                        .filter(|(_, value)| **value > mean + 2.0 * sigma)
                        .filter_map(|(index, _)| {
                            let lower = index.saturating_sub(20);
                            if index - lower > 5 {
                                Some(
                                    surge_pairs[lower..index]
                                        .iter()
                                        .map(|pair| pair.1)
                                        .sum::<f64>()
                                        / (index - lower) as f64,
                                )
                            } else {
                                None
                            }
                        })
                        .collect();
                    if !before.is_empty() {
                        output[[day, symbol, 2]] = before.iter().sum::<f64>() / before.len() as f64;
                    }
                }
            }
        }
    }
    Ok(output.into_pyarray(py))
}

#[pyfunction]
fn daily_candle_path_features<'py>(
    py: Python<'py>,
    open: PyReadonlyArray2<'py, f64>,
    high: PyReadonlyArray2<'py, f64>,
    low: PyReadonlyArray2<'py, f64>,
    close: PyReadonlyArray2<'py, f64>,
    day_offsets: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let open = open.as_array();
    let high = high.as_array();
    let low = low.as_array();
    let close = close.as_array();
    if open.dim() != high.dim() || open.dim() != low.dim() || open.dim() != close.dim() {
        return Err(PyValueError::new_err(
            "OHLC inputs must have the same shape",
        ));
    }
    let offsets = day_offsets.as_slice()?;
    if offsets.len() < 2 || offsets[0] != 0 || offsets[offsets.len() - 1] != close.nrows() as i64 {
        return Err(PyValueError::new_err("invalid day_offsets"));
    }
    let mut output = Array3::from_elem((offsets.len() - 1, close.ncols(), 2), f64::NAN);
    for day in 0..offsets.len() - 1 {
        let start = offsets[day] as usize;
        let end = offsets[day + 1] as usize;
        if end - start < 30 {
            continue;
        }
        for symbol in 0..close.ncols() {
            let high_low: Vec<(f64, f64)> = (start..end)
                .map(|row| (high[[row, symbol]], low[[row, symbol]]))
                .filter(|(h, l)| !h.is_nan() && !l.is_nan())
                .collect();
            if high_low.len() >= 30 {
                let mut plus = 0.0;
                let mut minus = 0.0;
                for pair in high_low.windows(2) {
                    let up = (pair[1].0 - pair[0].0).max(0.0);
                    let down = (pair[0].1 - pair[1].1).max(0.0);
                    if up > down && up > 0.0 {
                        plus += up;
                    }
                    if down > up && down > 0.0 {
                        minus += down;
                    }
                }
                let denominator = plus + minus;
                output[[day, symbol, 0]] = if denominator > 1e-12 {
                    100.0 * (plus - minus).abs() / denominator
                } else {
                    0.0
                };
            }
            let open_close: Vec<(f64, f64)> = (start..end)
                .map(|row| (open[[row, symbol]], close[[row, symbol]]))
                .filter(|(o, c)| !o.is_nan() && !c.is_nan())
                .collect();
            if open_close.len() >= 30 {
                let mut count = 0usize;
                for index in 1..open_close.len() - 1 {
                    let previous = open_close[index - 1].1 - open_close[index - 1].0;
                    let current = (open_close[index].1 - open_close[index].0).abs();
                    let next = open_close[index + 1].1 - open_close[index + 1].0;
                    let body1 = previous / previous.abs().max(1e-12);
                    let body2 = current / (current + previous.abs() + 1e-12).max(1e-12);
                    let body3 = next / next.abs().max(1e-12);
                    count += usize::from(body1 < -0.5 && body2 < 0.3 && body3 > 0.5);
                }
                output[[day, symbol, 1]] = count as f64 / open_close.len() as f64;
            }
        }
    }
    Ok(output.into_pyarray(py))
}

fn linear_slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (&x_value, &y_value) in x.iter().zip(y) {
        let centered = x_value - x_mean;
        numerator += centered * (y_value - y_mean);
        denominator += centered * centered;
    }
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

#[pyfunction]
fn daily_price_volume_features<'py>(
    py: Python<'py>,
    high: PyReadonlyArray2<'py, f64>,
    low: PyReadonlyArray2<'py, f64>,
    close: PyReadonlyArray2<'py, f64>,
    volume: PyReadonlyArray2<'py, f64>,
    amount: PyReadonlyArray2<'py, f64>,
    day_offsets: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let high = high.as_array();
    let low = low.as_array();
    let close = close.as_array();
    let volume = volume.as_array();
    let amount = amount.as_array();
    if high.dim() != close.dim()
        || low.dim() != close.dim()
        || volume.dim() != close.dim()
        || amount.dim() != close.dim()
    {
        return Err(PyValueError::new_err(
            "price-volume inputs must have the same shape",
        ));
    }
    let offsets = day_offsets.as_slice()?;
    if offsets.len() < 2 || offsets[0] != 0 || offsets[offsets.len() - 1] != close.nrows() as i64 {
        return Err(PyValueError::new_err("invalid day_offsets"));
    }
    let mut output = Array3::from_elem((offsets.len() - 1, close.ncols(), 12), f64::NAN);
    for day in 0..offsets.len() - 1 {
        let start = offsets[day] as usize;
        let end = offsets[day + 1] as usize;
        if end - start < 20 {
            continue;
        }
        for symbol in 0..close.ncols() {
            let amounts: Vec<f64> = (start..end)
                .map(|row| amount[[row, symbol]])
                .filter(|value| !value.is_nan())
                .collect();
            if amounts.len() >= 20 {
                let denominator = amounts.iter().sum::<f64>();
                output[[day, symbol, 0]] = if denominator > 0.0 {
                    amounts
                        .iter()
                        .enumerate()
                        .map(|(index, value)| (index + 1) as f64 * value)
                        .sum::<f64>()
                        / amounts.len() as f64
                        / denominator
                } else {
                    0.5
                };
            }

            let closes: Vec<f64> = (start..end)
                .map(|row| close[[row, symbol]])
                .filter(|value| !value.is_nan())
                .collect();
            if closes.len() >= 20 {
                let maximum = (start..end)
                    .map(|row| high[[row, symbol]])
                    .filter(|value| !value.is_nan())
                    .fold(f64::NEG_INFINITY, f64::max);
                let minimum = (start..end)
                    .map(|row| low[[row, symbol]])
                    .filter(|value| !value.is_nan())
                    .fold(f64::INFINITY, f64::min);
                let width = maximum - minimum;
                output[[day, symbol, 1]] = if width > 1e-12 {
                    (closes[closes.len() - 1] - minimum) / width
                } else {
                    0.5
                };
            }

            let mut returns = Vec::new();
            let mut return_volume = Vec::new();
            let mut previous = if start > 0 {
                close[[start - 1, symbol]]
            } else {
                f64::NAN
            };
            for row in start..end {
                let value = close[[row, symbol]];
                if !value.is_nan() && !previous.is_nan() {
                    let ret = value / previous - 1.0;
                    if !ret.is_nan() {
                        returns.push(ret);
                        let vol = volume[[row, symbol]];
                        if !vol.is_nan() {
                            return_volume.push((ret, vol));
                        }
                    }
                }
                previous = value;
            }
            if returns.len() >= 20 {
                let switches = returns
                    .windows(2)
                    .filter(|pair| {
                        let left = if pair[0] > 0.0 {
                            1
                        } else if pair[0] < 0.0 {
                            -1
                        } else {
                            0
                        };
                        let right = if pair[1] > 0.0 {
                            1
                        } else if pair[1] < 0.0 {
                            -1
                        } else {
                            0
                        };
                        left != right
                    })
                    .count();
                output[[day, symbol, 2]] = switches as f64 / (returns.len() - 1) as f64;
            }
            if return_volume.len() >= 20 {
                let volume_mean = return_volume.iter().map(|pair| pair.1).sum::<f64>()
                    / return_volume.len() as f64;
                if volume_mean >= 1e-12 {
                    let high_volume: Vec<f64> = return_volume
                        .iter()
                        .filter(|pair| pair.1 > volume_mean)
                        .map(|pair| pair.0)
                        .collect();
                    if high_volume.len() >= 3 {
                        let up = high_volume.iter().filter(|value| **value > 0.0).count();
                        let down = high_volume.iter().filter(|value| **value < 0.0).count();
                        output[[day, symbol, 4]] = up.max(down) as f64 / high_volume.len() as f64;
                    }
                    output[[day, symbol, 5]] = return_volume
                        .iter()
                        .map(|pair| pair.0.abs() / (pair.1 / volume_mean + 1e-12))
                        .sum::<f64>()
                        / return_volume.len() as f64;
                }
                let up: Vec<f64> = return_volume
                    .iter()
                    .filter(|pair| pair.0 > 0.0)
                    .map(|pair| pair.1)
                    .collect();
                let down: Vec<f64> = return_volume
                    .iter()
                    .filter(|pair| pair.0 < 0.0)
                    .map(|pair| pair.1)
                    .collect();
                let up_sum = up.iter().sum::<f64>();
                let down_sum = down.iter().sum::<f64>();
                let up_mean = if up.is_empty() {
                    0.0
                } else {
                    up_sum / up.len() as f64
                };
                let down_mean = if down.is_empty() {
                    0.0
                } else {
                    down_sum / down.len() as f64
                };
                output[[day, symbol, 6]] = if down_mean > 1e-12 {
                    up_mean / down_mean
                } else if up_mean > 1e-12 {
                    2.0
                } else {
                    1.0
                };
                output[[day, symbol, 7]] = if up_sum + down_sum > 1e-12 {
                    (up_sum - down_sum) / (up_sum + down_sum)
                } else {
                    0.0
                };
                if end - start >= 30 && return_volume.len() >= 30 {
                    let mut obv = Vec::with_capacity(return_volume.len());
                    let mut cumulative = 0.0;
                    for &(ret, vol) in &return_volume {
                        let direction = if ret > 0.0 {
                            1.0
                        } else if ret < 0.0 {
                            -1.0
                        } else {
                            0.0
                        };
                        cumulative += direction * vol;
                        obv.push(cumulative);
                    }
                    let time: Vec<f64> = (0..obv.len())
                        .map(|index| index as f64 / (obv.len() - 1).max(1) as f64)
                        .collect();
                    let base = return_volume.iter().map(|pair| pair.1).sum::<f64>();
                    output[[day, symbol, 11]] = if base > 1e-12 {
                        linear_slope(&time, &obv) / base
                    } else {
                        0.0
                    };
                }
            }

            let close_amount: Vec<(f64, f64)> = (start..end)
                .map(|row| (close[[row, symbol]], amount[[row, symbol]]))
                .filter(|pair| !pair.0.is_nan() && !pair.1.is_nan())
                .collect();
            if close_amount.len() >= 20 {
                let maximum = close_amount
                    .iter()
                    .map(|pair| pair.0)
                    .fold(f64::NEG_INFINITY, f64::max);
                let minimum = close_amount
                    .iter()
                    .map(|pair| pair.0)
                    .fold(f64::INFINITY, f64::min);
                let middle = (maximum + minimum) / 2.0;
                let upper = close_amount
                    .iter()
                    .filter(|pair| pair.0 > middle)
                    .map(|pair| pair.1)
                    .sum::<f64>();
                let lower = close_amount
                    .iter()
                    .filter(|pair| pair.0 < middle)
                    .map(|pair| pair.1)
                    .sum::<f64>();
                output[[day, symbol, 3]] = if lower > 1e-12 { upper / lower } else { 1.0 };
            }

            let close_volume: Vec<(f64, f64)> = (start..end)
                .map(|row| (close[[row, symbol]], volume[[row, symbol]]))
                .filter(|pair| !pair.0.is_nan() && !pair.1.is_nan())
                .collect();
            if close_volume.len() >= 20 {
                let volume_sum = close_volume.iter().map(|pair| pair.1).sum::<f64>();
                if volume_sum >= 1e-12 {
                    let close_mean = close_volume.iter().map(|pair| pair.0).sum::<f64>()
                        / close_volume.len() as f64;
                    output[[day, symbol, 8]] = (close_volume
                        .iter()
                        .map(|pair| (pair.0 - close_mean).powi(2))
                        .sum::<f64>()
                        / close_volume.len() as f64)
                        .sqrt();
                }
            }

            let depth: Vec<(f64, f64)> = (start..end)
                .map(|row| {
                    (
                        amount[[row, symbol]],
                        high[[row, symbol]] - low[[row, symbol]],
                    )
                })
                .filter(|pair| !pair.0.is_nan() && !pair.1.is_nan())
                .collect();
            if depth.len() >= 20 {
                let nonzero: Vec<f64> = depth
                    .iter()
                    .map(|pair| pair.1)
                    .filter(|value| *value != 0.0)
                    .collect();
                if nonzero.len() >= 10 {
                    let amount_mean =
                        depth.iter().map(|pair| pair.0).sum::<f64>() / depth.len() as f64;
                    let range_mean = nonzero.iter().sum::<f64>() / nonzero.len() as f64;
                    output[[day, symbol, 9]] = if range_mean > 1e-12 {
                        amount_mean / range_mean
                    } else {
                        0.0
                    };
                }
            }

            let high_volume: Vec<(f64, f64)> = (start..end)
                .map(|row| (high[[row, symbol]], volume[[row, symbol]]))
                .filter(|pair| !pair.0.is_nan() && !pair.1.is_nan())
                .collect();
            if high_volume.len() >= 20 {
                let volume_mean =
                    high_volume.iter().map(|pair| pair.1).sum::<f64>() / high_volume.len() as f64;
                if volume_mean >= 1e-12 {
                    let mut maximum = f64::NEG_INFINITY;
                    let mut new_high_volume = Vec::new();
                    for &(price, vol) in &high_volume {
                        if price > maximum {
                            maximum = price;
                            new_high_volume.push(vol);
                        }
                    }
                    output[[day, symbol, 10]] = if new_high_volume.len() >= 2 {
                        new_high_volume.iter().sum::<f64>()
                            / new_high_volume.len() as f64
                            / volume_mean
                    } else {
                        1.0
                    };
                }
            }
        }
    }
    Ok(output.into_pyarray(py))
}

fn linear_percentile(values: &[f64], quantile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let position = (sorted.len() - 1) as f64 * quantile;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
}

fn segmented_ohlc_volatility(
    values: &[(f64, f64, f64)],
    window: usize,
    positive_only: bool,
) -> Vec<f64> {
    values
        .chunks(window)
        .filter(|chunk| chunk.len() == window)
        .filter_map(|chunk| {
            let mean = chunk
                .iter()
                .map(|value| value.0 + value.1 + value.2)
                .sum::<f64>()
                / (3 * window) as f64;
            if mean <= 0.0 {
                return None;
            }
            let variance = chunk
                .iter()
                .flat_map(|value| [value.0, value.1, value.2])
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / (3 * window) as f64;
            let value = variance.sqrt() / mean;
            if positive_only && value <= 0.0 {
                None
            } else {
                Some(value)
            }
        })
        .collect()
}

#[pyfunction]
fn daily_price_path_features<'py>(
    py: Python<'py>,
    open: PyReadonlyArray2<'py, f64>,
    high: PyReadonlyArray2<'py, f64>,
    low: PyReadonlyArray2<'py, f64>,
    close: PyReadonlyArray2<'py, f64>,
    volume: PyReadonlyArray2<'py, f64>,
    day_offsets: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let open = open.as_array();
    let high = high.as_array();
    let low = low.as_array();
    let close = close.as_array();
    let volume = volume.as_array();
    if open.dim() != close.dim()
        || high.dim() != close.dim()
        || low.dim() != close.dim()
        || volume.dim() != close.dim()
    {
        return Err(PyValueError::new_err(
            "price-path inputs must have the same shape",
        ));
    }
    let offsets = day_offsets.as_slice()?;
    if offsets.len() < 2 || offsets[0] != 0 || offsets[offsets.len() - 1] != close.nrows() as i64 {
        return Err(PyValueError::new_err("invalid day_offsets"));
    }
    let mut output = Array3::from_elem((offsets.len() - 1, close.ncols(), 12), f64::NAN);
    for day in 0..offsets.len() - 1 {
        let start = offsets[day] as usize;
        let end = offsets[day + 1] as usize;
        if end - start < 20 {
            continue;
        }
        let open_window = ((end - start) / 4).clamp(10, 30);
        for symbol in 0..close.ncols() {
            let hlc: Vec<(f64, f64, f64)> = (start..end)
                .map(|row| {
                    (
                        high[[row, symbol]],
                        low[[row, symbol]],
                        close[[row, symbol]],
                    )
                })
                .filter(|value| !value.0.is_nan() && !value.1.is_nan() && !value.2.is_nan())
                .collect();

            let closes: Vec<f64> = (start..end)
                .map(|row| close[[row, symbol]])
                .filter(|value| !value.is_nan())
                .collect();
            let close_returns: Vec<f64> = closes
                .windows(2)
                .map(|pair| pair[1] / pair[0] - 1.0)
                .filter(|value| !value.is_nan())
                .collect();
            let log_ranges: Vec<f64> = (start..end)
                .filter_map(|row| {
                    let h = high[[row, symbol]];
                    let l = low[[row, symbol]];
                    let value = (h / l).ln();
                    (!h.is_nan() && !l.is_nan() && value.is_finite()).then_some(value)
                })
                .collect();
            let high_low_count = (start..end)
                .filter(|row| !high[[*row, symbol]].is_nan() && !low[[*row, symbol]].is_nan())
                .count();
            if high_low_count >= 20 && log_ranges.len() >= 10 {
                let parkinson = (log_ranges.iter().map(|value| value * value).sum::<f64>()
                    / (4.0 * log_ranges.len() as f64 * std::f64::consts::LN_2))
                    .sqrt();
                let volatility = if close_returns.is_empty() {
                    f64::NAN
                } else {
                    population_std(&close_returns)
                };
                output[[day, symbol, 0]] = if volatility < 1e-12 {
                    0.0
                } else {
                    -parkinson / volatility
                };
            }

            let mut previous = if start > 0 {
                close[[start - 1, symbol]]
            } else {
                f64::NAN
            };
            let mut hlc_return = Vec::new();
            let mut open_close_return = Vec::new();
            for row in start..end {
                let value = close[[row, symbol]];
                let ret = if !value.is_nan() && !previous.is_nan() {
                    value / previous - 1.0
                } else {
                    f64::NAN
                };
                let h = high[[row, symbol]];
                let l = low[[row, symbol]];
                if !h.is_nan() && !l.is_nan() && !ret.is_nan() {
                    hlc_return.push((h, l, ret));
                }
                let o = open[[row, symbol]];
                if !o.is_nan() && !value.is_nan() && !ret.is_nan() {
                    open_close_return.push((o, value, ret));
                }
                previous = value;
            }
            if hlc_return.len() >= 20 {
                let maximum = hlc_return
                    .iter()
                    .map(|value| value.0)
                    .fold(f64::NEG_INFINITY, f64::max);
                let minimum = hlc_return
                    .iter()
                    .map(|value| value.1)
                    .fold(f64::INFINITY, f64::min);
                let width = maximum - minimum;
                let absolute_sum = hlc_return.iter().map(|value| value.2.abs()).sum::<f64>();
                output[[day, symbol, 1]] = if width < 1e-12 {
                    0.0
                } else if absolute_sum > 1e-12 {
                    -(absolute_sum / width).log10() / (hlc_return.len() as f64).log10()
                } else {
                    0.0
                };
            }
            if open_close_return.len() >= 20 {
                let first_open = open_close_return[0].0;
                let volatility = population_std(
                    &open_close_return
                        .iter()
                        .map(|value| value.2)
                        .collect::<Vec<_>>(),
                );
                output[[day, symbol, 11]] = if first_open < 1e-12 {
                    f64::NAN
                } else if volatility > 1e-12 {
                    (open_close_return[open_close_return.len() - 1].1 / first_open - 1.0)
                        / volatility
                } else {
                    0.0
                };
            }

            if hlc.len() >= 20 {
                let maximum = hlc
                    .iter()
                    .map(|value| value.0)
                    .fold(f64::NEG_INFINITY, f64::max);
                let minimum = hlc
                    .iter()
                    .map(|value| value.1)
                    .fold(f64::INFINITY, f64::min);
                let middle = (maximum + minimum) / 2.0;
                let above: Vec<bool> = hlc.iter().map(|value| value.2 > middle).collect();
                let crossings = above.windows(2).filter(|pair| pair[0] != pair[1]).count();
                output[[day, symbol, 3]] =
                    above.iter().filter(|value| **value).count() as f64 / above.len() as f64;
                if end - start >= 30 && hlc.len() >= 30 {
                    output[[day, symbol, 2]] = -(crossings as f64);
                    let up = above.windows(2).filter(|pair| !pair[0] && pair[1]).count();
                    let down = above.windows(2).filter(|pair| pair[0] && !pair[1]).count();
                    output[[day, symbol, 7]] = if up + down > 0 {
                        up as f64 / (up + down) as f64
                    } else {
                        0.5
                    };
                    let width = maximum - minimum;
                    if width < 1e-12 {
                        output[[day, symbol, 5]] = 0.0;
                        output[[day, symbol, 6]] = 0.0;
                    } else {
                        let prices: Vec<f64> = hlc.iter().map(|value| value.2).collect();
                        output[[day, symbol, 5]] = (linear_percentile(&prices, 0.9)
                            - linear_percentile(&prices, 0.1))
                            / width;
                        output[[day, symbol, 6]] = -(prices
                            .iter()
                            .filter(|price| {
                                let position = (**price - minimum) / width;
                                position <= 0.1 || position >= 0.9
                            })
                            .count() as f64
                            / prices.len() as f64);
                    }
                }
            }

            let close_volume: Vec<(f64, f64)> = (start..end)
                .map(|row| (close[[row, symbol]], volume[[row, symbol]]))
                .filter(|value| !value.0.is_nan() && !value.1.is_nan())
                .collect();
            if close_volume.len() >= 30 {
                let close_values: Vec<f64> = close_volume.iter().map(|value| value.0).collect();
                let sigma = population_std(&close_values);
                output[[day, symbol, 4]] = if sigma < 1e-12 {
                    0.0
                } else {
                    let volume_sum = close_volume.iter().map(|value| value.1).sum::<f64>();
                    let anchor = if volume_sum > 1e-12 {
                        close_volume
                            .iter()
                            .map(|value| value.0 * value.1)
                            .sum::<f64>()
                            / volume_sum
                    } else {
                        close_values.iter().sum::<f64>() / close_values.len() as f64
                    };
                    close_values
                        .iter()
                        .filter(|value| (*value - anchor).abs() <= sigma)
                        .count() as f64
                        / close_values.len() as f64
                };
            }

            let ohlc: Vec<(f64, f64, f64, f64)> = (start..end)
                .map(|row| {
                    (
                        open[[row, symbol]],
                        high[[row, symbol]],
                        low[[row, symbol]],
                        close[[row, symbol]],
                    )
                })
                .filter(|value| {
                    !value.0.is_nan() && !value.1.is_nan() && !value.2.is_nan() && !value.3.is_nan()
                })
                .collect();
            if end - start >= 30 && ohlc.len() >= 30 {
                let maximum = ohlc
                    .iter()
                    .map(|value| value.1)
                    .fold(f64::NEG_INFINITY, f64::max);
                let minimum = ohlc
                    .iter()
                    .map(|value| value.2)
                    .fold(f64::INFINITY, f64::min);
                let width = maximum - minimum;
                if ohlc[0].0 >= 1e-12 && width >= 1e-12 {
                    output[[day, symbol, 8]] = (ohlc[open_window - 1].3 / ohlc[0].0 - 1.0) / width;
                }
            }

            if hlc.len() >= 40 {
                let vol5_all = segmented_ohlc_volatility(&hlc, 5, false);
                let vol5_positive = segmented_ohlc_volatility(&hlc, 5, true);
                let vol30 = segmented_ohlc_volatility(&hlc, 30, false);
                if vol5_positive.len() < 5 || vol30.len() < 3 {
                    output[[day, symbol, 9]] = 0.0;
                } else {
                    let threshold = linear_percentile(&vol5_positive, 0.95);
                    output[[day, symbol, 9]] = if threshold.abs() < 1e-12 {
                        1.0
                    } else {
                        let extreme: Vec<f64> = vol30
                            .iter()
                            .copied()
                            .filter(|value| *value > threshold)
                            .collect();
                        if extreme.is_empty() {
                            0.0
                        } else {
                            extreme.iter().sum::<f64>() / extreme.len() as f64 / threshold
                        }
                    };
                }
                output[[day, symbol, 10]] = if vol5_all.len() < 5 || vol30.len() < 3 {
                    0.0
                } else {
                    let threshold = linear_percentile(&vol5_all, 0.95);
                    vol30.iter().filter(|value| **value > threshold).count() as f64
                        / vol30.len() as f64
                };
            }
        }
    }
    Ok(output.into_pyarray(py))
}

fn mfdfa_hurst(values: &[f64], scales: &[usize], q: f64) -> f64 {
    let mut log_scales = Vec::new();
    let mut log_fluctuations = Vec::new();
    for &scale in scales {
        let segment_count = values.len() / scale;
        if segment_count < 2 {
            continue;
        }
        let time: Vec<f64> = (0..scale).map(|value| value as f64).collect();
        let mut variances = Vec::with_capacity(segment_count);
        for segment in 0..segment_count {
            let observed = &values[segment * scale..(segment + 1) * scale];
            let slope = linear_slope(&time, observed);
            let intercept =
                observed.iter().sum::<f64>() / scale as f64 - slope * (scale - 1) as f64 / 2.0;
            let variance = observed
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let residual = value - (slope * index as f64 + intercept);
                    residual * residual
                })
                .sum::<f64>()
                / scale as f64;
            variances.push(variance);
        }
        let fluctuation = if q == -2.0 {
            if variances.iter().any(|value| *value == 0.0) {
                0.0
            } else {
                (variances.iter().map(|value| value.powf(-1.0)).sum::<f64>()
                    / variances.len() as f64)
                    .powf(-0.5)
            }
        } else {
            (variances.iter().sum::<f64>() / variances.len() as f64).sqrt()
        };
        if fluctuation > 1e-12 {
            log_scales.push((scale as f64).ln());
            log_fluctuations.push(fluctuation.ln());
        }
    }
    if log_scales.len() < 2 {
        0.5
    } else {
        linear_slope(&log_scales, &log_fluctuations)
    }
}

#[pyfunction]
fn daily_mfdfa_width<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    day_offsets: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let close = close.as_array();
    let offsets = day_offsets.as_slice()?;
    if offsets.len() < 2 || offsets[0] != 0 || offsets[offsets.len() - 1] != close.nrows() as i64 {
        return Err(PyValueError::new_err("invalid day_offsets"));
    }
    let mut output = Array2::from_elem((offsets.len() - 1, close.ncols()), f64::NAN);
    for day in 0..offsets.len() - 1 {
        let start = offsets[day] as usize;
        let end = offsets[day + 1] as usize;
        if end - start < 60 {
            continue;
        }
        for symbol in 0..close.ncols() {
            let prices: Vec<f64> = (start..end)
                .map(|row| close[[row, symbol]])
                .filter(|value| !value.is_nan())
                .collect();
            if prices.len() < 60 {
                continue;
            }
            let positive: Vec<f64> = prices.into_iter().filter(|value| *value > 0.0).collect();
            let returns: Vec<f64> = positive
                .windows(2)
                .map(|pair| pair[1].ln() - pair[0].ln())
                .collect();
            if returns.len() < 40 {
                continue;
            }
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            let mut cumulative = Vec::with_capacity(returns.len());
            let mut total = 0.0;
            for value in returns {
                total += value - mean;
                cumulative.push(total);
            }
            let n = cumulative.len();
            let scales: Vec<usize> = [n / 8.max(1), n / 4.max(1), n / 2.max(1)]
                .into_iter()
                .zip([6, 8, 10])
                .map(|(value, minimum)| value.max(minimum))
                .filter(|value| *value < n - 2)
                .collect();
            let low = mfdfa_hurst(&cumulative, &scales, -2.0);
            let high = mfdfa_hurst(&cumulative, &scales, 2.0);
            output[[day, symbol]] = (low - high).max(0.0);
        }
    }
    Ok(output.into_pyarray(py))
}

fn histogram_bin(value: f64) -> Option<usize> {
    if !(-4.0..=4.0).contains(&value) {
        None
    } else if value == 4.0 {
        Some(9)
    } else {
        Some(((value + 4.0) / 0.8).floor() as usize)
    }
}

#[pyfunction]
fn daily_histogram_stability<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    day_offsets: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let values = values.as_array();
    let offsets = day_offsets.as_slice()?;
    if offsets.len() < 2 || offsets[0] != 0 || offsets[offsets.len() - 1] != values.nrows() as i64 {
        return Err(PyValueError::new_err("invalid day_offsets"));
    }
    let mut output = Array2::from_elem((offsets.len() - 1, values.ncols()), f64::NAN);
    for day in 0..offsets.len() - 1 {
        let start = offsets[day] as usize;
        let end = offsets[day + 1] as usize;
        if end - start < 30 {
            continue;
        }
        for symbol in 0..values.ncols() {
            let observed: Vec<f64> = (start..end)
                .map(|row| values[[row, symbol]])
                .filter(|value| !value.is_nan())
                .collect();
            if observed.len() < 30 {
                output[[day, symbol]] = 0.0;
                continue;
            }
            let mean = observed.iter().sum::<f64>() / observed.len() as f64;
            let std = (observed
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / observed.len() as f64)
                .sqrt();
            if std < 1e-12 {
                output[[day, symbol]] = 0.0;
                continue;
            }
            let normalized: Vec<f64> = observed.iter().map(|value| (value - mean) / std).collect();
            let mut total = [0usize; 10];
            for &value in &normalized {
                if let Some(bin) = histogram_bin(value) {
                    total[bin] += 1;
                }
            }
            let mut distances = Vec::new();
            let mut invalid = false;
            for window in normalized.chunks(5) {
                if window.len() < 3 || normalized.len() - window.len() < 10 {
                    continue;
                }
                let mut inside = [0usize; 10];
                for &value in window {
                    if let Some(bin) = histogram_bin(value) {
                        inside[bin] += 1;
                    }
                }
                let inside_total = inside.iter().sum::<usize>();
                let outside_total = total.iter().sum::<usize>() - inside_total;
                if inside_total == 0 || outside_total == 0 {
                    invalid = true;
                    break;
                }
                distances.push(
                    (0..10)
                        .map(|bin| {
                            let inner = inside[bin] as f64 / inside_total as f64;
                            let outer = (total[bin] - inside[bin]) as f64 / outside_total as f64;
                            (inner - outer).abs()
                        })
                        .sum::<f64>(),
                );
            }
            if invalid {
                continue;
            }
            if distances.is_empty() {
                output[[day, symbol]] = 0.0;
            } else {
                let distance_mean = distances.iter().sum::<f64>() / distances.len() as f64;
                output[[day, symbol]] = (distances
                    .iter()
                    .map(|value| (value - distance_mean).powi(2))
                    .sum::<f64>()
                    / distances.len() as f64)
                    .sqrt();
            }
        }
    }
    Ok(output.into_pyarray(py))
}

#[pymodule]
fn _mf_factor_kernels(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(daily_return_stats, module)?)?;
    module.add_function(wrap_pyfunction!(daily_unary_stats, module)?)?;
    module.add_function(wrap_pyfunction!(daily_pair_stats, module)?)?;
    module.add_function(wrap_pyfunction!(daily_lagged_pair_stats, module)?)?;
    module.add_function(wrap_pyfunction!(daily_tail_means, module)?)?;
    module.add_function(wrap_pyfunction!(daily_breakout_features, module)?)?;
    module.add_function(wrap_pyfunction!(daily_smart_money_v4, module)?)?;
    module.add_function(wrap_pyfunction!(daily_oi_features, module)?)?;
    module.add_function(wrap_pyfunction!(daily_return_path_features, module)?)?;
    module.add_function(wrap_pyfunction!(daily_volume_shock_features, module)?)?;
    module.add_function(wrap_pyfunction!(daily_candle_path_features, module)?)?;
    module.add_function(wrap_pyfunction!(daily_price_volume_features, module)?)?;
    module.add_function(wrap_pyfunction!(daily_price_path_features, module)?)?;
    module.add_function(wrap_pyfunction!(daily_mfdfa_width, module)?)?;
    module.add_function(wrap_pyfunction!(daily_histogram_stability, module)?)?;
    Ok(())
}
