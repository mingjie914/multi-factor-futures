import numpy as np
import pandas as pd
import pytest

from factors.synthesizer import residualize_against_factors
from factors.synthesizer import remove_common_components


def panels():
    rng = np.random.default_rng(924)
    dates = pd.bdate_range('2020-01-01', periods=70)
    x = rng.normal(size=(70, 9, 3))
    q = np.array([np.linalg.qr(row-row.mean(axis=0))[0] for row in x])*np.sqrt(8)
    frame = lambda a: pd.DataFrame(a, index=dates, columns=list('ABCDEFGHI'))
    return {'a': frame(q[:, :, 0]), 'b': frame(q[:, :, 1]),
            'candidate': frame(.6*q[:, :, 0]+.8*q[:, :, 2])}, dates[:50]


def test_joint_residual_is_orthogonal_and_not_rescaled():
    values, dates = panels()
    residual, info = residualize_against_factors(values, ['a', 'b'], dates)
    detail = info['candidate']
    np.testing.assert_allclose(detail['coefficients'], [0, .6, 0], atol=1e-12)
    assert detail['retained_variance_ratio'] == pytest.approx(.64)
    assert detail['max_normal_equation_error'] < 1e-12
    np.testing.assert_allclose(residual['candidate'].std(axis=1, ddof=0), .8, atol=1e-12)


def test_future_perturbation_does_not_change_fit_or_past_output():
    values, dates = panels()
    a, info = residualize_against_factors(values, ['a', 'b'], dates)
    changed = {n: v.copy() for n, v in values.items()}
    for v in changed.values():
        v.loc[v.index > dates.max()] = np.arange(9)*12345
    b, updated = residualize_against_factors(changed, ['a', 'b'], dates)
    assert updated == info
    pd.testing.assert_frame_equal(a['candidate'].loc[dates], b['candidate'].loc[dates])


def test_direction_is_supplied_once_and_reference_order_is_irrelevant():
    values, dates = panels()
    a, _ = residualize_against_factors(values, ['a', 'b'], dates)
    values['candidate'] *= -1
    b, _ = residualize_against_factors(values, ['b', 'a'], dates)
    np.testing.assert_allclose(b['candidate'], -a['candidate'], atol=1e-12)


def test_missing_values_and_constant_candidates_are_not_filled():
    values, dates = panels()
    values['candidate'].iloc[0, 0] = np.nan
    values['constant'] = values['a']*0+1
    result, info = residualize_against_factors(values, ['a', 'b'], dates)
    assert pd.isna(result['candidate'].iloc[0, 0])
    assert result['constant'].isna().all().all()
    assert info['constant']['status'] == 'insufficient_or_constant'


def test_collinear_references_fail_instead_of_silent_fallback():
    values, dates = panels()
    values['b'] = values['a'].copy()
    with pytest.raises(ValueError, match='rank deficient'):
        residualize_against_factors(values, ['a', 'b'], dates)


@pytest.mark.parametrize('problem', ['axes', 'dates'])
def test_invalid_alignment_or_training_dates_fail(problem):
    values, dates = panels()
    if problem == 'axes':
        values['candidate'] = values['candidate'].iloc[::-1]
    else:
        dates = dates.append(pd.DatetimeIndex(['2040-01-01']))
    with pytest.raises(ValueError):
        residualize_against_factors(values, ['a', 'b'], dates)


def test_pca_removes_projection_and_is_order_independent():
    values, dates = panels()
    output, info = remove_common_components(values, dates)
    assert info['max_projection_error'] < 1e-12
    assert all(0 <= v <= 1 for v in info['retained_variance_ratio'].values())
    reverse, _ = remove_common_components(dict(reversed(list(values.items()))), dates)
    for name in values:
        np.testing.assert_allclose(output[name], reverse[name], atol=1e-12)


def test_pca_future_missingness_does_not_change_past_fit():
    values, dates = panels()
    output, info = remove_common_components(values, dates)
    changed = {n: v.copy() for n, v in values.items()}
    changed['a'].loc[changed['a'].index > dates.max()] = np.nan
    other, other_info = remove_common_components(changed, dates)
    np.testing.assert_allclose(info['components'], other_info['components'], atol=1e-12)
    for name in values:
        np.testing.assert_allclose(output[name].loc[dates], other[name].loc[dates], atol=1e-12)
        assert other[name].loc[other[name].index > dates.max()].isna().all().all()


@pytest.mark.parametrize('count', [0, 3, True])
def test_pca_cannot_remove_every_dimension(count):
    values, dates = panels()
    with pytest.raises(ValueError, match='component count'):
        remove_common_components(values, dates, n_components=count)
