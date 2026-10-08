"""Small synthetic dynamical tests; no provider data or downloads."""
import numpy as np
import pandas as pd
import pytest
from celegans_dtpr_v4.dynamics import DynamicsModel, ModelParams, fit_node_readout_gain
from celegans_dtpr_v4.experiment import simulate_full_or_local, simulate_selective
from celegans_dtpr_v4.kernel_window import _eval_kernel_array, _window_stat


def model():
    return DynamicsModel(
        n=2, chemical=np.array([[0., 1.], [1., 0.]]),
        gap=np.array([[0., 1.], [1., 0.]]), reversal=np.zeros((2, 2)),
        params=ModelParams(g_chem=.14, g_gap=.02, g_leak=.7,
                          stimulus_amplitude=1., v_half=-26., k_slope=4.,
                          capacitance=1., resting_potential=-35.,
                          synapse_rise_rate=2., synapse_decay_rate=1.),
        dt=.0125, steps=32, stimulus_steps=8, response_start_step=0,
        observation_window_steps=32, correction_decision_steps=4,
        readout_gain=np.ones(2), calibration_table=pd.DataFrame())


@pytest.mark.parametrize('threshold', [0., 1e-5, 1e10])
def test_finite_transport_preserves_represented_state(threshold):
    m = model()
    full = simulate_full_or_local(m, np.arange(2), True, 'full')
    selective = simulate_selective(m, np.arange(2), 'finite_displacement', np.ones(4), threshold)
    # Stored residuals are float32; the recurrence and closure audit are float64.
    # Test endpoint agreement at storage precision, and audit exact closure below.
    np.testing.assert_allclose(selective.states[-1] + selective.residuals[-1],
                               full.states[-1], rtol=0., atol=np.finfo(np.float32).eps *
                               max(1., np.max(np.abs(selective.residuals[-1]))))
    assert selective.max_closure_error < 1e-12
    if threshold == 0.:
        np.testing.assert_allclose(selective.states[-1], full.states[-1], atol=1e-12)
    if threshold == 1e10:
        local = simulate_full_or_local(m, np.arange(2), False, 'local')
        np.testing.assert_allclose(selective.states[-1], local.states[-1], atol=1e-12)
        assert np.linalg.norm(selective.residuals[-1]) > 0.


def test_nonnegative_response_local_readout():
    x = np.array([[1., 2.], [1., 2.]])
    y = np.array([[2., 4.], [-2., -4.]])
    gain = fit_node_readout_gain(x, y, np.ones((2, 2), bool), 0.)
    np.testing.assert_allclose(gain, [2., 0.])


def test_signed_kernel_window_keeps_negative_influence():
    t = np.linspace(0., 1., 101)
    y = _eval_kernel_array(np.array([0., -2., 0., 0.]), t)
    assert _window_stat(t, y, 1., 'signed_mean') == pytest.approx(-2.)
    assert _eval_kernel_array(np.ones(3), t) is None
