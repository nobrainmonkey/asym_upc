"""Analytic and closure checks for complex event-amplitude moments."""

import contextlib
import importlib
import io
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

import upcmodel.ensemble as ensemble
from upcmodel.ensemble import amplitude_moments


class AmplitudeMomentsTests(unittest.TestCase):
    def test_identical_complex_events_have_zero_variance(self):
        amplitudes = np.tile([1 + 2j, 3 - 4j], (7, 1))
        mean, variance = amplitude_moments(amplitudes)
        assert_array_equal(mean, [1 + 2j, 3 - 4j])
        assert_array_equal(variance, [0, 0])

    def test_opposite_amplitudes_cancel_coherently(self):
        amplitudes = np.array([[1 + 2j, 3 - 4j], [-1 - 2j, -3 + 4j]])
        mean, variance = amplitude_moments(amplitudes)
        assert_array_equal(mean, [0j, 0j])
        assert_allclose(variance, [5, 25], rtol=1e-14)

    def test_complex_population_variance_uses_modulus_and_divides_by_e(self):
        mean, variance = amplitude_moments([[1 + 2j], [3 + 4j]])
        assert_array_equal(mean, [2 + 3j])
        assert_allclose(variance, [2], rtol=1e-14)
        self.assertTrue(np.iscomplexobj(mean))
        self.assertFalse(np.iscomplexobj(variance))

    def test_momentum_bins_are_not_averaged_together(self):
        mean, variance = amplitude_moments([[1, 10, -2], [3, 14, -2]])
        assert_array_equal(mean, [2, 12, -2])
        assert_array_equal(variance, [1, 4, 0])

    def test_single_event_retains_all_momentum_bins(self):
        mean, variance = amplitude_moments([[1 + 2j, 3 - 4j]])
        self.assertEqual(mean.shape, (2,))
        assert_array_equal(mean, [1 + 2j, 3 - 4j])
        assert_array_equal(variance, [0, 0])

    def test_extra_axes_and_good_walker_closure(self):
        rng = np.random.default_rng(20260920)
        amplitudes = (
            rng.normal(size=(37, 13, 4, 2))
            + 1j * rng.normal(size=(37, 13, 4, 2))
        )
        mean, variance = amplitude_moments(amplitudes)
        self.assertEqual(mean.shape, (13, 4, 2))
        self.assertEqual(variance.shape, (13, 4, 2))
        self.assertTrue(np.all(variance >= 0))
        assert_allclose(
            abs(mean)**2 + variance,
            np.mean(abs(amplitudes)**2, axis=0),
            rtol=2e-14,
        )

    def test_common_complex_factor_preserves_the_correction_convention(self):
        amplitudes = np.array([[1 + 2j, 3], [-1j, 2 - 4j], [2, -3j]])
        mean, variance = amplitude_moments(amplitudes)
        factor = 1.2 * np.exp(0.7j)
        corrected_mean, corrected_variance = amplitude_moments(factor * amplitudes)
        assert_allclose(corrected_mean, factor * mean, rtol=1e-14)
        assert_allclose(corrected_variance, abs(factor)**2 * variance, rtol=1e-14)

    def test_event_dependent_survival_can_enter_before_the_same_moments(self):
        # Fixed bare amplitude 2i; sqrt(P) for the two events is 1/2 and 1.
        probability = np.array([0.25, 1])
        path_amplitudes = np.sqrt(probability)[:, None] * np.full((2, 1), 2j)
        mean, variance = amplitude_moments(path_amplitudes)
        assert_array_equal(mean, [1.5j])
        assert_array_equal(variance, [0.25])
        assert_array_equal(abs(mean)**2 + variance, [2.5])

    def test_centering_resolves_small_fluctuations_on_a_large_mean(self):
        center = 1e12 + 1e12j
        amplitudes = np.array([[center + 1 + 2j], [center - 1 - 2j]])
        mean, variance = amplitude_moments(amplitudes)
        assert_array_equal(mean, [center])
        assert_allclose(variance, [5], rtol=1e-14)

    def test_input_array_is_not_mutated(self):
        amplitudes = np.array([[1 + 2j, 3], [2 - 1j, 4]])
        before = amplitudes.copy()
        amplitudes.setflags(write=False)
        amplitude_moments(amplitudes)
        assert_array_equal(amplitudes, before)

    def test_missing_event_axis_and_empty_axes_are_rejected(self):
        for invalid in (
            1j, np.array([1 + 2j, 3 + 4j]), [],
            np.empty((0, 3)), np.empty((2, 0)), np.empty((2, 3, 0)),
        ):
            with self.subTest(shape=np.shape(invalid)), self.assertRaises(ValueError):
                amplitude_moments(invalid)

    def test_nonfinite_amplitudes_are_rejected(self):
        for invalid in (np.nan, np.inf, -np.inf, complex(1, np.inf)):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                amplitude_moments([[1], [invalid]])

    def test_import_does_not_execute_the_demo(self):
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            importlib.reload(ensemble)
        self.assertEqual(captured.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
