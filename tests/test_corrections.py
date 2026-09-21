"""Analytic references for model-independent amplitude slopes and skewness.

Run from the project directory:
    .venv/bin/python -m unittest discover -s tests -v
"""

import math
import unittest

import numpy as np
from numpy.testing import assert_allclose

from upcmodel.corrections import (
    effective_lambda,
    effective_lambda_from_derivative,
    effective_lambda_from_samples,
    skewness_factor,
)


class EffectiveLambdaTests(unittest.TestCase):
    def test_power_laws_preserve_axes_and_are_not_gbw_limited(self):
        powers = np.array([[-0.3, 0, 0.2], [0.4, 0.8, 1.2]])
        phases = np.exp(1j * np.arange(6).reshape(2, 3))
        result = effective_lambda(lambda x: phases * x**(-powers), 1e-3)
        self.assertEqual(result.shape, (2, 3))
        assert_allclose(result, powers, rtol=1e-12, atol=1e-13)

    def test_scalar_slope_and_invariance_under_units_and_phase(self):
        for scale in (1e-200, 1, 1e200):
            for phase in (1, -1, 1j, np.exp(1.2j)):
                with self.subTest(scale=scale, phase=phase):
                    result = effective_lambda(lambda x: scale * phase * x**-0.37, 0.002)
                    self.assertEqual(np.ndim(result), 0)
                    self.assertAlmostEqual(result, 0.37, places=10)

    def test_samples_are_centered_in_log_x_and_called_only_twice(self):
        xs = []
        def amplitude(x):
            xs.append(x)
            y = np.log(1 / x)
            return 2j * np.exp(0.1 * y + 0.02 * y**2)

        x, step = 0.003, 0.04
        result = effective_lambda(amplitude, x, dlogx=step)
        assert_allclose(xs, [x * np.exp(-step), x * np.exp(step)], rtol=1e-15)
        self.assertAlmostEqual(result, 0.1 + 0.04 * np.log(1 / x), places=12)

    def test_cached_samples_match_the_callable_interface(self):
        x, step = 1e-3, 0.05
        def amplitude(value):
            return 1j * (1 - np.exp(-0.6 * value**-0.17))

        cached = effective_lambda_from_samples(
            amplitude(x * np.exp(-step)), amplitude(x * np.exp(step)), dlogx=step
        )
        self.assertEqual(cached, effective_lambda(amplitude, x, dlogx=step))

    def test_changing_complex_phase_does_not_change_magnitude_slope(self):
        def amplitude(x):
            y = np.log(1 / x)
            return np.exp(0.35 * y + 1j * y**2)

        x = 0.002
        value = amplitude(x)
        derivative = value * (0.35 + 2j * np.log(1 / x))
        self.assertAlmostEqual(effective_lambda(amplitude, x), 0.35, places=12)
        self.assertAlmostEqual(
            effective_lambda_from_derivative(value, derivative), 0.35, places=12
        )

    def test_nonlinear_amplitude_has_second_order_convergence(self):
        x = 0.03
        y = np.log(1 / x)
        def amplitude(value):
            return 1j * np.exp(np.sin(np.log(1 / value)))

        exact = np.cos(y)
        coarse = abs(effective_lambda(amplitude, x, dlogx=0.1) - exact)
        fine = abs(effective_lambda(amplitude, x, dlogx=0.05) - exact)
        self.assertAlmostEqual(coarse / fine, 4, delta=0.002)
        self.assertAlmostEqual(
            effective_lambda_from_derivative(amplitude(x), amplitude(x) * exact),
            exact, places=13,
        )

    def test_analytic_x_derivative_conversion_and_zero_derivative(self):
        x = 0.01
        powers = np.array([0.0, -0.2, 0.7])
        amplitude = 1j * x**(-powers)
        derivative_x = -powers * amplitude / x
        result = effective_lambda_from_derivative(amplitude, -x * derivative_x)
        assert_allclose(result, powers, atol=1e-14)

    def test_analytic_derivative_is_stable_for_large_and_small_normalizations(self):
        for amplitude in (complex(1.7e308, 1.7e308), complex(1e-310, 1e-310)):
            with self.subTest(amplitude=amplitude):
                self.assertAlmostEqual(
                    effective_lambda_from_derivative(amplitude, amplitude / 2),
                    0.5, places=12,
                )

    def test_ensemble_mean_is_differentiated_before_taking_the_log(self):
        x = 0.001
        powers = np.array([0.1, 0.6])
        def events(value):
            return 1j * np.array([1, 3]) * value**(-powers)

        slope = effective_lambda(lambda value: events(value).mean(), x, dlogx=0.001)
        expected = np.real(np.sum(powers * events(x)) / np.sum(events(x)))
        self.assertAlmostEqual(slope, expected, places=8)
        self.assertGreater(abs(slope - powers.mean()), 0.2)
        assert_allclose(effective_lambda(events, x), powers, atol=1e-12)

    def test_log_samples_avoid_overflow_in_amplitude_magnitude(self):
        # Both complex components are finite even though abs(A_high) overflows.
        high = complex(1.7e308, 1.7e308)
        low = high / 2
        self.assertAlmostEqual(
            effective_lambda_from_samples(low, high, dlogx=0.5), -np.log(2), places=11
        )

    def test_invalid_steps_and_x_are_rejected_before_evaluating(self):
        def should_not_run(x):
            self.fail("An invalid evaluation window reached the amplitude")

        for x in (0, -1, 1, 2, np.nan, np.inf, 0.001j, [0.001]):
            with self.subTest(x=x), self.assertRaises(ValueError):
                effective_lambda(should_not_run, x)
        for step in (0, -0.1, 1e-30, 1000, np.nan, np.inf, 0.1j, [0.05]):
            with self.subTest(step=step), self.assertRaises(ValueError):
                effective_lambda(should_not_run, 0.001, dlogx=step)

    def test_invalid_amplitudes_and_mismatched_axes_are_rejected(self):
        for invalid in (0, [], [1, 0], np.nan, np.inf, 1j * np.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    effective_lambda(lambda x: invalid, 0.001)
                with self.assertRaises(ValueError):
                    effective_lambda_from_derivative(invalid, 1)
        with self.assertRaises(ValueError):
            effective_lambda_from_samples([1, 2], 1, dlogx=0.05)
        with self.assertRaises(ValueError):
            effective_lambda_from_derivative([1, 2], [[1, 2]])
        with self.assertRaises(ValueError):
            effective_lambda_from_derivative(1, np.inf)
        with self.assertRaises(ValueError):
            effective_lambda(2j, 0.001)


class SkewnessFactorTests(unittest.TestCase):
    def test_known_exact_values_and_shape(self):
        powers = np.array([[0, 1, 2], [3, 4, -1]])
        factors = skewness_factor(powers)
        self.assertEqual(factors.shape, powers.shape)
        assert_allclose(factors, [[1, 2.5, 7], [21, 66, 0.5]], rtol=1e-13)
        self.assertEqual(np.ndim(skewness_factor(0)), 0)

    def test_log_gamma_handles_large_slope_without_gamma_overflow(self):
        # Independent recurrence: R_g(l+1)/R_g(l) = 4*(l+2.5)/(l+4).
        expected = math.prod(4 * (index + 2.5) / (index + 4) for index in range(200))
        assert_allclose(skewness_factor(200), expected, rtol=5e-13)

    def test_amplitude_and_cross_section_corrections_agree(self):
        amplitude = np.array([2j, 1 + 3j])
        factor = skewness_factor(0.2)
        assert_allclose(abs(factor * amplitude)**2, factor**2 * abs(amplitude)**2)

    def test_invalid_slopes_and_unrepresentable_factors_are_rejected(self):
        for invalid in ([], np.nan, np.inf, -np.inf, -2.5, -3, 0.2j, 1000):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                skewness_factor(invalid)


if __name__ == "__main__":
    unittest.main()
