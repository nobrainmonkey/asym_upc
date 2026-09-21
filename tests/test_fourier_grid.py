"""Physical Fourier modes, units, and grids prepared before an event FFT."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from upcmodel.amplitudes import dipole_profile_fourier_x, momentum_transfer_grid
from upcmodel.fourier_transform import fast_fourier_transform_1D, fft_frequency_grid
from upcmodel.parameters import HBARC_GEV_FM as h


class FourierGridTests(unittest.TestCase):
    def test_plane_wave_mode_and_normalization_on_shifted_grids(self):
        for size in (64, 65):
            for mode in (-5, 5):
                grid = -3.7 + 0.15 * np.arange(size)
                period = size * 0.15
                wave_number = 2 * np.pi * mode / period
                wave = np.exp(1j * wave_number * grid)
                for angular in (False, True):
                    with self.subTest(size=size, mode=mode, angular=angular):
                        spectrum, frequency = fast_fourier_transform_1D(wave, grid, angular)
                        assert_array_equal(frequency, fft_frequency_grid(grid, angular))
                        peak = np.argmax(abs(spectrum))
                        expected_frequency = wave_number if angular else mode / period
                        self.assertAlmostEqual(frequency[peak], expected_frequency)
                        expected = np.zeros(size, dtype=complex)
                        expected[peak] = period / np.sqrt(2 * np.pi) if angular else period
                        assert_allclose(spectrum, expected, atol=3e-13, rtol=1e-13)

    def test_precomputed_momenta_match_translated_gaussian_transform(self):
        for size in (127, 128):
            with self.subTest(size=size):
                bx = np.linspace(-8, 8, size, endpoint=False)
                by = np.linspace(-8, 8, 161)
                delta = momentum_transfer_grid(bx)
                keep = (delta >= 0) & (delta <= 3)
                x, y = np.meshgrid(bx, by, indexing="ij")
                sx, sy, cx, cy = 0.6, 0.8, 0.35, -0.4
                profile = np.exp(-0.5 * ((x-cx)/sx)**2 - 0.5 * ((y-cy)/sy)**2)
                transformed, fft_delta = dipole_profile_fourier_x(profile.reshape(1, -1), bx, by)
                assert_array_equal(delta, fft_delta)
                expected = 2*np.pi*sx*sy/h**2 * np.exp(
                    -0.5*(sx*delta/h)**2 - 1j*cx*delta/h
                )
                assert_allclose(transformed[0, keep], expected[keep], rtol=1e-12, atol=1e-12)

    def test_invalid_fft_grids_are_rejected_before_overlap_precomputation(self):
        for grid in ([], [0], [[0, 1]], [0, 0], [1, 0], [0, 1, 3],
                     [0, np.nan], [0, np.inf], [0, 1j]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                momentum_transfer_grid(grid)


if __name__ == "__main__":
    unittest.main()
