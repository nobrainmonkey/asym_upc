"""Proton-profile normalization and compatibility with existing amplitudes."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from upcmodel.amplitudes import (
    gbw_dipole_amplitude,
    nuclear_dipole_profile,
    proton_dipole_profile,
)
from upcmodel.parameters import GBW_PAPER, RHO_HS


class ProtonProfileTests(unittest.TestCase):
    def test_gaussian_normalization_and_radius_axis(self):
        grid = np.linspace(-4, 4, 201)
        bx, by = np.meshgrid(grid, grid, indexing="ij")
        points = np.column_stack((bx.ravel(), by.ravel()))
        radii = np.array([0, 0.1, 1])
        sigma0 = 4 * np.pi * RHO_HS.B_p_fm2
        profile = proton_dipole_profile(
            1e-3, radii, points, [[0, 0, 7]], weights=[1], sigma_fm=0.5,
            sigma_0_fm2=sigma0,
        )
        self.assertEqual(profile.shape, (3, points.shape[0]))
        integrated = np.trapezoid(
            np.trapezoid(profile.reshape(3, 201, 201), x=grid, axis=2),
            x=grid, axis=1,
        )
        assert_allclose(integrated, sigma0 * gbw_dipole_amplitude(1e-3, radii), atol=1e-12)

    def test_agrees_with_a_one_where_the_nuclear_branch_is_valid(self):
        args = (1e-3, [0.02, 0.05], [[0, 0], [0.2, 0.1]], [[0, 0, 0]], GBW_PAPER)
        kwargs = dict(weights=[1], sigma_fm=np.sqrt(RHO_HS.B_p_fm2),
                      sigma_0_fm2=4 * np.pi * RHO_HS.B_p_fm2)
        assert_allclose(
            proton_dipole_profile(*args, **kwargs),
            nuclear_dipole_profile(*args, **kwargs, A=1), rtol=1e-14,
        )

    def test_compact_hotspot_uses_the_linear_proton_prescription(self):
        radius = 2.0
        sigma0 = 4 * np.pi * RHO_HS.B_p_fm2
        result = proton_dipole_profile(
            1e-4, radius, [[0, 0]], [[0, 0, 0]], weights=[1],
            sigma_fm=np.sqrt(RHO_HS.B_hs_fm2), sigma_0_fm2=sigma0,
        )
        expected = sigma0 * gbw_dipole_amplitude(1e-4, radius) / (2*np.pi*RHO_HS.B_hs_fm2)
        self.assertEqual(result.shape, (1, 1))
        self.assertGreater(result[0, 0], 2)
        assert_allclose(result[0, 0], expected)

    def test_hotspot_weights_are_normalized_and_nonnegative(self):
        for invalid in ([0.5, 0.6], [-0.1, 1.1], [np.nan, 1]):
            with self.subTest(weights=invalid), self.assertRaises(ValueError):
                proton_dipole_profile(
                    1e-3, 0.1, [[0, 0]], [[0, 0, 0], [1, 0, 0]],
                    weights=invalid, sigma_fm=0.2, sigma_0_fm2=3,
                )


if __name__ == "__main__":
    unittest.main()
