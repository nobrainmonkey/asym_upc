"""Event orchestration checked against analytic and direct-integral amplitudes."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from upcmodel.amplitudes import (
    event_amplitude, gbw_dipole_amplitude, momentum_transfer_grid,
    nuclear_dipole_profile,
)
from upcmodel.geometry import position_to_gaussian_thickness, transverse_grid
from upcmodel.parameters import GBW_PAPER, HBARC_GEV_FM as h, JPSI_GAUS_LC_PARAM
from upcmodel.wavefunctions import gaus_lc_overlap_integral_z


class EventAmplitudeTests(unittest.TestCase):
    def setUp(self):
        self.bx, self.by, self.points = transverse_grid(6.0, 128)
        self.r = np.geomspace(0.002, 3.0, 41)
        self.delta = momentum_transfer_grid(self.bx)
        self.H = gaus_lc_overlap_integral_z(
            JPSI_GAUS_LC_PARAM, self.r, abs(self.delta), 0.7, nz=24,
        )
        self.grid = dict(r_fm=self.r, bx_fm=self.bx, by_fm=self.by)

    def test_smooth_and_hotspot_protons_match_analytic_gaussian_transform(self):
        sources = (
            (np.array([[0.35, -0.2, 1.0]]), np.array([1.0]), 0.55),
            (np.array([[-0.45, 0.2, 1.0], [0.6, -0.3, -2.0]]),
             np.array([0.4, 0.6]), 0.28),
        )
        x, sigma0 = 1e-3, 1.7
        N = gbw_dipole_amplitude(x, self.r)
        for centers, weights, sigma in sources:
            with self.subTest(sources=len(weights)):
                actual = event_amplitude(
                    x, centers, weights, **self.grid, overlap_H=self.H,
                    profile_kwargs=dict(sigma_fm=sigma, sigma_0_fm2=sigma0),
                )
                charge_sum = np.exp(-1j*self.delta[:, None]*centers[:, 0]/h) @ weights
                gaussian_F = (sigma0/h**2 * N[:, None]
                              * np.exp(-0.5*(sigma*self.delta/h)**2)[None, :]
                              * charge_sum[None, :])
                expected = 1j/(2*h**2) * np.trapezoid(
                    self.r[:, None]*self.H*gaussian_F, x=self.r, axis=0,
                )
                assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
                self.assertEqual(actual.shape, self.delta.shape)
                self.assertTrue(np.iscomplexobj(actual))

    def test_nuclear_profile_matches_direct_b_integral(self):
        bx = np.linspace(-4, 4, 64, endpoint=False)
        by = np.linspace(-3, 3, 49)
        r = np.array([0.03, 0.09, 0.25, 0.7])
        delta = momentum_transfer_grid(bx)
        keep = np.abs(delta) <= 0.4
        q = delta[keep]
        H = gaus_lc_overlap_integral_z(JPSI_GAUS_LC_PARAM, r, abs(q), 0.0, nz=24)
        centers = np.array([[-0.5, 0.1, 0.4], [0.7, -0.2, -0.3]])
        weights = np.ones(2)
        settings = dict(gbw_parameters=GBW_PAPER, sigma_fm=0.6, sigma_0_fm2=0.9, A=2)
        actual = event_amplitude(
            2e-4, centers, weights, r_fm=r, bx_fm=bx, by_fm=by,
            overlap_H=H, momentum_mask=keep,
            profile=nuclear_dipole_profile, profile_kwargs=settings,
        )
        X, Y = np.meshgrid(bx, by, indexing="ij")
        D = nuclear_dipole_profile(
            2e-4, r, np.column_stack((X.ravel(), Y.ravel())), centers,
            weights=weights, **settings,
        ).reshape(len(r), len(bx), len(by))
        phase = np.exp(-1j*bx[:, None]*q[None, :]/h)
        integrand = D[:, :, :, None] * phase[None, :, None, :]
        direct_F = np.trapezoid(integrand, x=by, axis=2).sum(axis=1)*(bx[1]-bx[0])/h**2
        expected = 1j/(2*h**2)*np.trapezoid(r[:, None]*H*direct_F, x=r, axis=0)
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_custom_profile_has_no_gbw_dependency(self):
        def custom_profile(x, radii, points, centers, *, weights, strength):
            thickness = position_to_gaussian_thickness(points, centers, 0.5, weights)
            return strength*x**(-0.2)*radii[:, None]**2*thickness[None, :]

        x, strength = 0.002, 2.7
        actual = event_amplitude(
            x, [[0, 0, 0]], [1], **self.grid, overlap_H=self.H,
            profile=custom_profile, profile_kwargs=dict(strength=strength),
        )
        expected = (1j*strength*x**(-0.2)/(2*h**4)
                    * np.exp(-0.5*(0.5*self.delta/h)**2)
                    * np.trapezoid(self.r[:, None]**3*self.H, x=self.r, axis=0))
        assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)

    def test_mask_selects_matching_overlap_and_fft_columns(self):
        options = dict(**self.grid, profile_kwargs=dict(sigma_fm=0.5, sigma_0_fm2=1.7))
        full = event_amplitude(1e-3, [[0.4, 0.1, 0]], [1], overlap_H=self.H, **options)
        mask = np.zeros(len(self.delta), dtype=bool)
        mask[[61, 64, 66, 68]] = True
        selected = event_amplitude(
            1e-3, [[0.4, 0.1, 0]], [1], overlap_H=self.H[:, mask],
            momentum_mask=mask, **options,
        )
        assert_allclose(selected, full[mask], rtol=1e-14, atol=1e-15)
        with self.assertRaisesRegex(ValueError, "H_T and F"):
            event_amplitude(
                1e-3, [[0.4, 0.1, 0]], [1], overlap_H=self.H,
                momentum_mask=mask, **options,
            )

    def test_invalid_momentum_selections_are_rejected(self):
        for mask in ([0, 1], True, [True, False], np.zeros(128, dtype=bool)):
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "momentum_mask"):
                event_amplitude(
                    1e-3, [[0, 0, 0]], [1], **self.grid, overlap_H=self.H,
                    momentum_mask=mask,
                )


if __name__ == "__main__":
    unittest.main()
