"""Point-charge EPA limits, geometry, and integrated-flux normalization."""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import k0, k1

from upcmodel.epa import epa_flux_per_log_energy, point_charge_epa_field
from upcmodel.parameters import ALPHA_EM, HBARC_GEV_FM


class PointChargeEPATests(unittest.TestCase):
    def test_soft_photon_limit_has_the_coulomb_direction_and_units(self):
        points = np.array([[6.0, 8.0], [-6.0, -8.0], [0.0, 20.0]])
        gamma, charge = 100.0, 8
        beta = np.sqrt(1.0 - gamma**-2)
        expected = (charge*np.sqrt(ALPHA_EM)/(np.pi*beta)
                    * points / np.sum(points**2, axis=1)[:, None])
        # The second case makes xi underflow: the analytic limit stays finite.
        for energy in (1e-12, np.nextafter(0.0, 1.0)):
            field = point_charge_epa_field(energy, gamma, points, charge)
            assert_allclose(field, expected, rtol=2e-15, atol=0)
            self.assertEqual(field.shape, (3, 2))
            self.assertTrue(np.iscomplexobj(field))

    def test_rotation_charge_scaling_and_coherent_cancellation(self):
        points = np.array([[4.0, -7.0], [0.0, 13.0], [-16.0, 2.0]])
        theta = 0.43
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        field = point_charge_epa_field(2.0, 100.0, points, 1)
        rotated = point_charge_epa_field(2.0, 100.0, points @ rotation.T, 1)
        assert_allclose(rotated, field @ rotation.T, rtol=1e-14, atol=1e-17)
        ion_field = point_charge_epa_field(2.0, 100.0, points, 8)
        assert_allclose(ion_field, 8*field, rtol=1e-15)
        assert_allclose(epa_flux_per_log_energy(ion_field),
                        64*epa_flux_per_log_energy(field), rtol=1e-15)
        opposite = point_charge_epa_field(2.0, 100.0, -points, 1)
        assert_allclose(epa_flux_per_log_energy(field + opposite), 0, atol=0)
        assert_allclose(point_charge_epa_field(2.0, 100.0, points, 0), 0, atol=0)

    def test_integrated_flux_matches_the_analytic_sharp_cut(self):
        charge, cutoff_fm = 8, 7.0
        for gamma, xi0 in ((3.0, 0.02), (100.0, 0.5), (2700.0, 3.0)):
            with self.subTest(gamma=gamma, xi0=xi0):
                beta = np.sqrt(1.0 - gamma**-2)
                a = xi0/cutoff_fm
                energy = a*gamma*beta*HBARC_GEV_FM

                def radial_integrand(xi):
                    radius = xi/a
                    field = point_charge_epa_field(energy, gamma, [[radius, 0]], charge)
                    flux = epa_flux_per_log_energy(field)[0]
                    return 2*np.pi*radius*flux/a

                numerical, _ = quad(radial_integrand, xi0, np.inf,
                                    epsabs=1e-12, epsrel=1e-10)
                analytic = 2*charge**2*ALPHA_EM/(np.pi*beta**2) * (
                    xi0*k0(xi0)*k1(xi0)
                    - 0.5*xi0**2*(k1(xi0)**2 - k0(xi0)**2)
                )
                assert_allclose(numerical, analytic, rtol=1e-9, atol=1e-13)

    def test_large_distance_field_has_the_bessel_asymptotic_tail(self):
        energy, gamma, charge = 2.0, 100.0, 8
        beta = np.sqrt(1.0 - gamma**-2)
        a = energy/(gamma*beta*HBARC_GEV_FM)
        xi = 50.0
        radius = xi/a
        field = point_charge_epa_field(energy, gamma, [[radius, 0]], charge)
        coulomb = charge*np.sqrt(ALPHA_EM)/(np.pi*beta*radius)
        asymptotic = np.sqrt(np.pi*xi/2)*np.exp(-xi)*(1 + 3/(8*xi))
        assert_allclose(field[0, 0]/coulomb, asymptotic, rtol=5e-5, atol=0)
        tail = point_charge_epa_field(energy, gamma, [[1e5/a, 0]], charge)
        assert_allclose(tail, 0, atol=0)

    def test_complex_flux_preserves_batch_axes_and_common_phase(self):
        field = np.array([[1+2j, 3-4j], [1j, 2]]).reshape(1, 2, 1, 2)
        expected = np.array([30.0, 5.0]).reshape(1, 2, 1)
        assert_allclose(epa_flux_per_log_energy(field), expected)
        assert_allclose(epa_flux_per_log_energy(field*np.exp(0.73j)), expected)
        self.assertEqual(epa_flux_per_log_energy(field).shape, (1, 2, 1))

    def test_singular_or_invalid_inputs_are_rejected(self):
        defaults = dict(omega_GeV=1.0, gamma=100.0,
                        observation_xy_fm=[[10.0, 0.0]], Z=8)
        invalid = {
            "omega_GeV": [0, -1, np.nan, np.inf, [1], 1+1j],
            "gamma": [0, 1, -2, np.inf, [100], 100+1j],
            "Z": [-1, np.nan, np.inf, [8], 8+1j],
            "observation_xy_fm": [[], [10, 0], [[1, 2, 3]], [[0, 0]],
                                  [[np.nan, 0]], [[np.inf, 0]], [[1j, 2]]],
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    point_charge_epa_field(**{**defaults, name: value})
        for field in ([], 1, [[1, 2, 3]], [[1, np.nan]], [[np.inf, 0]]):
            with self.subTest(field=field), self.assertRaises(ValueError):
                epa_flux_per_log_energy(field)


if __name__ == "__main__":
    unittest.main()
