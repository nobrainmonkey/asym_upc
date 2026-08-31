from densities import fermi_radial_pdf
from scipy.integrate import cumulative_trapezoid
from parameters import O16_3PF
from parameters import FermiParameters
import numpy as np

rng = np.random.default_rng(seed=2026)

def fermi_radial_cdf(r_grid_fm, radial_pdf_values):
    r_grid_fm=np.asarray(r_grid_fm,dtype=float)
    radial_pdf_values = np.asarray(radial_pdf_values,dtype=float)
    if r_grid_fm.size != radial_pdf_values.size:
        raise ValueError("Radial grid and radial PDF values have different dimension")
    cdf = cumulative_trapezoid(radial_pdf_values,r_grid_fm, initial=0)
    return cdf/cdf[-1]

def sample_fermi_radii(n_samples, r_grid_fm, radial_cdf, rng=rng):
    uniform_values = rng.random(n_samples)
    return np.interp(uniform_values, radial_cdf, r_grid_fm)

def sample_isotropic_positions(radii_fm, rng):
    radii_fm = np.asarray(radii_fm, dtype=float)

    if np.any(~np.isfinite(radii_fm)):
        raise ValueError("Radii must be finite")

    if np.any(radii_fm < 0):
        raise ValueError("Radii cannot be negative")

    phi = rng.uniform(
        0.0,
        2.0 * np.pi,
        size=radii_fm.shape,
    )

    cos_theta = rng.uniform(
        -1.0,
        1.0,
        size=radii_fm.shape,
    )

    sin_theta = np.sqrt(
        np.clip(1.0 - cos_theta**2, 0.0, None)
    )
    x_fm = radii_fm * sin_theta * np.cos(phi)
    y_fm = radii_fm * sin_theta * np.sin(phi)
    z_fm = radii_fm * cos_theta

    return np.stack(
        (x_fm, y_fm, z_fm),
        axis=-1,
    )

def passes_hard_sphere_exclusion(positions_fm, min_separation_fm):
    positions_fm = np.asarray(positions_fm, dtype=float)
    A = positions_fm.shape[0]
    pair_i, pair_j = np.triu_indices(A, k=1)
    pair_displacements=(positions_fm[pair_i] - positions_fm[pair_j])

    pair_distance_sq=np.sum(pair_displacements**2,axis=1)
    return np.all(pair_distance_sq >= min_separation_fm**2)

def recenter_on_center_of_mass(event_positions_fm):
    event_positions_fm = np.asarray(event_positions_fm, dtype=float)
    if event_positions_fm.ndim != 2 or event_positions_fm.shape[1] != 3:
        raise ValueError("positions_fm must have shape (A, 3)")
    center_of_mass_fm = np.mean(event_positions_fm,axis=0)
    centered_event_positions_fm=event_positions_fm - center_of_mass_fm
    return centered_event_positions_fm

def sample_proton_mask(parameter: FermiParameters, rng):
    proton_indices = rng.choice(
        parameter.A,
        size=parameter.Z,
        replace=False,
    )

    is_proton = np.zeros(parameter.A, dtype=bool)
    is_proton[proton_indices] = True

    return is_proton

def sample_3pf_events(n_events,parameter:FermiParameters,rng, min_seperation_fm = 0.5, max_attempts_per_event=10_000):
    if(min_seperation_fm < 0):
        raise ValueError("Min seperate in fermi cannot be smaller than 0")
    A = parameter.A
    r_grid = np.linspace(0,parameter.r_max_fm,parameter.radial_grid_points)
    fermi_radial_pdf_grid=fermi_radial_pdf(r_grid,parameter)
    fermi_radial_cdf_grid = fermi_radial_cdf(r_grid,fermi_radial_pdf_grid)
    accepted_events = np.empty((n_events,A,3),dtype=float)

    proton_masks = np.empty((n_events, A),dtype=bool,)

    accepted_counts = 0
    attempts = 0
    max_attempts = n_events * max_attempts_per_event
    while accepted_counts < n_events and attempts < max_attempts:
        attempts +=1
        event_candidate_radii_fm = sample_fermi_radii(A,r_grid,fermi_radial_cdf_grid,rng)
        event_candidate_xyz_fm= sample_isotropic_positions(event_candidate_radii_fm,rng)
        if passes_hard_sphere_exclusion(event_candidate_xyz_fm,min_seperation_fm):
            accepted_events[accepted_counts] = recenter_on_center_of_mass(event_candidate_xyz_fm)
            proton_masks[accepted_counts] = sample_proton_mask(parameter,rng)
            accepted_counts+=1

    if accepted_counts<n_events:
        raise RuntimeError(f"Accepted only {accepted_counts}/{n_events} events after {attempts} attempts")
    return accepted_events,proton_masks

events, proton_masks = sample_3pf_events(
    1000,
    O16_3PF,
    rng,
)

assert events.shape == (1000, O16_3PF.A, 3)
assert proton_masks.shape == (1000, O16_3PF.A)
assert np.all(proton_masks.sum(axis=1) == O16_3PF.Z)

proton_positions = np.stack([
    event[mask]
    for event, mask in zip(events, proton_masks)
])

assert proton_positions.shape == (
    1000,
    O16_3PF.Z,
    3,
)

print("3pF nucleon and proton geometry passed")