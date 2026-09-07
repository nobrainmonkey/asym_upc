from scipy.integrate import cumulative_trapezoid
import numpy as np
from scipy.stats import poisson


if __package__:
    from .densities import fermi_radial_pdf
    from .parameters import (
        O16_3PF, O16_ALPHA, RHO_HS,
        AlphaClusterParameters, FermiParameters, HotspotParameters,
    )
else:
    from densities import fermi_radial_pdf
    from parameters import (
        O16_3PF, O16_ALPHA, RHO_HS,
        AlphaClusterParameters, FermiParameters, HotspotParameters,
    )

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

def sample_3pf_events(n_events,parameter:FermiParameters,rng, max_attempts_per_event=10_000):
    min_separation_fm = parameter.min_separation_fm
    if(min_separation_fm < 0):
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
        if passes_hard_sphere_exclusion(event_candidate_xyz_fm,min_separation_fm):
            accepted_events[accepted_counts] = recenter_on_center_of_mass(event_candidate_xyz_fm)
            proton_masks[accepted_counts] = sample_proton_mask(parameter,rng)
            accepted_counts+=1

    if accepted_counts<n_events:
        raise RuntimeError(f"Accepted only {accepted_counts}/{n_events} events after {attempts} attempts")
    return accepted_events,proton_masks


#alpha cluster construction
def regular_tetrahedron_centers(parameter: AlphaClusterParameters):
    if parameter.n_clusters != 4:
        raise ValueError("Only tetrahedron configuration is supported for now")
    v1 = np.array([-np.sqrt(6)/3, -np.sqrt(2)/3, -1/3])
    v2 = np.array([np.sqrt(6)/3, -np.sqrt(2)/3, -1/3])
    v3 = np.array([0, 2*np.sqrt(2)/3, -1/3])
    v4 = np.array([0, 0, 1])
    vertices = np.array([v1, v2, v3,v4])
    vertices *= parameter.cluster_radius_fm
    return vertices

def nucleon_to_cluster_id(parameter: AlphaClusterParameters):
    cluster_ids = np.repeat(np.arange(parameter.n_clusters), parameter.cluster_A)
    return cluster_ids

def sample_uniform_rotation_matrix(rng):
    """Use ZYZ Euler angles to sample a uniform rotation matrix
       phi = rotation about z
       theta = rotation about y
       psi = rotation about z
    """
    psi = rng.uniform(0, 2 * np.pi)
    theta = np.arccos(rng.uniform(-1, 1))
    phi = rng.uniform(0, 2 * np.pi)

    Rz_psi = np.array([[np.cos(psi), -np.sin(psi), 0],
                       [np.sin(psi), np.cos(psi), 0],
                       [0, 0, 1]])
    Rz_phi = np.array([[np.cos(phi), -np.sin(phi), 0],
                       [np.sin(phi), np.cos(phi), 0],
                       [0, 0, 1]])
    Ry_theta = np.array([[np.cos(theta), 0, np.sin(theta)],
                       [0, 1, 0],
                       [-np.sin(theta), 0, np.cos(theta)]])
    return Rz_psi @ Ry_theta @ Rz_phi

def rotate_event_positions(positions_fm, rotation_matrix):
    positions_fm = np.asarray(positions_fm, dtype=float)
    rotation_matrix = np.asarray(rotation_matrix, dtype=float)
    if positions_fm.ndim != 2 or positions_fm.shape[1] != 3:
        raise ValueError("positions_fm must have shape (A, 3)")
    if rotation_matrix.shape != (3, 3):
        raise ValueError("rotation_matrix must have shape (3, 3)")
    return positions_fm @ rotation_matrix.T

def sample_alpha_proton_mask(parameter: AlphaClusterParameters, cluster_ids, rng):
    """
    for an event, a cluster id is assigned to each nucleon, then for each cluster, we randomly select Z/n_clusters nucleons to be protons. cluster_id should have dimension (A,) and contain integers from 0 to n_clusters-1
    """
    cluster_ids = np.asarray(cluster_ids, dtype=int)
    if cluster_ids.shape[0] != parameter.A:
        raise ValueError("cluster_ids must have shape (A,)")
    if np.any(cluster_ids < 0) or np.any(cluster_ids >= parameter.n_clusters):
        raise ValueError("cluster_ids must contain integers from 0 to n_clusters-1")

    is_proton = np.zeros(parameter.A, dtype=bool)
    
    for cluster_id in range(parameter.n_clusters):
        nucleon_indices = np.where(cluster_ids == cluster_id)[0]
        if nucleon_indices.size != parameter.cluster_A:
            raise ValueError(f"Cluster {cluster_id} has {len(nucleon_indices)} nucleons, expected {parameter.cluster_A}")
        proton_indices = rng.choice(
            nucleon_indices,
            size=parameter.cluster_Z,
            replace=False,
        )
        is_proton[proton_indices] = True
    if np.sum(is_proton) != parameter.Z:
        raise ValueError(f"Total number of protons {np.sum(is_proton)} does not match expected Z {parameter.Z}")
    return is_proton
    

def sample_tetrahedron_nucleon_events(n_events,parameter: AlphaClusterParameters, rng, max_attempts_per_event=10000):
    cluster_centers = regular_tetrahedron_centers(parameter)
    cluster_ids = nucleon_to_cluster_id(parameter)
    nucleon_positions = np.empty((parameter.A, 3), dtype=float)
    accepted_counts = 0
    attempts = 0
    max_attempts = n_events * max_attempts_per_event
    all_nucleon_positions = np.empty((n_events, parameter.A, 3), dtype=float)
    all_proton_masks = np.empty((n_events, parameter.A), dtype=bool)
    while accepted_counts < n_events and attempts < max_attempts:
        attempts += 1
        for cluster_id in range(parameter.n_clusters):
            cluster_center = cluster_centers[cluster_id]
            nucleon_indices = np.where(cluster_ids == cluster_id)[0]
            nucleon_positions[nucleon_indices] = rng.normal(
                loc=cluster_center,
                scale=parameter.nucleon_sigma_fm,
                size=(len(nucleon_indices), 3),
            )
        if passes_hard_sphere_exclusion(nucleon_positions, parameter.min_separation_fm):
            nucleon_positions_centered = recenter_on_center_of_mass(nucleon_positions)
            event_rotation_matrix = sample_uniform_rotation_matrix(rng)
            nucleon_positions_centered = rotate_event_positions(nucleon_positions_centered, event_rotation_matrix)
            proton_mask = sample_alpha_proton_mask(parameter, cluster_ids, rng)
            all_proton_masks[accepted_counts] = proton_mask
            all_nucleon_positions[accepted_counts] = nucleon_positions_centered
            accepted_counts += 1
    if accepted_counts < n_events:
        raise RuntimeError(f"Accepted only {accepted_counts}/{n_events} events after {attempts} attempts")
    return all_nucleon_positions, all_proton_masks

def mean_hotspot_count(bj_x, parameters: HotspotParameters):
    """Return the underlying Poisson mean lambda(x) for scalar or array x.

    The mean after rejecting zero is lambda / (1 - exp(-lambda)).
    Reference: https://arxiv.org/html/2312.11320v2
    """
    bj_x = np.asarray(bj_x, dtype=float)
    if np.any(~np.isfinite(bj_x)):
        raise ValueError("bj_x must be finite")
    if np.any(bj_x <= 0.0):
        raise ValueError("bj_x must be positive")

    parameters_array = np.asarray(
        [parameters.p0, parameters.p1, parameters.p2], dtype=float
    )
    if parameters_array.shape != (3,):
        raise ValueError("p0, p1, and p2 must be scalars")
    if np.any(~np.isfinite(parameters_array)):
        raise ValueError("p0, p1, and p2 must be finite")

    p0, p1, p2 = parameters_array
    with np.errstate(over="ignore", invalid="ignore"):
        poisson_mean = p0 * bj_x**p1 * (1.0 + p2 * np.sqrt(bj_x))
    if np.any(~np.isfinite(poisson_mean)) or np.any(poisson_mean <= 0.0):
        raise ValueError("Hotspot parameters must produce a finite positive Poisson mean")
    return poisson_mean

def sample_zero_truncated_poisson(poisson_mean, size, rng):
    """Draw positive counts from a Poisson distribution with a scalar mean."""
    poisson_mean = np.asarray(poisson_mean, dtype=float)
    if poisson_mean.ndim != 0:
        raise ValueError("poisson_mean must be a scalar")
    if not np.isfinite(poisson_mean) or poisson_mean <= 0.0:
        raise ValueError("poisson_mean must be finite and positive")
    poisson_mean = float(poisson_mean)

    probability_zero = np.exp(-poisson_mean)
    u = rng.random(size=size)
    poisson_quantile = probability_zero + u * (-np.expm1(-poisson_mean))
    poisson_quantile = np.clip(
        poisson_quantile,
        np.nextafter(probability_zero, 1.0),
        np.nextafter(1.0, 0.0),
    )
    samples = poisson.ppf(poisson_quantile, poisson_mean).astype(int)
    if np.any(samples < 1):
        raise RuntimeError("Zero-truncated Poisson sampling failed")
    return samples


def sample_target_hotspots(nucleon_positions_fm, bj_x, hotspot_parameters: HotspotParameters, rng):
    """Sample one 3D hotspot configuration from nucleon positions (A, 3).

    Return positions (M, 3), parent IDs (M,), weights (M,), and counts (A,),
    where M is the sum of the counts. Coordinates are in fm; bj_x is scalar.
    """
    nucleon_positions_fm = np.asarray(nucleon_positions_fm, dtype=float)
    if (
        nucleon_positions_fm.ndim != 2
        or nucleon_positions_fm.shape[1] != 3
        or nucleon_positions_fm.shape[0] == 0
    ):
        raise ValueError("nucleon_positions_fm must have shape (A, 3) with A >= 1")
    if np.any(~np.isfinite(nucleon_positions_fm)):
        raise ValueError("nucleon_positions_fm must be finite")

    bj_x = np.asarray(bj_x, dtype=float)
    if bj_x.ndim != 0:
        raise ValueError("bj_x must be a scalar for one target configuration")
    poisson_mean = mean_hotspot_count(bj_x, hotspot_parameters)

    widths_fm2 = np.asarray(
        [hotspot_parameters.B_p_fm2, hotspot_parameters.B_hs_fm2], dtype=float
    )
    if (
        widths_fm2.shape != (2,)
        or np.any(~np.isfinite(widths_fm2))
        or np.any(widths_fm2 <= 0.0)
    ):
        raise ValueError("B_p and B_hs must be finite positive scalars")

    hotspot_positions_fm = []
    hotspot_parent_ids = []
    hotspot_weights = []
    hotspot_counts = []

    for nucleon_index, nucleon_position_fm in enumerate(nucleon_positions_fm):
        n_hotspots = int(sample_zero_truncated_poisson(poisson_mean, 1, rng)[0])
        hotspot_counts.append(n_hotspots)
        for _ in range(n_hotspots):
            hotspot_positions_fm.append(rng.normal(loc=nucleon_position_fm, scale=np.sqrt(hotspot_parameters.B_p_fm2), size=3))
            hotspot_parent_ids.append(nucleon_index)
            hotspot_weights.append(1.0 / n_hotspots)
    return (
        np.asarray(hotspot_positions_fm, dtype=float),
        np.asarray(hotspot_parent_ids, dtype=int),
        np.asarray(hotspot_weights, dtype=float),
        np.asarray(hotspot_counts, dtype=int),
    )


def position_to_gaussian_density(grid_fm, centers_fm, sigma_fm, weights, mode="hotspot"):
    """Evaluate a weighted 3D Gaussian density in fm^-3.

    P = number of evaluation points, M = number of centers
    grid_fm has shape (P,3)
    centers_fm has shape (M,3)
    weights has shape (M,)
    sigma_fm is a finite positive scalar standard deviation in fm.
    The output density has shape (P,)

    mode labels the sources; both modes use the supplied weights.
    Pass np.ones(M) for unit-weight nucleons. The volume integral equals
    weights.sum().
    """
    grid_fm = np.asarray(grid_fm, dtype=float)
    centers_fm = np.asarray(centers_fm, dtype=float)

    if grid_fm.ndim != 2 or grid_fm.shape[1] != 3:
        raise ValueError("grid_fm must have shape (P, 3)")
    if centers_fm.ndim != 2 or centers_fm.shape[1] != 3:
        raise ValueError("centers_fm must have shape (M, 3)")
    if np.any(~np.isfinite(grid_fm)):
        raise ValueError("grid_fm must be finite")
    if np.any(~np.isfinite(centers_fm)):
        raise ValueError("centers_fm must be finite")

    sigma_fm = np.asarray(sigma_fm, dtype=float)
    if sigma_fm.ndim != 0:
        raise ValueError("sigma_fm must be a scalar")
    if not np.isfinite(sigma_fm) or sigma_fm <= 0.0:
        raise ValueError("sigma_fm must be finite and positive")
    sigma_fm = float(sigma_fm)

    if mode not in ["hotspot", "nucleon"]:
        raise ValueError("mode must be either 'hotspot' or 'nucleon'")

    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(centers_fm),):
        raise ValueError("weights must have shape (M,)")
    if np.any(~np.isfinite(weights)):
        raise ValueError("weights must be finite")

    # (P,1,3) - (1,M,3) -> (P,M,3) displacement vectors
    diff = grid_fm[:, None, :] - centers_fm[None, :, :]
    r2 = np.sum(diff**2, axis=-1)                       # (P,M) squared distances

    norm_const = (2.0 * np.pi * sigma_fm**2) ** (-1.5)
    gauss = norm_const * np.exp(-r2 / (2.0 * sigma_fm**2))   # (P,M)

    density_values = gauss @ weights                    # (P,)
    return density_values

def position_to_gaussian_thickness(grid_fm, centers_fm, sigma_fm, weights):
    """Evaluate the full z integral of the 3D Gaussian density in fm^-2.

    Integrating from -infinity to infinity gives a factor sqrt(2*pi)*sigma,
    leaving a normalized 2D Gaussian independent of each center's z position.
    grid_fm has shape (P,2), or (P,3) with only x and y used.
    centers_fm has shape (M,3).
    weights has shape (M,)
    sigma_fm is a finite positive scalar standard deviation in fm.
    The output thickness has shape (P,). Its transverse integral equals
    weights.sum(), for either hotspot weights or unit-weight nucleons.
    """
    grid_fm = np.asarray(grid_fm, dtype=float)
    centers_fm = np.asarray(centers_fm, dtype=float)

    if grid_fm.ndim != 2 or grid_fm.shape[1] not in (2, 3):
        raise ValueError("grid_fm must have shape (P, 2) or (P, 3)")
    if centers_fm.ndim != 2 or centers_fm.shape[1] != 3:
        raise ValueError("centers_fm must have shape (M, 3)")
    if np.any(~np.isfinite(grid_fm)):
        raise ValueError("grid_fm must be finite")
    if np.any(~np.isfinite(centers_fm)):
        raise ValueError("centers_fm must be finite")

    sigma_fm = np.asarray(sigma_fm, dtype=float)
    if sigma_fm.ndim != 0:
        raise ValueError("sigma_fm must be a scalar")
    if not np.isfinite(sigma_fm) or sigma_fm <= 0.0:
        raise ValueError("sigma_fm must be finite and positive")
    sigma_fm = float(sigma_fm)

    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(centers_fm),):
        raise ValueError("weights must have shape (M,)")
    if np.any(~np.isfinite(weights)):
        raise ValueError("weights must be finite")

    diff = grid_fm[:, None, :2] - centers_fm[None, :, :2]  # (P,M,2)
    r2 = np.sum(diff**2, axis=-1)  # (P,M) squared distances in the transverse plane

    norm_const = (2.0 * np.pi * sigma_fm**2) ** (-1.0)  # 2D Gaussian normalization
    gauss = norm_const * np.exp(-r2 / (2.0 * sigma_fm**2))  # (P,M)

    thickness_values = gauss @ weights  # (P,)
    return thickness_values
