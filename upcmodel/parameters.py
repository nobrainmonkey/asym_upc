from dataclasses import dataclass
import math

import numpy as np


HBARC_GEV_FM = 0.1973269804  # GeV*fm
ALPHA_EM = 1/137.036
E_EM = math.sqrt(4. * math.pi * ALPHA_EM)
@dataclass(frozen=True)
class FermiParameters:
    name: str
    A: int
    Z: int
    R_fm: float
    a_fm: float
    w: float
    r_max_fm: float
    radial_grid_points: int
    min_separation_fm: float

    def __post_init__(self):
        if not math.isfinite(self.R_fm) or self.R_fm <= 0:
            raise ValueError("R_fm must be finite and positive")

        if not math.isfinite(self.a_fm) or self.a_fm <= 0:
            raise ValueError("a_fm must be finite and positive")

        if self.A <= 0:
            raise ValueError("A must be positive")

        if not 0 <= self.Z <= self.A:
            raise ValueError("Z must satisfy 0 <= Z <= A")

        if not math.isfinite(self.w):
            raise ValueError("w must be finite")

        if not math.isfinite(self.r_max_fm) or self.r_max_fm <= 0:
            raise ValueError("r_max_fm must be finite and positive")

        if not math.isfinite(self.min_separation_fm) or self.min_separation_fm < 0:
            raise ValueError("min_separation_fm must be finite and non-negative")

        if self.radial_grid_points < 2:
            raise ValueError("radial_grid_points must be at least 2")
    

        if self.w < 0:
            edge_numerator = ( 1.0+ self.w * (self.r_max_fm / self.R_fm) ** 2)
            if edge_numerator < 0:
                raise ValueError(
                "The 3pF numerator becomes negative before r_max_fm"
            )

@dataclass(frozen=True)
class AlphaClusterParameters:
    name: str
    A: int
    Z: int
    n_clusters: int
    cluster_A: int
    cluster_Z: int
    min_separation_fm: float
    cluster_radius_fm: float # disance from nuclear center to each cluster center
    nucleon_sigma_fm: float # width of the gaussian distribution of nucleons around each cluster center

    def __post_init__(self):
        if self.A != self.n_clusters * self.cluster_A:
            raise ValueError("A must equal n_clusters * cluster_A")
        if self.Z != self.n_clusters * self.cluster_Z:
            raise ValueError("Z must equal n_clusters * cluster_Z")
        if self.cluster_radius_fm <= 0:
            raise ValueError("cluster_radius_fm must be positive")
        if self.nucleon_sigma_fm <= 0:
            raise ValueError("nucleon_sigma_fm must be positive")
        if self.min_separation_fm < 0:
            raise ValueError("min_separation_fm must be non-negative")
@dataclass(frozen=True)
class HotspotParameters:
    name:str
    p0: float
    p1: float
    p2: float
    B_p_GeV_m2: float
    B_hs_GeV_m2: float
    @property
    def B_p_fm2(self):
        return self.B_p_GeV_m2 * HBARC_GEV_FM**2
    @property
    def B_hs_fm2(self):
        return self.B_hs_GeV_m2 * HBARC_GEV_FM**2

O16_3PF = FermiParameters(
    name="O16_3PF",
    A=16,
    Z=8,
    R_fm=2.608,
    a_fm=0.513,
    w=-0.051,
    r_max_fm=10.0,
    radial_grid_points=20001,
    min_separation_fm=0.5,
)

O16_ALPHA = AlphaClusterParameters(
    name="O16_ALPHA",
    A=16,
    Z=8,
    n_clusters=4,
    cluster_A=4,
    cluster_Z=2,
    min_separation_fm=0.5,
    cluster_radius_fm=2.1,
    nucleon_sigma_fm= 1.7/math.sqrt(3)
)

RHO_HS= HotspotParameters(
    name="RHO_HS",
    p0=0.015,
    p1=-0.58,
    p2=300,
    B_p_GeV_m2 = 6.4,
    B_hs_GeV_m2 = 0.8,
)

JPSI_HS= HotspotParameters(
    name="JPSI_HS",
    p0=0.015,
    p1=-0.58,
    p2=300,
    B_p_GeV_m2 = 4.75,
    B_hs_GeV_m2 = 0.8,
)

@dataclass(frozen=True)
class GBWParameters:
    """GBW scale parameters, with Q0 stored in GeV as in the paper."""

    name:str
    Q0_GeV: float
    x0: float
    lambda_gbw: float

    def __post_init__(self):
        if not math.isfinite(self.Q0_GeV) or self.Q0_GeV <= 0:
            raise ValueError("Q0_GeV must be finite and positive")
        if not math.isfinite(self.x0) or self.x0 <= 0 or self.x0 > 1:
            raise ValueError("x0 must be finite and in (0, 1]")
        if not math.isfinite(self.lambda_gbw) or self.lambda_gbw <= 0:
            raise ValueError("lambda_gbw must be finite and positive")

GBW_PAPER = GBWParameters(
    name="GBW_PAPER",
    Q0_GeV=1.0,
    x0=2e-4,
    lambda_gbw=0.21,
)


@dataclass(frozen=True)
class GausLCParameters:
    """Transverse Gaus-LC wavefunction parameters; m_f is the quark mass."""

    m_f_GeV: float
    N_T: float                     # Dimensionless normalization.
    R_T_Squared_GeV_m2: float       # R_T^2 in GeV^-2; already squared.
    eff_charge: float              # Dimensionless effective flavor charge.

    def __post_init__(self):
        positive_fields = ("m_f_GeV", "N_T", "R_T_Squared_GeV_m2")
        for name in positive_fields:
            value = np.asarray(getattr(self, name), dtype=float)
            if value.ndim != 0 or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive scalar")

        charge = np.asarray(self.eff_charge, dtype=float)
        if charge.ndim != 0 or not np.isfinite(charge):
            raise ValueError("eff_charge must be a finite scalar")


# Meson masses and Gaus-LC parameters from Kowalski, Motyka and Watt,
# hep-ph/0606272, Table 1: https://arxiv.org/html/hep-ph/0606272
# These masses enter x=(Q^2+M_V^2)/(Q^2+W^2); m_f below enters the overlap.
JPSI_MASS_GEV = 3.097
RHO_MASS_GEV = 0.776

JPSI_GAUS_LC_PARAM = GausLCParameters(
    m_f_GeV=1.4,
    N_T=1.23,
    R_T_Squared_GeV_m2=6.5,
    eff_charge=2 / 3,
)

RHO_GAUS_LC_PARAM = GausLCParameters(
    m_f_GeV=0.14,
    N_T=4.47,
    R_T_Squared_GeV_m2=21.9,
    eff_charge=1 / np.sqrt(2),
)
