from dataclasses import dataclass
import math


HBARC_GEV_FM = 0.1973269804  # GeV*fm
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
