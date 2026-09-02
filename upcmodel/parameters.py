from dataclasses import dataclass
import math


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