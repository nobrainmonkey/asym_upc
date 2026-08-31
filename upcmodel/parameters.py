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

        if self.radial_grid_points < 2:
            raise ValueError("radial_grid_points must be at least 2")

        if self.w < 0:
            edge_numerator = ( 1.0+ self.w * (self.r_max_fm / self.R_fm) ** 2)
            if edge_numerator < 0:
                raise ValueError(
                "The 3pF numerator becomes negative before r_max_fm"
            )

O16_3PF = FermiParameters(
    name="O16_3PF",
    A=16,
    Z=8,
    R_fm=2.608,
    a_fm=0.513,
    w=-0.051,
    r_max_fm=10.0,
    radial_grid_points=20001,
)