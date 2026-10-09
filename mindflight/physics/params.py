"""Crazyflie 2.x physical parameters.

Source: cf2x.urdf in utiasDSL/gym-pybullet-drones, which cites Förster 2015 (ETH system identification of the
Crazyflie 2.0), Landry 2015 (MIT) and Luis & Le Ny 2016. Values not from that source are marked ASSUMED.
"""

import math
from dataclasses import dataclass

GRAVITY_MPS2 = 9.81


@dataclass(frozen=True)
class CrazyflieParams:
    mass_kg: float = 0.027
    arm_m: float = 0.0397
    ixx_kgm2: float = 1.4e-5
    iyy_kgm2: float = 1.4e-5
    izz_kgm2: float = 2.17e-5
    kf_n_per_rpm2: float = 3.16e-10  # thrust per motor = kf * rpm^2
    km_nm_per_rpm2: float = 7.94e-12  # drag torque per motor = km * rpm^2
    thrust_to_weight: float = 2.25
    max_speed_kmh: float = 30.0
    prop_radius_m: float = 2.31348e-2
    drag_coeff_xy: float = 9.1785e-7
    drag_coeff_z: float = 10.311e-7
    ground_effect_coeff: float = 11.36859
    collision_radius_m: float = 0.06
    collision_height_m: float = 0.025
    motor_time_constant_s: float = 0.02  # ASSUMED: not in the source, revisit

    @property
    def weight_n(self) -> float:
        """Force of gravity on the drone, m * g (N)."""
        return self.mass_kg * GRAVITY_MPS2

    @property
    def hover_rpm(self) -> float:
        """RPM where 4 motors exactly hold up the weight: 4 * kf * rpm^2 = m * g."""
        return math.sqrt(self.weight_n / (4 * self.kf_n_per_rpm2))

    @property
    def max_rpm(self) -> float:
        """RPM at full throttle: total thrust = thrust_to_weight * weight."""
        return math.sqrt(self.thrust_to_weight * self.weight_n / (4 * self.kf_n_per_rpm2))

    @property
    def arm_xy_m(self) -> float:
        """d in PLAN.md: each motor's x and y offset in the X layout, arm / sqrt(2)."""
        return self.arm_m / math.sqrt(2)


CRAZYFLIE = CrazyflieParams()
