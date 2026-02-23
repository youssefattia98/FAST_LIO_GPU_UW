# Pressure Sensor IKF Extension Report

## Summary

This report documents the implementation of pressure-depth fusion in `FAST_LIO_GPU_UW` by extending the existing IKF pipeline used for IMU + LiDAR + DVL.

The pressure topic fused is:

- `/auv/pressure/scaled2` (`sensor_msgs/msg/FluidPressure`)

The implementation was completed on **February 23, 2026**.

## Estimator Model

### 1. State Extension

A new scalar pressure bias state was added:

- `b_pressure` (units: m)

State DOF changed:

- from `30` to `31`

Process noise DOF changed:

- from `15` to `16`

### 2. Process Model

Pressure bias uses a random walk:

- `dot(b_pressure) = n_b_pressure`

with process-noise parameter:

- `mapping.process_noise.nb_pressure`

### 3. Pressure-to-Depth Conversion

Absolute pressure is converted to depth:

```text
depth = (p_abs - p_ref) / (rho * g)
```

where:

- `p_abs` is `FluidPressure.fluid_pressure` [Pa]
- `p_ref` is reference pressure [Pa] (`pressure.reference_pressure_pa`, then auto-initialized from startup samples)
- `rho` is fixed to fresh-water density (`997 kg/m^3`)
- `g` is the current gravity magnitude estimated by the IKF state (`|state.grav|`)

### 4. Measurement Model

Depth measurement is fused as a 1D IKF measurement:

```text
z_p = s_z * e3^T * (p + R * r_bp) + b_pressure + v_p
```

where:

- `p` is IMU position in world
- `R` is body-to-world rotation
- `r_bp` is pressure sensor position wrt IMU/body (`pressure.extrinsic_T`)
- `s_z` is derived from gravity sign (`sign(state.grav_z)`), so no manual sign tuning is needed
- `v_p` is measurement noise

### 5. Jacobians Used

The measurement Jacobian includes:

- wrt position: `H_pos = [0 0 s_z]`
- wrt rotation: `H_rot = s_z * row_z(-R * [r_bp]_x)`
- wrt pressure bias: `H_b_pressure = 1`

### 6. Measurement Covariance

Depth variance uses only a configured floor:

- `R_p = pressure.covariance_floor_std^2`

Given your sensor spec:

- ±200 mbar ≈ ±2.04 m (fresh water, 0..45C)
- ±400 mbar ≈ ±4.09 m (fresh water, -20..85C)
- resolution 0.2 mbar ≈ 0.002 m

Default config uses conservative floor:

- `pressure.covariance_floor_std: 4.09`

### 7. Reference Initialization

Reference pressure is initialized automatically at startup:

- collect the first internal startup pressure samples
- initialize `pressure.reference_pressure_pa` from their mean
- skip pressure updates until reference initialization completes

## Integration Pattern (Matched to DVL Flow)

Pressure was integrated with the same architecture used for DVL:

1. Subscriber callback with loop-back protection and buffer push
2. Sync into `MeasureGroup` per LiDAR frame window
3. Sequential IKF update in timer loop after IMU prediction/update

## Files Changed

- `include/common_lib.h`
  - added `FluidPressure` include
  - added `MeasureGroup::pressure` buffer

- `include/use-ikfom.hpp`
  - added `b_pressure` state and `nb_pressure` process noise
  - upgraded matrix dimensions to new DOF
  - added pressure measurement functions:
    - `set_pressure_cov`
    - `set_pressure_mount`
    - `pressure_meas_predict`
    - `h_pressure_share`

- `src/IMU_Processing.hpp`
  - moved IKF template/noise dimensions to `process_noise_ikfom::DOF`
  - added pressure-bias process noise handling
  - initialized pressure bias state and covariance

- `src/laserMapping.cpp`
  - added pressure topic include, globals, buffering, callback, sync
  - added pressure parameter declaration/loading
  - added pressure mount initialization
  - added pressure EKF update block (conversion, covariance, update)
  - switched estimator dimensions to dynamic DOF constants for robustness
  - updated LiDAR Jacobian matrix width to state DOF

- `config/WaterLinked.yaml`
  - added `mapping.process_noise.nb_pressure`
  - added full `pressure:` config block

## New Parameters

```yaml
mapping:
  process_noise:
    nb_pressure: 0.0001

pressure:
  enable: true
  topic: "/auv/pressure/scaled2"
  extrinsic_T: [0.0, 0.0, 0.0]
  covariance_floor_std: 4.09
  reference_pressure_pa: 101325.0
```

## Notes

- Pressure depth is **not** treated as ground truth.
- Bias modeling was intentionally added because long-term pressure offset and drift dominate over raw resolution.
- The implementation keeps DVL and pressure independent and sequential in the same frame cycle.
