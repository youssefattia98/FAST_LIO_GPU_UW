#ifndef USE_IKFOM_H
#define USE_IKFOM_H

#include <IKFoM_toolkit/esekfom/esekfom.hpp>
#include <algorithm>
#include <cmath>
#include "dynamics_bridge.hpp"

typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

vect3 SO3ToEuler(const SO3 &orient);

inline Eigen::Vector3d g_imu_accel_noise_diag = Eigen::Vector3d(0.1, 0.1, 0.1);
inline Eigen::Vector3d g_imu_gyro_noise_diag = Eigen::Vector3d(0.1, 0.1, 0.1);
inline Eigen::Vector3d g_thruster_accel_noise_diag = Eigen::Vector3d(0.1, 0.1, 0.1);
inline Eigen::Matrix3d g_dvl_meas_cov = Eigen::Matrix3d::Identity();
inline Eigen::Matrix3d g_dvl_R_b_d = Eigen::Matrix3d::Identity();
inline Eigen::Vector3d g_dvl_r_bd_b = Eigen::Vector3d::Zero();
inline double g_pressure_meas_var = 1.0;
inline Eigen::Vector3d g_pressure_r_bp_b = Eigen::Vector3d::Zero();

inline void set_imu_accel_noise_diag(const Eigen::Vector3d &diag)
{
	g_imu_accel_noise_diag = diag;
}

inline void set_imu_gyro_noise_diag(const Eigen::Vector3d &diag)
{
	g_imu_gyro_noise_diag = diag;
}

inline void set_thruster_accel_noise_diag(const Eigen::Vector3d &diag)
{
	g_thruster_accel_noise_diag = diag;
}

inline void set_dvl_cov(const Eigen::Matrix3d &cov)
{
	g_dvl_meas_cov = cov;
}

inline void set_dvl_mount(const Eigen::Vector3d &r_bd_b, const Eigen::Matrix3d &R_b_d)
{
	g_dvl_r_bd_b = r_bd_b;
	g_dvl_R_b_d = R_b_d;
}

inline void set_pressure_cov(const double var)
{
	g_pressure_meas_var = std::max(var, 1e-9);
}

inline void set_pressure_mount(const Eigen::Vector3d &r_bp_b)
{
	g_pressure_r_bp_b = r_bp_b;
}

MTK_BUILD_MANIFOLD(state_ikfom,
((vect3, pos))
((SO3, rot))
((SO3, offset_R_L_I))
((vect3, offset_T_L_I))
((vect3, vel))
((vect3, omega))
((vect3, bg))
((vect3, ba))
((vect3, grav))
((vect3, b_dvl))
((vect1, b_pressure))
);

MTK_BUILD_MANIFOLD(input_ikfom,
((vect3, acc))
((vect3, gyro))
);

MTK_BUILD_MANIFOLD(process_noise_ikfom,
((vect3, nv))
((vect3, nw))
((vect3, nbg))
((vect3, nba))
((vect3, nb_dvl))
((vect1, nb_pressure))
);

constexpr int kStateDof = state_ikfom::DOF;
constexpr int kProcessNoiseDof = process_noise_ikfom::DOF;

MTK::get_cov<process_noise_ikfom>::type process_noise_cov()
{
	MTK::get_cov<process_noise_ikfom>::type cov = MTK::get_cov<process_noise_ikfom>::type::Zero();
	MTK::setDiagonal<process_noise_ikfom, vect3, 0>(cov, &process_noise_ikfom::nv, 0.0001);// linear accel noise
	MTK::setDiagonal<process_noise_ikfom, vect3, 3>(cov, &process_noise_ikfom::nw, 0.0001); // angular accel noise
	MTK::setDiagonal<process_noise_ikfom, vect3, 6>(cov, &process_noise_ikfom::nbg, 0.00001); // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
	MTK::setDiagonal<process_noise_ikfom, vect3, 9>(cov, &process_noise_ikfom::nba, 0.00001);   //0.001 0.05 0.0001/out 0.01
	MTK::setDiagonal<process_noise_ikfom, vect3, 12>(cov, &process_noise_ikfom::nb_dvl, 0.00001);
	MTK::setDiagonal<process_noise_ikfom, vect1, 15>(cov, &process_noise_ikfom::nb_pressure, 0.00001);
	return cov;
}

//double L_offset_to_I[3] = {0.04165, 0.02326, -0.0284}; // Avia 
//vect3 Lidar_offset_to_IMU(L_offset_to_I, 3);
Eigen::Matrix<double, kStateDof, 1> get_f(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, kStateDof, 1> res = Eigen::Matrix<double, kStateDof, 1>::Zero();

	vect3 omega_meas;
	in.gyro.boxminus(omega_meas, s.bg);
	Eigen::Vector3d omega_body(s.omega[0], s.omega[1], s.omega[2]);
	Eigen::Vector3d vel_body(s.vel[0], s.vel[1], s.vel[2]);
	Eigen::Vector3d rot_rate_body = omega_body;

	// Kinematics
	Eigen::Vector3d pos_dot = s.rot.toRotationMatrix() * vel_body;

	// Dynamics (body frame)
	Eigen::Vector3d vel_dot = Eigen::Vector3d::Zero();
	Eigen::Vector3d omega_dot = Eigen::Vector3d::Zero();
	bool dynamics_used = false;
	if (fastlio::dynamics::has_model())
	{
		vect3 euler_deg = SO3ToEuler(s.rot);
		constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
		Eigen::Vector3d euler_rad(euler_deg[0] * kDegToRad, euler_deg[1] * kDegToRad, euler_deg[2] * kDegToRad);
		Eigen::Matrix<double, 6, 1> pose_world6;
		pose_world6 << s.pos[0], s.pos[1], s.pos[2], euler_rad(0), euler_rad(1), euler_rad(2);

		Eigen::Matrix<double, 6, 1> vel_body6;
		vel_body6 << vel_body(0), vel_body(1), vel_body(2), omega_body(0), omega_body(1), omega_body(2);

		Eigen::Matrix<double, 6, 1> accel_body6;
		if (fastlio::dynamics::compute_body_accel(vel_body6, pose_world6, accel_body6))
		{
			vel_dot = accel_body6.head<3>();
			omega_dot = accel_body6.tail<3>();
			dynamics_used = true;
		}
	}

	if (!dynamics_used)
	{
		rot_rate_body = Eigen::Vector3d(omega_meas[0], omega_meas[1], omega_meas[2]);
		Eigen::Vector3d grav_world(s.grav[0], s.grav[1], s.grav[2]);
		Eigen::Vector3d grav_body = s.rot.toRotationMatrix().transpose() * grav_world;
		Eigen::Vector3d specific_force = Eigen::Vector3d(in.acc[0], in.acc[1], in.acc[2]) - Eigen::Vector3d(s.ba[0], s.ba[1], s.ba[2]);
		vel_dot = specific_force - omega_body.cross(vel_body) + grav_body;
		omega_dot = Eigen::Vector3d::Zero();
	}

	for (int i = 0; i < 3; ++i)
	{
		res(i) = pos_dot(i);
		res(i + 3) = rot_rate_body(i);
	}

	for (int i = 0; i < 3; ++i)
	{
		res(i + 12) = vel_dot(i);
		res(i + 15) = omega_dot(i);
	}
	return res;
}

Eigen::Matrix<double, kStateDof, kStateDof> df_dx(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, kStateDof, kStateDof> cov = Eigen::Matrix<double, kStateDof, kStateDof>::Zero();

	// pos_dot = R * v_body
	Eigen::Vector3d vel_body_eig(s.vel[0], s.vel[1], s.vel[2]);
	vect3 vel_body;
	vel_body << vel_body_eig(0), vel_body_eig(1), vel_body_eig(2);
	cov.template block<3, 3>(0, 12) = s.rot.toRotationMatrix();
	cov.template block<3, 3>(0, 3) = -s.rot.toRotationMatrix() * MTK::hat(vel_body);

	// rot_dot = omega_body
	cov.template block<3, 3>(3, 15) = Eigen::Matrix3d::Identity();

	// vel_dot depends on accel bias and gravity (fallback model)
	cov.template block<3, 3>(12, 21) = -Eigen::Matrix3d::Identity();
	cov.template block<3, 3>(12, 24) = s.rot.toRotationMatrix().transpose();

	return cov;
}


Eigen::Matrix<double, kStateDof, kProcessNoiseDof> df_dw(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, kStateDof, kProcessNoiseDof> cov = Eigen::Matrix<double, kStateDof, kProcessNoiseDof>::Zero();
	// velocity and omega process noise
	cov.template block<3, 3>(12, 0) = Eigen::Matrix3d::Identity();
	cov.template block<3, 3>(15, 3) = Eigen::Matrix3d::Identity();
	// bias random walks
	cov.template block<3, 3>(18, 6) = Eigen::Matrix3d::Identity();
	cov.template block<3, 3>(21, 9) = Eigen::Matrix3d::Identity();
	// dvl bias random walk
	cov.template block<3, 3>(27, 12) = Eigen::Matrix3d::Identity();
	// pressure bias random walk
	cov(30, 15) = 1.0;
	return cov;
}

vect3 SO3ToEuler(const SO3 &orient) 
{
	Eigen::Matrix<double, 3, 1> _ang;
	Eigen::Vector4d q_data = orient.coeffs().transpose();
	//scalar w=orient.coeffs[3], x=orient.coeffs[0], y=orient.coeffs[1], z=orient.coeffs[2];
	double sqw = q_data[3]*q_data[3];
	double sqx = q_data[0]*q_data[0];
	double sqy = q_data[1]*q_data[1];
	double sqz = q_data[2]*q_data[2];
	double unit = sqx + sqy + sqz + sqw; // if normalized is one, otherwise is correction factor
	double test = q_data[3]*q_data[1] - q_data[2]*q_data[0];

	if (test > 0.49999*unit) { // singularity at north pole
	
		_ang << 2 * std::atan2(q_data[0], q_data[3]), M_PI/2, 0;
		double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
		vect3 euler_ang(temp, 3);
		return euler_ang;
	}
	if (test < -0.49999*unit) { // singularity at south pole
		_ang << -2 * std::atan2(q_data[0], q_data[3]), -M_PI/2, 0;
		double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
		vect3 euler_ang(temp, 3);
		return euler_ang;
	}
		
	_ang <<
			std::atan2(2*q_data[0]*q_data[3]+2*q_data[1]*q_data[2] , -sqx - sqy + sqz + sqw),
			std::asin (2*test/unit),
			std::atan2(2*q_data[2]*q_data[3]+2*q_data[1]*q_data[0] , sqx - sqy - sqz + sqw);
	double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
	vect3 euler_ang(temp, 3);
		// euler_ang[0] = roll, euler_ang[1] = pitch, euler_ang[2] = yaw
	return euler_ang;
}

inline vect3 h_imu_accel_share(state_ikfom &s, esekfom::dyn_runtime_share_datastruct<double> &dyn_share)
{
	dyn_share.h_x = Eigen::Matrix<double, 3, kStateDof>::Zero();
	dyn_share.h_v = Eigen::Matrix<double, 3, 3>::Identity();
	dyn_share.R = g_imu_accel_noise_diag.asDiagonal();

	Eigen::Vector3d vel_body(s.vel[0], s.vel[1], s.vel[2]);
	Eigen::Vector3d omega_body(s.omega[0], s.omega[1], s.omega[2]);

	Eigen::Matrix<double, 6, 1> vel_body6;
	vel_body6 << vel_body(0), vel_body(1), vel_body(2), omega_body(0), omega_body(1), omega_body(2);

	vect3 euler_deg = SO3ToEuler(s.rot);
	constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
	Eigen::Vector3d euler_rad(euler_deg[0] * kDegToRad, euler_deg[1] * kDegToRad, euler_deg[2] * kDegToRad);
	Eigen::Matrix<double, 6, 1> pose_world6;
	pose_world6 << s.pos[0], s.pos[1], s.pos[2], euler_rad(0), euler_rad(1), euler_rad(2);

	Eigen::Vector3d accel_body = Eigen::Vector3d::Zero();
	Eigen::Matrix<double, 6, 1> accel_body6;
	if (fastlio::dynamics::compute_body_accel(vel_body6, pose_world6, accel_body6))
	{
		accel_body = accel_body6.head<3>();
	}

	Eigen::Vector3d grav_world(s.grav[0], s.grav[1], s.grav[2]);
	Eigen::Vector3d grav_body = s.rot.toRotationMatrix().transpose() * grav_world;

	Eigen::Vector3d specific_force = accel_body + omega_body.cross(vel_body) - grav_body;
	Eigen::Vector3d accel_meas_pred = specific_force + Eigen::Vector3d(s.ba[0], s.ba[1], s.ba[2]);

	// Jacobian w.r.t. accel bias
	dyn_share.h_x.template block<3, 3>(0, 21) = Eigen::Matrix3d::Identity();

	// Jacobian w.r.t. gravity vector
	dyn_share.h_x.template block<3, 3>(0, 24) = -s.rot.toRotationMatrix().transpose();

	double temp[3] = {accel_meas_pred(0), accel_meas_pred(1), accel_meas_pred(2)};
	vect3 out(temp, 3);
	return out;
}

inline vect3 h_thruster_accel_share(state_ikfom &s, esekfom::dyn_runtime_share_datastruct<double> &dyn_share)
{
	dyn_share.h_x = Eigen::Matrix<double, 3, kStateDof>::Zero();
	dyn_share.h_v = Eigen::Matrix<double, 3, 3>::Identity();
	dyn_share.R = g_thruster_accel_noise_diag.asDiagonal();

	Eigen::Vector3d vel_body(s.vel[0], s.vel[1], s.vel[2]);
	Eigen::Vector3d omega_body(s.omega[0], s.omega[1], s.omega[2]);

	Eigen::Matrix<double, 6, 1> vel_body6;
	vel_body6 << vel_body(0), vel_body(1), vel_body(2), omega_body(0), omega_body(1), omega_body(2);

	vect3 euler_deg = SO3ToEuler(s.rot);
	constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
	Eigen::Vector3d euler_rad(euler_deg[0] * kDegToRad, euler_deg[1] * kDegToRad, euler_deg[2] * kDegToRad);
	Eigen::Matrix<double, 6, 1> pose_world6;
	pose_world6 << s.pos[0], s.pos[1], s.pos[2], euler_rad(0), euler_rad(1), euler_rad(2);

	Eigen::Vector3d accel_body = Eigen::Vector3d::Zero();
	Eigen::Matrix<double, 6, 1> accel_body6;
	if (fastlio::dynamics::compute_body_accel(vel_body6, pose_world6, accel_body6))
	{
		accel_body = accel_body6.head<3>();
	}

	Eigen::Vector3d grav_world(s.grav[0], s.grav[1], s.grav[2]);
	Eigen::Vector3d grav_body = s.rot.toRotationMatrix().transpose() * grav_world;

	Eigen::Vector3d specific_force = accel_body + omega_body.cross(vel_body) - grav_body;

	// Jacobian w.r.t. gravity vector
	dyn_share.h_x.template block<3, 3>(0, 24) = -s.rot.toRotationMatrix().transpose();

	double temp[3] = {specific_force(0), specific_force(1), specific_force(2)};
	vect3 out(temp, 3);
	return out;
}

inline vect3 h_imu_gyro_share(state_ikfom &s, esekfom::dyn_runtime_share_datastruct<double> &dyn_share)
{
	dyn_share.h_x = Eigen::Matrix<double, 3, kStateDof>::Zero();
	dyn_share.h_v = Eigen::Matrix<double, 3, 3>::Identity();
	dyn_share.R = g_imu_gyro_noise_diag.asDiagonal();

	Eigen::Vector3d gyro_pred(s.omega[0] + s.bg[0], s.omega[1] + s.bg[1], s.omega[2] + s.bg[2]);

	// Jacobian w.r.t. omega and gyro bias
	dyn_share.h_x.template block<3, 3>(0, 15) = Eigen::Matrix3d::Identity();
	dyn_share.h_x.template block<3, 3>(0, 18) = Eigen::Matrix3d::Identity();

	double temp[3] = {gyro_pred(0), gyro_pred(1), gyro_pred(2)};
	vect3 out(temp, 3);
	return out;
}

inline vect3 h_dvl_share(state_ikfom &s, esekfom::dyn_runtime_share_datastruct<double> &dyn_share)
{
	dyn_share.h_x = Eigen::Matrix<double, 3, kStateDof>::Zero();
	dyn_share.h_v = Eigen::Matrix<double, 3, 3>::Identity();
	dyn_share.R = g_dvl_meas_cov;

	Eigen::Vector3d vel_body(s.vel[0], s.vel[1], s.vel[2]);
	Eigen::Vector3d omega_body(s.omega[0], s.omega[1], s.omega[2]);
	Eigen::Vector3d v_dvl_body = vel_body + omega_body.cross(g_dvl_r_bd_b);
	Eigen::Vector3d z_pred = g_dvl_R_b_d * v_dvl_body + Eigen::Vector3d(s.b_dvl[0], s.b_dvl[1], s.b_dvl[2]);

	vect3 r_bd;
	r_bd << g_dvl_r_bd_b(0), g_dvl_r_bd_b(1), g_dvl_r_bd_b(2);
	Eigen::Matrix3d skew_r = MTK::hat(r_bd);

	dyn_share.h_x.template block<3, 3>(0, 12) = g_dvl_R_b_d;
	dyn_share.h_x.template block<3, 3>(0, 15) = -g_dvl_R_b_d * skew_r;
	dyn_share.h_x.template block<3, 3>(0, 27) = Eigen::Matrix3d::Identity();

	double temp[3] = {z_pred(0), z_pred(1), z_pred(2)};
	vect3 out(temp, 3);
	return out;
}

inline Eigen::Vector3d pressure_down_axis_world(const state_ikfom &s)
{
	Eigen::Vector3d g_world(s.grav[0], s.grav[1], s.grav[2]);
	const double g_norm = g_world.norm();
	if (g_norm < 1e-6)
	{
		// Fallback: world Z-up assumption -> down axis is -Z.
		return Eigen::Vector3d(0.0, 0.0, -1.0);
	}
	return g_world / g_norm;
}

inline double pressure_meas_predict(const state_ikfom &s)
{
	Eigen::Vector3d p_sensor_world = Eigen::Vector3d(s.pos[0], s.pos[1], s.pos[2]) +
	                                 s.rot.toRotationMatrix() * g_pressure_r_bp_b;
	const Eigen::Vector3d down_axis_world = pressure_down_axis_world(s);
	return down_axis_world.dot(p_sensor_world) + s.b_pressure[0];
}

inline vect1 h_pressure_share(state_ikfom &s, esekfom::dyn_runtime_share_datastruct<double> &dyn_share)
{
	dyn_share.h_x = Eigen::Matrix<double, 1, kStateDof>::Zero();
	dyn_share.h_v = Eigen::Matrix<double, 1, 1>::Identity();
	dyn_share.R = Eigen::Matrix<double, 1, 1>::Constant(g_pressure_meas_var);

	vect3 r_bp;
	r_bp << g_pressure_r_bp_b(0), g_pressure_r_bp_b(1), g_pressure_r_bp_b(2);
	Eigen::Matrix3d d_Rr_dtheta = -s.rot.toRotationMatrix() * MTK::hat(r_bp);
	const Eigen::Vector3d down_axis_world = pressure_down_axis_world(s);

	dyn_share.h_x.template block<1, 3>(0, 0) = down_axis_world.transpose();
	dyn_share.h_x.template block<1, 3>(0, 3) = down_axis_world.transpose() * d_Rr_dtheta;
	dyn_share.h_x(0, 30) = 1.0;

	double temp[1] = {pressure_meas_predict(s)};
	vect1 out(temp, 1);
	return out;
}

#endif
