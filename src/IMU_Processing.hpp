#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include "use-ikfom.hpp"
#include "dynamics_bridge.hpp"

/// *************Preconfiguration

#define MAX_INI_COUNT (10)

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

/// *************IMU Process and undistortion
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();
  
  void Reset();
  // void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void Reset(double start_timestamp, const sensor_msgs::msg::Imu::ConstSharedPtr &lastimu);
  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4,4) &T);
  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  void set_dynamics_trust(double trust);
  void set_thruster_meas(bool enable, const V3D &cov);
  void set_dvl_params(double cov_floor_std, double min_speed);
  void set_dvl_hold(bool enable, double max_age_sec);
  void set_process_noise(const V3D &nv, const V3D &nw, const V3D &nbg, const V3D &nba, const V3D &nb_dvl, double nb_pressure);
  Eigen::Matrix<double, process_noise_ikfom::DOF, process_noise_ikfom::DOF> Q;
  void Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_);

  ofstream fout_imu;
  V3D cov_acc;
  V3D cov_gyr;
  V3D cov_acc_scale;
  V3D cov_gyr_scale;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;
  V3D cov_proc_nv;
  V3D cov_proc_nw;
  V3D cov_proc_nbg;
  V3D cov_proc_nba;
  V3D cov_proc_nb_dvl;
  double cov_proc_nb_pressure;
  double dynamics_model_trust_ = 1.0;
  bool thruster_meas_en_ = false;
  V3D thruster_acc_cov_ = V3D(0.1, 0.1, 0.1);
  double dvl_cov_floor_std_ = 0.01;
  double dvl_min_speed_ = 0.0;
  bool dvl_hold_enabled_ = false;
  double dvl_hold_max_age_sec_ = 0.0;
  double first_lidar_time;

 private:
  void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom> &kf_state, int &N);
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out);

  PointCloudXYZI::Ptr cur_pcl_un_;
  // sensor_msgs::ImuConstPtr last_imu_;
  sensor_msgs::msg::Imu::ConstSharedPtr last_imu_;
  deque<sensor_msgs::msg::Imu::ConstSharedPtr> v_imu_;
  vector<Pose6D> IMUpose;
  vector<M3D>    v_rot_pcl_;
  M3D Lidar_R_wrt_IMU;
  V3D Lidar_T_wrt_IMU;
  V3D mean_acc;
  V3D mean_gyr;
  V3D angvel_last;
  V3D acc_s_last;
  double start_timestamp_;
  double last_lidar_end_time_;
  int    init_iter_num = 1;
  bool   b_first_frame_ = true;
  bool   imu_need_init_ = true;
  struct DvlHoldSample
  {
    bool valid = false;
    double stamp = 0.0;
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
  } last_dvl_hold_sample_;
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
  init_iter_num = 1;
  Q = process_noise_cov();
  cov_acc       = V3D(0.1, 0.1, 0.1);
  cov_gyr       = V3D(0.1, 0.1, 0.1);
  cov_bias_gyr  = V3D(0.0001, 0.0001, 0.0001);
  cov_bias_acc  = V3D(0.0001, 0.0001, 0.0001);
  cov_proc_nv   = V3D(0.0001, 0.0001, 0.0001);
  cov_proc_nw   = V3D(0.0001, 0.0001, 0.0001);
  cov_proc_nbg  = V3D(0.00001, 0.00001, 0.00001);
  cov_proc_nba  = V3D(0.00001, 0.00001, 0.00001);
  cov_proc_nb_dvl = V3D(0.00001, 0.00001, 0.00001);
  cov_proc_nb_pressure = 0.00001;
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last     = Zero3d;
  Lidar_T_wrt_IMU = Zero3d;
  Lidar_R_wrt_IMU = Eye3d;
  last_imu_.reset(new sensor_msgs::msg::Imu());
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() 
{
  // ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last       = Zero3d;
  imu_need_init_    = true;
  start_timestamp_  = -1;
  init_iter_num     = 1;
  v_imu_.clear();
  IMUpose.clear();
  last_imu_.reset(new sensor_msgs::msg::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
  last_dvl_hold_sample_.valid = false;
  last_dvl_hold_sample_.stamp = 0.0;
  last_dvl_hold_sample_.vel.setZero();
}

void ImuProcess::set_extrinsic(const MD(4,4) &T)
{
  Lidar_T_wrt_IMU = T.block<3,1>(0,3);
  Lidar_R_wrt_IMU = T.block<3,3>(0,0);
}

void ImuProcess::set_extrinsic(const V3D &transl)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}

void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}

void ImuProcess::set_process_noise(const V3D &nv, const V3D &nw, const V3D &nbg, const V3D &nba, const V3D &nb_dvl, double nb_pressure)
{
  cov_proc_nv = nv;
  cov_proc_nw = nw;
  cov_proc_nbg = nbg;
  cov_proc_nba = nba;
  cov_proc_nb_dvl = nb_dvl;
  cov_proc_nb_pressure = std::max(nb_pressure, 0.0);
}

void ImuProcess::set_dynamics_trust(double trust)
{
  dynamics_model_trust_ = std::max(0.0, std::min(1.0, trust));
}

void ImuProcess::set_thruster_meas(bool enable, const V3D &cov)
{
  thruster_meas_en_ = enable;
  thruster_acc_cov_ = cov;
  set_thruster_accel_noise_diag(Eigen::Vector3d(cov(0), cov(1), cov(2)));
}

void ImuProcess::set_dvl_params(double cov_floor_std, double min_speed)
{
  dvl_cov_floor_std_ = std::max(1e-6, cov_floor_std);
  dvl_min_speed_ = std::max(0.0, min_speed);
}

void ImuProcess::set_dvl_hold(bool enable, double max_age_sec)
{
  dvl_hold_enabled_ = enable;
  dvl_hold_max_age_sec_ = std::max(0.0, max_age_sec);
}

void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom> &kf_state, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_)
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    first_lidar_time = meas.lidar_beg_time;
  }

  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;

    N ++;
  }
  state_ikfom init_state = kf_state.get_x();
  init_state.grav = (- mean_acc / mean_acc.norm() * G_m_s2);
  
  //state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  init_state.bg  = mean_gyr;
  init_state.omega = Zero3d;
  init_state.b_dvl = Zero3d;
  init_state.b_pressure[0] = 0.0;
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;
  init_state.offset_R_L_I = Lidar_R_wrt_IMU;
  kf_state.change_x(init_state);

  esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom>::cov init_P = kf_state.get_P();
  init_P.setIdentity();
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;
  init_P(21,21) = init_P(22,22) = init_P(23,23) = 0.00001;
  init_P(27,27) = init_P(28,28) = init_P(29,29) = 0.001;
  init_P(30,30) = 4.0;
  kf_state.change_P(init_P);
  last_imu_ = meas.imu.back();

}

void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom> &kf_state, PointCloudXYZI &pcl_out)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = rclcpp::Time(v_imu.front()->header.stamp).seconds();
  const double &imu_end_time = rclcpp::Time(v_imu.back()->header.stamp).seconds();
  const double &pcl_beg_time = meas.lidar_beg_time;
  const double &pcl_end_time = meas.lidar_end_time;
  
  /*** sort point clouds by offset time ***/
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

  /*** prepare thruster forces in this window ***/
  struct ThrusterSample
  {
    double stamp = 0.0;
    Eigen::VectorXd forces;
  };
  std::vector<ThrusterSample> thr_samples;
  thr_samples.reserve(meas.thruster_forces.size());
  for (const auto &thr_msg : meas.thruster_forces)
  {
    ThrusterSample sample;
    sample.stamp = rclcpp::Time(thr_msg->header.stamp).seconds();
    const auto &effort = thr_msg->effort;
    sample.forces.resize(static_cast<long>(effort.size()));
    for (size_t i = 0; i < effort.size(); ++i)
    {
      sample.forces(static_cast<long>(i)) = effort[i];
    }
    thr_samples.push_back(std::move(sample));
  }
  size_t thr_idx = 0;
  Eigen::VectorXd thr_current;
  const bool dynamics_available = fastlio::dynamics::has_model();
  struct DvlSample
  {
    double stamp = 0.0;
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
  };
  std::vector<DvlSample> dvl_samples;
  dvl_samples.reserve(meas.dvl.size());
  for (const auto &dvl_msg : meas.dvl)
  {
    const auto &lin = dvl_msg->twist.twist.linear;
    if (!std::isfinite(lin.x) || !std::isfinite(lin.y) || !std::isfinite(lin.z))
    {
      continue;
    }

    DvlSample sample;
    sample.stamp = rclcpp::Time(dvl_msg->header.stamp).seconds();
    sample.vel = Eigen::Vector3d(lin.x, lin.y, lin.z);
    dvl_samples.push_back(std::move(sample));
  }
  size_t dvl_idx = 0;
  while (dvl_idx < dvl_samples.size() && dvl_samples[dvl_idx].stamp <= last_lidar_end_time_)
  {
    ++dvl_idx;
  }

  /*** Initialize IMU pose ***/
  state_ikfom imu_state = kf_state.get_x();
  IMUpose.clear();
  V3D vel_world_init = imu_state.rot * imu_state.vel;
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, vel_world_init, imu_state.pos, imu_state.rot.toRotationMatrix()));

  /*** forward propagation at each imu point ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;

  double dt = 0;
  int dvl_updates_applied_this_scan = 0;

  input_ikfom in;
  auto apply_dvl_update = [&](const DvlSample &sample)
  {
    if (dvl_min_speed_ > 0.0 && sample.vel.norm() < dvl_min_speed_)
    {
      return;
    }

    const double floor_var = dvl_cov_floor_std_ * dvl_cov_floor_std_;
    Eigen::Matrix3d R_dvl = Eigen::Matrix3d::Zero();
    R_dvl.diagonal() << floor_var, floor_var, floor_var;
    set_dvl_cov(R_dvl);

    double z_arr[3] = {sample.vel(0), sample.vel(1), sample.vel(2)};
    vect3 z_dvl(z_arr, 3);
    kf_state.update_iterated_dyn_runtime_share(z_dvl, h_dvl_share);
    ++dvl_updates_applied_this_scan;
  };

  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);

    double tail_stamp = rclcpp::Time(tail->header.stamp).seconds();
    double head_stamp = rclcpp::Time(head->header.stamp).seconds();

    if (tail_stamp < last_lidar_end_time_)    continue;
    
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    // fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;

    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm(); // - state_inout.ba;

    const double seg_start = std::max(head_stamp, last_lidar_end_time_);
    if (tail_stamp <= seg_start)
    {
      continue;
    }

    V3D specific_force_dyn = acc_avr;
    bool dyn_acc_valid = false;
    double current_stamp = seg_start;
    while (dvl_idx < dvl_samples.size() && dvl_samples[dvl_idx].stamp <= tail_stamp)
    {
      const double target_stamp = dvl_samples[dvl_idx].stamp;
      if (target_stamp > current_stamp)
      {
        imu_state = kf_state.get_x();
        while (thr_idx < thr_samples.size() && thr_samples[thr_idx].stamp <= current_stamp)
        {
          thr_current = thr_samples[thr_idx].forces;
          ++thr_idx;
        }
        if (dynamics_available && thr_current.size() > 0)
        {
          fastlio::dynamics::set_thruster_forces(thr_current);
        }
        V3D omega_body = imu_state.omega;
        V3D vel_body = imu_state.vel;
        dyn_acc_valid = false;
        specific_force_dyn = acc_avr;
        if (dynamics_available)
        {
          Eigen::Matrix<double, 6, 1> vel_body6;
          vel_body6 << vel_body(0), vel_body(1), vel_body(2), omega_body(0), omega_body(1), omega_body(2);

          vect3 euler_deg = SO3ToEuler(imu_state.rot);
          Eigen::Vector3d euler_rad = Eigen::Vector3d(euler_deg[0], euler_deg[1], euler_deg[2]) * (PI_M / 180.0);
          Eigen::Matrix<double, 6, 1> pose_world6;
          pose_world6 << imu_state.pos(0), imu_state.pos(1), imu_state.pos(2), euler_rad(0), euler_rad(1), euler_rad(2);

          Eigen::Matrix<double, 6, 1> accel_body6;
          if (fastlio::dynamics::compute_body_accel(vel_body6, pose_world6, accel_body6))
          {
            V3D grav_world(imu_state.grav[0], imu_state.grav[1], imu_state.grav[2]);
            V3D grav_body = imu_state.rot.conjugate() * grav_world;
            V3D acc_body_lin(accel_body6(0), accel_body6(1), accel_body6(2));
            specific_force_dyn = acc_body_lin + omega_body.cross(vel_body) - grav_body;
            in.acc = specific_force_dyn + imu_state.ba;
            dyn_acc_valid = true;
          }
        }
        if (!dyn_acc_valid)
        {
          in.acc = acc_avr;
        }
        in.gyro = angvel_avr;
        Q.block<3, 3>(0, 0).diagonal() = cov_proc_nv;
        Q.block<3, 3>(3, 3).diagonal() = cov_proc_nw;
        Q.block<3, 3>(6, 6).diagonal() = cov_proc_nbg;
        Q.block<3, 3>(9, 9).diagonal() = cov_proc_nba;
        Q.block<3, 3>(12, 12).diagonal() = cov_proc_nb_dvl;
        Q(15, 15) = cov_proc_nb_pressure;
        dt = target_stamp - current_stamp;
        kf_state.predict(dt, Q, in);
        current_stamp = target_stamp;
      }
      apply_dvl_update(dvl_samples[dvl_idx]);
      last_dvl_hold_sample_.valid = true;
      last_dvl_hold_sample_.stamp = dvl_samples[dvl_idx].stamp;
      last_dvl_hold_sample_.vel = dvl_samples[dvl_idx].vel;
      ++dvl_idx;
    }

    dt = tail_stamp - current_stamp;
    if (dt > 0.0)
    {
      imu_state = kf_state.get_x();
      while (thr_idx < thr_samples.size() && thr_samples[thr_idx].stamp <= current_stamp)
      {
        thr_current = thr_samples[thr_idx].forces;
        ++thr_idx;
      }
      if (dynamics_available && thr_current.size() > 0)
      {
        fastlio::dynamics::set_thruster_forces(thr_current);
      }
      V3D omega_body = imu_state.omega;
      V3D vel_body = imu_state.vel;

      dyn_acc_valid = false;
      specific_force_dyn = acc_avr;
      if (dynamics_available)
      {
        Eigen::Matrix<double, 6, 1> vel_body6;
        vel_body6 << vel_body(0), vel_body(1), vel_body(2), omega_body(0), omega_body(1), omega_body(2);

        vect3 euler_deg = SO3ToEuler(imu_state.rot);
        Eigen::Vector3d euler_rad = Eigen::Vector3d(euler_deg[0], euler_deg[1], euler_deg[2]) * (PI_M / 180.0);
        Eigen::Matrix<double, 6, 1> pose_world6;
        pose_world6 << imu_state.pos(0), imu_state.pos(1), imu_state.pos(2), euler_rad(0), euler_rad(1), euler_rad(2);

        Eigen::Matrix<double, 6, 1> accel_body6;
        if (fastlio::dynamics::compute_body_accel(vel_body6, pose_world6, accel_body6))
        {
          V3D grav_world(imu_state.grav[0], imu_state.grav[1], imu_state.grav[2]);
          V3D grav_body = imu_state.rot.conjugate() * grav_world;
          V3D acc_body_lin(accel_body6(0), accel_body6(1), accel_body6(2));
          specific_force_dyn = acc_body_lin + omega_body.cross(vel_body) - grav_body;
          in.acc = specific_force_dyn + imu_state.ba;
          dyn_acc_valid = true;
        }
      }
      if (!dyn_acc_valid)
      {
        in.acc = acc_avr;
      }
      in.gyro = angvel_avr;
      Q.block<3, 3>(0, 0).diagonal() = cov_proc_nv;
      Q.block<3, 3>(3, 3).diagonal() = cov_proc_nw;
      Q.block<3, 3>(6, 6).diagonal() = cov_proc_nbg;
      Q.block<3, 3>(9, 9).diagonal() = cov_proc_nba;
      Q.block<3, 3>(12, 12).diagonal() = cov_proc_nb_dvl;
      Q(15, 15) = cov_proc_nb_pressure;
      kf_state.predict(dt, Q, in);
    }

    double lambda_dyn = fastlio::dynamics::has_model() ? std::max(0.01, dynamics_model_trust_) : 1.0;
    set_imu_accel_noise_diag(Eigen::Vector3d(cov_acc(0), cov_acc(1), cov_acc(2)) * lambda_dyn);
    set_imu_gyro_noise_diag(Eigen::Vector3d(cov_gyr(0), cov_gyr(1), cov_gyr(2)));
    double z_arr[3] = {acc_avr(0), acc_avr(1), acc_avr(2)};
    vect3 z_acc(z_arr, 3);
    kf_state.update_iterated_dyn_runtime_share(z_acc, h_imu_accel_share);
    double z_gyr_arr[3] = {angvel_avr(0), angvel_avr(1), angvel_avr(2)};
    vect3 z_gyr(z_gyr_arr, 3);
    kf_state.update_iterated_dyn_runtime_share(z_gyr, h_imu_gyro_share);
    if (thruster_meas_en_ && dyn_acc_valid)
    {
      double z_thr_arr[3] = {specific_force_dyn(0), specific_force_dyn(1), specific_force_dyn(2)};
      vect3 z_thr(z_thr_arr, 3);
      kf_state.update_iterated_dyn_runtime_share(z_thr, h_thruster_accel_share);
    }

    /* save the poses at each IMU measurements */
    imu_state = kf_state.get_x();
    angvel_last = imu_state.omega;
    acc_s_last  = imu_state.rot * (in.acc - imu_state.ba);
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];
    }
    double &&offs_t = tail_stamp - pcl_beg_time;
    V3D vel_world_tail = imu_state.rot * imu_state.vel;
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, vel_world_tail, imu_state.pos, imu_state.rot.toRotationMatrix()));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  double current_tail_time = imu_end_time;
  while (dvl_idx < dvl_samples.size() && dvl_samples[dvl_idx].stamp <= pcl_end_time)
  {
    const double target_stamp = dvl_samples[dvl_idx].stamp;
    if (target_stamp > current_tail_time)
    {
      dt = target_stamp - current_tail_time;
      kf_state.predict(dt, Q, in);
      current_tail_time = target_stamp;
    }
    apply_dvl_update(dvl_samples[dvl_idx]);
    last_dvl_hold_sample_.valid = true;
    last_dvl_hold_sample_.stamp = dvl_samples[dvl_idx].stamp;
    last_dvl_hold_sample_.vel = dvl_samples[dvl_idx].vel;
    ++dvl_idx;
  }
  if (dvl_hold_enabled_ &&
      dvl_updates_applied_this_scan == 0 &&
      last_dvl_hold_sample_.valid)
  {
    const double hold_age_sec = pcl_end_time - last_dvl_hold_sample_.stamp;
    if (hold_age_sec >= 0.0 && hold_age_sec <= dvl_hold_max_age_sec_)
    {
      dt = pcl_end_time - current_tail_time;
      if (std::abs(dt) > 0.0)
      {
        kf_state.predict(dt, Q, in);
      }
      current_tail_time = pcl_end_time;

      DvlSample held_sample;
      held_sample.stamp = last_dvl_hold_sample_.stamp;
      held_sample.vel = last_dvl_hold_sample_.vel;
      apply_dvl_update(held_sample);
    }
  }
  dt = note * (pcl_end_time - current_tail_time);
  if (std::abs(dt) > 0.0)
  {
    kf_state.predict(dt, Q, in);
  }
  
  imu_state = kf_state.get_x();
  last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;

  /*** undistort each lidar point (backward propagation) ***/
  if (pcl_out.points.begin() == pcl_out.points.end()) return;
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu<<MAT_FROM_ARRAY(head->rot);
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
    vel_imu<<VEC_FROM_ARRAY(head->vel);
    pos_imu<<VEC_FROM_ARRAY(head->pos);
    acc_imu<<VEC_FROM_ARRAY(tail->acc);
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr);

    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      M3D R_i(R_imu * Exp(angvel_avr, dt));
      
      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
      V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);// not accurate!
      
      // save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}

void ImuProcess::Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1,t2,t3;
  t1 = omp_get_wtime();

  if(meas.imu.empty()) {return;};
  assert(meas.lidar != nullptr);

  if (imu_need_init_)
  {
    /// The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);

    imu_need_init_ = true;
    
    last_imu_   = meas.imu.back();

    state_ikfom imu_state = kf_state.get_x();
    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;

      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      // ROS_INFO("IMU Initial Done: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
      //          imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
    }

    return;
  }

  UndistortPcl(meas, kf_state, *cur_pcl_un_);

  t2 = omp_get_wtime();
  t3 = omp_get_wtime();
  
  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}
