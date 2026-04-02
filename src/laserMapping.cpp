// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <cmath>
#include <thread>
#include <fstream>
#include <csignal>
#include <chrono>
#include <atomic>
#include <sstream>
#include <iomanip>
#include <unistd.h>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <array>
#include <unordered_map>
#include <Python.h>
#include <so3_math.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <libconfig.h++>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "IMU_Processing.hpp"
#include "dynamics_bridge.hpp"
#include "underwater_vehicle_model.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <fast_lio/msg/frontend_frame.hpp>
#include <fast_lio/msg/optimized_keyframes.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/fluid_pressure.hpp>
#include <std_msgs/msg/u_int32.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#ifdef FASTLIO_HAS_LIVOX
#include <livox_ros_driver2/msg/custom_msg.hpp>
#endif
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
#ifdef FASTLIO_USE_CUDA
#include "gpu/gpu_voxel_map.hpp"
#include "gpu/point_transform_gpu.hpp"
#endif

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double stage_downsample_time = 0.0, stage_transform_time = 0.0, stage_knn_time = 0.0, stage_map_add_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;
std::atomic<bool> process_wakeup_requested{false};

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;
string dvl_topic;
string pressure_topic;
string map_frame_id;
string odom_frame_id = "camera_init";
string body_frame_id = "body";
string frontend_frame_topic = "/fastlio/frontend_frame";
string loop_closure_keyframe_id_topic = "/fastlio/keyframe_id";
string loop_closure_optimized_keyframes_topic = "/fastlio/optimized_keyframes";
string loop_closure_map_to_odom_topic = "/fastlio/map_to_odom";

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double last_timestamp_thruster = -1.0;
double last_timestamp_dvl = -1.0;
double last_timestamp_pressure = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
double last_sync_lidar_end_time = -std::numeric_limits<double>::infinity();
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
double publish_blind_max = 0.0;
bool    is_first_lidar = true;
bool    dvl_enabled = false;
double  dvl_cov_floor_std = 0.01;
double  dvl_min_speed = 0.0;
bool    dvl_hold_enabled = false;
double  dvl_hold_max_age_sec = 0.25;
bool    pressure_enabled = false;
double  pressure_cov_floor_std = 4.09;
double  pressure_reference_pa = 101325.0;
bool    pressure_ref_initialized = false;
bool    loop_closure_enabled = false;
bool    loop_closure_frontend_feedback_enable = false;
int     loop_closure_frontend_frame_stride = 1;
size_t  loop_closure_frontend_frame_cache_size = 256;
double  loop_closure_feedback_min_translation = 0.10;
double  loop_closure_feedback_min_yaw_deg = 0.5;
double  loop_closure_feedback_pose_position_cov = 1.0e-4;
double  loop_closure_feedback_pose_rotation_cov = 1.0e-4;
double  loop_closure_feedback_cross_cov_scale = 0.25;
int     loop_closure_feedback_rebuild_search_num = 10;
double  pressure_ref_sum = 0.0;
int     pressure_ref_count = 0;
constexpr int kPressureRefInitSamples = 30;
constexpr double kPressureDensityFreshWater = 997.0;
uint32_t frontend_scan_id = 0;
std::mutex loop_closure_tf_mutex;
geometry_msgs::msg::TransformStamped loop_closure_map_to_odom_tf;
bool loop_closure_map_to_odom_available = false;
std::mutex loop_closure_feedback_mutex;
uint64_t loop_closure_feedback_optimized_version = 0;
uint64_t loop_closure_feedback_applied_version = 0;
uint32_t loop_closure_feedback_last_applied_scan_id = 0;

struct LoopClosurePose6D
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
};

struct LoopClosureFrontendFrameCache
{
    uint32_t scan_id = 0;
    double stamp_sec = 0.0;
    LoopClosurePose6D raw_pose;
    PointCloudXYZI::Ptr cloud_body;
};

struct LoopClosureOptimizedKeyframesCache
{
    uint64_t version = 0;
    std::vector<uint32_t> scan_ids;
    std::vector<LoopClosurePose6D> poses_in_map;
};

deque<LoopClosureFrontendFrameCache> loop_closure_recent_frontend_frames;
std::vector<LoopClosureFrontendFrameCache> loop_closure_keyframe_cache;
std::unordered_map<uint32_t, size_t> loop_closure_keyframe_index_by_scan_id;
LoopClosureOptimizedKeyframesCache loop_closure_optimized_keyframes_cache;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
vector<double>       dvl_extrinT(3, 0.0);
vector<double>       dvl_extrinR(9, 0.0);
vector<double>       pressure_extrinT(3, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;
deque<sensor_msgs::msg::JointState::ConstSharedPtr> thruster_buffer;
deque<geometry_msgs::msg::TwistWithCovarianceStamped::ConstSharedPtr> dvl_buffer;
deque<sensor_msgs::msg::FluidPressure::ConstSharedPtr> pressure_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;
#ifdef FASTLIO_USE_CUDA
std::unique_ptr<fastlio::gpu::VoxelDownsampler> gpu_downsampler_surf;
std::unique_ptr<fastlio::gpu::PointTransformer> gpu_point_transformer;
#endif

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::Quaternion geoQuat;
geometry_msgs::msg::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

void SigHandle(int sig)
{
    flg_exit = true;
    std::cout << "catch sig %d" << sig << std::endl;
    sig_buffer.notify_all();
    rclcpp::shutdown();
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

LoopClosurePose6D loop_closure_pose_from_state(const state_ikfom &state)
{
    LoopClosurePose6D pose;
    pose.x = state.pos(0);
    pose.y = state.pos(1);
    pose.z = state.pos(2);
    const V3D euler = SO3ToEuler(state.rot);
    constexpr double kDegToRad = M_PI / 180.0;
    pose.roll = euler(0) * kDegToRad;
    pose.pitch = euler(1) * kDegToRad;
    pose.yaw = euler(2) * kDegToRad;
    return pose;
}

LoopClosurePose6D loop_closure_pose_from_msg(const geometry_msgs::msg::Pose &pose_msg)
{
    LoopClosurePose6D pose;
    pose.x = pose_msg.position.x;
    pose.y = pose_msg.position.y;
    pose.z = pose_msg.position.z;
    Eigen::Quaterniond quat(
        pose_msg.orientation.w,
        pose_msg.orientation.x,
        pose_msg.orientation.y,
        pose_msg.orientation.z);
    const Eigen::Vector3d ypr = quat.toRotationMatrix().eulerAngles(2, 1, 0);
    pose.yaw = ypr[0];
    pose.pitch = ypr[1];
    pose.roll = ypr[2];
    return pose;
}

Eigen::Affine3d loop_closure_pose_to_affine(const LoopClosurePose6D &pose)
{
    Eigen::Affine3d transform = Eigen::Affine3d::Identity();
    transform.translation() << pose.x, pose.y, pose.z;
    transform.linear() =
        (Eigen::AngleAxisd(pose.yaw, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(pose.pitch, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(pose.roll, Eigen::Vector3d::UnitX())).toRotationMatrix();
    return transform;
}

LoopClosurePose6D loop_closure_pose_from_affine(const Eigen::Affine3d &transform)
{
    LoopClosurePose6D pose;
    pose.x = transform.translation().x();
    pose.y = transform.translation().y();
    pose.z = transform.translation().z();
    const Eigen::Vector3d ypr = transform.rotation().eulerAngles(2, 1, 0);
    pose.yaw = ypr[0];
    pose.pitch = ypr[1];
    pose.roll = ypr[2];
    return pose;
}

double loop_closure_normalize_angle(double angle)
{
    while (angle > M_PI) {
        angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }
    return angle;
}

void cache_loop_closure_frontend_frame(
    uint32_t scan_id,
    double stamp_sec,
    const LoopClosurePose6D &raw_pose,
    const PointCloudXYZI::Ptr &cloud_body)
{
    if (!loop_closure_frontend_feedback_enable || !cloud_body) {
        return;
    }

    LoopClosureFrontendFrameCache frame;
    frame.scan_id = scan_id;
    frame.stamp_sec = stamp_sec;
    frame.raw_pose = raw_pose;
    frame.cloud_body.reset(new PointCloudXYZI(*cloud_body));

    std::lock_guard<std::mutex> lock(loop_closure_feedback_mutex);
    loop_closure_recent_frontend_frames.push_back(frame);
    while (loop_closure_recent_frontend_frames.size() > loop_closure_frontend_frame_cache_size) {
        loop_closure_recent_frontend_frames.pop_front();
    }
}

void loop_closure_keyframe_id_cbk(const std_msgs::msg::UInt32::SharedPtr msg)
{
    if (!loop_closure_frontend_feedback_enable) {
        return;
    }

    std::lock_guard<std::mutex> lock(loop_closure_feedback_mutex);
    if (loop_closure_keyframe_index_by_scan_id.count(msg->data) > 0) {
        return;
    }

    auto recent_it = std::find_if(
        loop_closure_recent_frontend_frames.rbegin(),
        loop_closure_recent_frontend_frames.rend(),
        [&](const LoopClosureFrontendFrameCache &frame) { return frame.scan_id == msg->data; });
    if (recent_it == loop_closure_recent_frontend_frames.rend()) {
        return;
    }

    loop_closure_keyframe_cache.push_back(*recent_it);
    loop_closure_keyframe_index_by_scan_id[msg->data] = loop_closure_keyframe_cache.size() - 1;
}

void loop_closure_optimized_keyframes_cbk(const fast_lio::msg::OptimizedKeyframes::SharedPtr msg)
{
    if (!loop_closure_frontend_feedback_enable) {
        return;
    }
    if (msg->scan_ids.size() != msg->poses_in_map.size()) {
        return;
    }

    LoopClosureOptimizedKeyframesCache cache;
    cache.scan_ids = msg->scan_ids;
    cache.poses_in_map.reserve(msg->poses_in_map.size());
    for (const auto &pose_msg : msg->poses_in_map) {
        cache.poses_in_map.push_back(loop_closure_pose_from_msg(pose_msg));
    }

    std::lock_guard<std::mutex> lock(loop_closure_feedback_mutex);
    cache.version = loop_closure_feedback_optimized_version + 1;
    loop_closure_feedback_optimized_version = cache.version;
    loop_closure_optimized_keyframes_cache = std::move(cache);
}

void reset_loop_closure_map_to_odom_identity()
{
    std::lock_guard<std::mutex> lock(loop_closure_tf_mutex);
    loop_closure_map_to_odom_tf.header.frame_id = map_frame_id;
    loop_closure_map_to_odom_tf.child_frame_id = odom_frame_id;
    loop_closure_map_to_odom_tf.transform.translation.x = 0.0;
    loop_closure_map_to_odom_tf.transform.translation.y = 0.0;
    loop_closure_map_to_odom_tf.transform.translation.z = 0.0;
    loop_closure_map_to_odom_tf.transform.rotation.x = 0.0;
    loop_closure_map_to_odom_tf.transform.rotation.y = 0.0;
    loop_closure_map_to_odom_tf.transform.rotation.z = 0.0;
    loop_closure_map_to_odom_tf.transform.rotation.w = 1.0;
    loop_closure_map_to_odom_available = true;
}

void apply_loop_closure_pose_covariance()
{
    auto P_updated = kf.get_P();
    for (int row = 0; row < 6; ++row) {
        for (int col = 6; col < P_updated.cols(); ++col) {
            P_updated(row, col) *= loop_closure_feedback_cross_cov_scale;
            P_updated(col, row) = P_updated(row, col);
        }
    }

    for (int idx = 0; idx < 3; ++idx) {
        P_updated(idx, idx) = std::min(P_updated(idx, idx), loop_closure_feedback_pose_position_cov);
    }
    for (int idx = 3; idx < 6; ++idx) {
        P_updated(idx, idx) = std::min(P_updated(idx, idx), loop_closure_feedback_pose_rotation_cov);
    }

    kf.change_P(P_updated);
}

void maybe_apply_loop_closure_frontend_feedback(rclcpp::Logger logger)
{
    if (!loop_closure_enabled || !loop_closure_frontend_feedback_enable) {
        return;
    }

    struct PendingLoopClosureFeedback
    {
        uint64_t version = 0;
        uint32_t scan_id = 0;
        size_t keyframe_index = 0;
        LoopClosurePose6D raw_keyframe_pose;
        LoopClosurePose6D optimized_keyframe_pose;
        std::vector<std::pair<LoopClosurePose6D, PointCloudXYZI::Ptr>> rebuild_frames;
    } pending;

    {
        std::lock_guard<std::mutex> lock(loop_closure_feedback_mutex);
        if (loop_closure_optimized_keyframes_cache.version == 0 ||
            loop_closure_optimized_keyframes_cache.version == loop_closure_feedback_applied_version ||
            loop_closure_keyframe_cache.empty())
        {
            return;
        }

        std::unordered_map<uint32_t, size_t> optimized_index_by_scan_id;
        optimized_index_by_scan_id.reserve(loop_closure_optimized_keyframes_cache.scan_ids.size());
        for (size_t idx = 0; idx < loop_closure_optimized_keyframes_cache.scan_ids.size(); ++idx) {
            optimized_index_by_scan_id[loop_closure_optimized_keyframes_cache.scan_ids[idx]] = idx;
        }

        for (int idx = static_cast<int>(loop_closure_optimized_keyframes_cache.scan_ids.size()) - 1; idx >= 0; --idx) {
            const uint32_t scan_id = loop_closure_optimized_keyframes_cache.scan_ids[static_cast<size_t>(idx)];
            const auto keyframe_it = loop_closure_keyframe_index_by_scan_id.find(scan_id);
            if (keyframe_it == loop_closure_keyframe_index_by_scan_id.end()) {
                continue;
            }

            pending.version = loop_closure_optimized_keyframes_cache.version;
            pending.scan_id = scan_id;
            pending.keyframe_index = keyframe_it->second;
            pending.raw_keyframe_pose = loop_closure_keyframe_cache[pending.keyframe_index].raw_pose;
            pending.optimized_keyframe_pose = loop_closure_optimized_keyframes_cache.poses_in_map[static_cast<size_t>(idx)];

            const int start_idx = std::max<int>(0, static_cast<int>(pending.keyframe_index) - loop_closure_feedback_rebuild_search_num);
            const int end_idx = std::min<int>(
                static_cast<int>(loop_closure_keyframe_cache.size()) - 1,
                static_cast<int>(pending.keyframe_index) + loop_closure_feedback_rebuild_search_num);

            for (int key_idx = start_idx; key_idx <= end_idx; ++key_idx) {
                const auto &keyframe = loop_closure_keyframe_cache[static_cast<size_t>(key_idx)];
                const auto opt_it = optimized_index_by_scan_id.find(keyframe.scan_id);
                if (opt_it == optimized_index_by_scan_id.end() || !keyframe.cloud_body || keyframe.cloud_body->empty()) {
                    continue;
                }
                pending.rebuild_frames.emplace_back(
                    loop_closure_optimized_keyframes_cache.poses_in_map[opt_it->second],
                    keyframe.cloud_body);
            }
            break;
        }
    }

    if (pending.version == 0 || pending.scan_id == 0) {
        return;
    }

    const Eigen::Affine3d raw_keyframe_affine = loop_closure_pose_to_affine(pending.raw_keyframe_pose);
    const Eigen::Affine3d optimized_keyframe_affine = loop_closure_pose_to_affine(pending.optimized_keyframe_pose);
    const Eigen::Affine3d correction = optimized_keyframe_affine * raw_keyframe_affine.inverse();

    const double correction_translation = correction.translation().norm();
    const Eigen::Vector3d correction_ypr = correction.rotation().eulerAngles(2, 1, 0);
    const double correction_yaw_deg = std::abs(loop_closure_normalize_angle(correction_ypr[0])) * 180.0 / M_PI;

    {
        std::lock_guard<std::mutex> lock(loop_closure_feedback_mutex);
        if (pending.version == loop_closure_feedback_applied_version &&
            pending.scan_id == loop_closure_feedback_last_applied_scan_id)
        {
            return;
        }
    }

    if (correction_translation < loop_closure_feedback_min_translation &&
        correction_yaw_deg < loop_closure_feedback_min_yaw_deg)
    {
        std::lock_guard<std::mutex> lock(loop_closure_feedback_mutex);
        loop_closure_feedback_applied_version = pending.version;
        loop_closure_feedback_last_applied_scan_id = pending.scan_id;
        return;
    }

    state_ikfom state_updated = kf.get_x();
    const Eigen::Affine3d current_pose = loop_closure_pose_to_affine(loop_closure_pose_from_state(state_updated));
    const Eigen::Affine3d corrected_pose = correction * current_pose;
    const Eigen::Quaterniond corrected_quat(corrected_pose.rotation());
    const Eigen::Vector3d corrected_pos = corrected_pose.translation();
    state_updated.rot = corrected_quat;
    state_updated.pos[0] = corrected_pos.x();
    state_updated.pos[1] = corrected_pos.y();
    state_updated.pos[2] = corrected_pos.z();
    kf.change_x(state_updated);
    apply_loop_closure_pose_covariance();

    state_point = kf.get_x();
    euler_cur = SO3ToEuler(state_point.rot);
    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
    geoQuat.x = state_point.rot.coeffs()[0];
    geoQuat.y = state_point.rot.coeffs()[1];
    geoQuat.z = state_point.rot.coeffs()[2];
    geoQuat.w = state_point.rot.coeffs()[3];

    if (!pending.rebuild_frames.empty()) {
        PointCloudXYZI::Ptr rebuilt_submap(new PointCloudXYZI());
        for (const auto &entry : pending.rebuild_frames) {
            PointCloudXYZI transformed_cloud;
            pcl::transformPointCloud(*entry.second, transformed_cloud, loop_closure_pose_to_affine(entry.first).matrix().cast<float>());
            *rebuilt_submap += transformed_cloud;
        }

        if (!rebuilt_submap->empty()) {
            pcl::VoxelGrid<PointType> downsampler;
            downsampler.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
            PointCloudXYZI::Ptr rebuilt_submap_ds(new PointCloudXYZI());
            downsampler.setInputCloud(rebuilt_submap);
            downsampler.filter(*rebuilt_submap_ds);
            if (!rebuilt_submap_ds->empty()) {
                ikdtree.set_downsample_param(filter_size_map_min);
                ikdtree.Build(rebuilt_submap_ds->points);
            }
        }
    }

    reset_loop_closure_map_to_odom_identity();

    {
        std::lock_guard<std::mutex> lock(loop_closure_feedback_mutex);
        loop_closure_feedback_applied_version = pending.version;
        loop_closure_feedback_last_applied_scan_id = pending.scan_id;
    }

}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::UniquePtr msg) 
{
    mtx_buffer.lock();
    scan_count ++;
    double cur_time = get_time_sec(msg->header.stamp);
    double preprocess_start_time = omp_get_wtime();
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
        last_sync_lidar_end_time = -std::numeric_limits<double>::infinity();
    }
    if (is_first_lidar)
    {
        is_first_lidar = false;
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(cur_time);
    last_timestamp_lidar = cur_time;
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    process_wakeup_requested.store(true, std::memory_order_release);
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
#ifdef FASTLIO_HAS_LIVOX
void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::UniquePtr msg) 
{
    mtx_buffer.lock();
    double cur_time = get_time_sec(msg->header.stamp);
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
        last_sync_lidar_end_time = -std::numeric_limits<double>::infinity();
    }
    if(is_first_lidar)
    {
        is_first_lidar = false;
    }
    last_timestamp_lidar = cur_time;
    
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    process_wakeup_requested.store(true, std::memory_order_release);
    sig_buffer.notify_all();
}
#endif

void imu_cbk(const sensor_msgs::msg::Imu::UniquePtr msg_in)
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));
    

    msg->header.stamp = get_ros_time(get_time_sec(msg_in->header.stamp) - time_diff_lidar_to_imu);
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        rclcpp::Time(timediff_lidar_wrt_imu + get_time_sec(msg_in->header.stamp));
    }

    double timestamp = get_time_sec(msg->header.stamp);

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    process_wakeup_requested.store(true, std::memory_order_release);
    sig_buffer.notify_all();
}

void thruster_cbk(const sensor_msgs::msg::JointState::UniquePtr msg_in)
{
    auto msg = sensor_msgs::msg::JointState::SharedPtr(new sensor_msgs::msg::JointState(*msg_in));
    double timestamp = get_time_sec(msg->header.stamp);

    mtx_buffer.lock();

    if (timestamp < last_timestamp_thruster)
    {
        std::cerr << "Thruster forces loop back, clear buffer" << std::endl;
        thruster_buffer.clear();
    }

    last_timestamp_thruster = timestamp;
    thruster_buffer.push_back(msg);

    mtx_buffer.unlock();
    process_wakeup_requested.store(true, std::memory_order_release);
    sig_buffer.notify_all();
}

void dvl_cbk(const geometry_msgs::msg::TwistWithCovarianceStamped::UniquePtr msg_in)
{
    auto msg = geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr(
        new geometry_msgs::msg::TwistWithCovarianceStamped(*msg_in));
    double timestamp = get_time_sec(msg->header.stamp);

    mtx_buffer.lock();

    if (timestamp < last_timestamp_dvl)
    {
        std::cerr << "DVL loop back, clear buffer" << std::endl;
        dvl_buffer.clear();
    }

    last_timestamp_dvl = timestamp;
    dvl_buffer.push_back(msg);

    mtx_buffer.unlock();
    process_wakeup_requested.store(true, std::memory_order_release);
    sig_buffer.notify_all();
}

void pressure_cbk(const sensor_msgs::msg::FluidPressure::UniquePtr msg_in)
{
    auto msg = sensor_msgs::msg::FluidPressure::SharedPtr(new sensor_msgs::msg::FluidPressure(*msg_in));
    double timestamp = get_time_sec(msg->header.stamp);

    mtx_buffer.lock();

    if (timestamp < last_timestamp_pressure)
    {
        std::cerr << "Pressure loop back, clear buffer" << std::endl;
        pressure_buffer.clear();
    }

    last_timestamp_pressure = timestamp;
    pressure_buffer.push_back(msg);

    mtx_buffer.unlock();
    process_wakeup_requested.store(true, std::memory_order_release);
    sig_buffer.notify_all();
}

void loop_closure_map_to_odom_cbk(const geometry_msgs::msg::TransformStamped::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(loop_closure_tf_mutex);
    loop_closure_map_to_odom_tf = *msg;
    loop_closure_map_to_odom_available = true;
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    std::lock_guard<std::mutex> lock(mtx_buffer);

    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            std::cerr << "Too few input point cloud!\n";
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;
        meas.prev_lidar_end_time = last_sync_lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    const double sensor_window_start = std::isfinite(meas.prev_lidar_end_time)
        ? meas.prev_lidar_end_time
        : -std::numeric_limits<double>::infinity();

    /*** push imu data, and pop from imu buffer ***/
    meas.imu.clear();
    while (!imu_buffer.empty())
    {
        const double imu_time = get_time_sec(imu_buffer.front()->header.stamp);
        if (imu_time <= sensor_window_start)
        {
            imu_buffer.pop_front();
            continue;
        }
        if (imu_time > lidar_end_time)
        {
            break;
        }
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    /*** push thruster data, and pop from thruster buffer ***/
    meas.thruster_forces.clear();
    while (!thruster_buffer.empty())
    {
        double thruster_time = get_time_sec(thruster_buffer.front()->header.stamp);
        if (thruster_time <= sensor_window_start)
        {
            thruster_buffer.pop_front();
            continue;
        }
        if (thruster_time > lidar_end_time)
        {
            break;
        }
        meas.thruster_forces.push_back(thruster_buffer.front());
        thruster_buffer.pop_front();
    }

    /*** push DVL data, and pop from DVL buffer ***/
    meas.dvl.clear();
    while (!dvl_buffer.empty())
    {
        double dvl_time = get_time_sec(dvl_buffer.front()->header.stamp);
        if (dvl_time <= sensor_window_start)
        {
            dvl_buffer.pop_front();
            continue;
        }
        if (dvl_time > lidar_end_time)
        {
            break;
        }
        meas.dvl.push_back(dvl_buffer.front());
        dvl_buffer.pop_front();
    }

    /*** push pressure data, and pop from pressure buffer ***/
    meas.pressure.clear();
    while (!pressure_buffer.empty())
    {
        double pressure_time = get_time_sec(pressure_buffer.front()->header.stamp);
        if (pressure_time <= sensor_window_start)
        {
            pressure_buffer.pop_front();
            continue;
        }
        if (pressure_time > lidar_end_time)
        {
            break;
        }
        meas.pressure.push_back(pressure_buffer.front());
        pressure_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    last_sync_lidar_end_time = lidar_end_time;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);

    bool transformed_on_gpu = false;
#ifdef FASTLIO_USE_CUDA
    if (gpu_point_transformer && gpu_point_transformer->available())
    {
        const double transform_start = omp_get_wtime();
        const Eigen::Matrix3d rot_world_body_d = state_point.rot.toRotationMatrix();
        const Eigen::Matrix3d rot_body_lidar_d = state_point.offset_R_L_I.toRotationMatrix();
        const Eigen::Matrix3d rot_world_lidar_d = rot_world_body_d * rot_body_lidar_d;
        const Eigen::Vector3d t_body_lidar_d(state_point.offset_T_L_I[0], state_point.offset_T_L_I[1], state_point.offset_T_L_I[2]);
        const Eigen::Vector3d t_world_lidar_d = rot_world_body_d * t_body_lidar_d + Eigen::Vector3d(state_point.pos[0], state_point.pos[1], state_point.pos[2]);

        transformed_on_gpu = gpu_point_transformer->transform(*feats_down_body,
                                                               *feats_down_world,
                                                               rot_world_lidar_d.cast<float>(),
                                                               t_world_lidar_d.cast<float>());
        stage_transform_time += omp_get_wtime() - transform_start;
    }
#endif

    const double transform_start_cpu = omp_get_wtime();
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        #ifdef FASTLIO_USE_CUDA
        if (!transformed_on_gpu)
        #endif
        {
            pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        }
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }
    if (!transformed_on_gpu)
    {
        stage_transform_time += omp_get_wtime() - transform_start_cpu;
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
    stage_map_add_time += kdtree_incremental_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI());
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
PointCloudXYZI::Ptr publish_world_buffer(new PointCloudXYZI());
PointCloudXYZI::Ptr publish_body_buffer(new PointCloudXYZI());
PointCloudXYZI::Ptr publish_effect_buffer(new PointCloudXYZI());
PointCloudXYZI::Ptr publish_world_filter_buffer(new PointCloudXYZI());
PointCloudXYZI::Ptr publish_body_filter_buffer(new PointCloudXYZI());
void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        PointCloudXYZI::Ptr cloud_for_transform = laserCloudFullRes;
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld = publish_world_buffer;
        laserCloudWorld->clear();
        laserCloudWorld->reserve(size);
        const double blind_max_sq = publish_blind_max > 0.0 ? publish_blind_max * publish_blind_max : 0.0;

        if (publish_blind_max > 0.0)
        {
            PointCloudXYZI::Ptr filtered = publish_world_filter_buffer;
            filtered->clear();
            filtered->reserve(size);
            for (int i = 0; i < size; i++)
            {
                const PointType &src_pt = laserCloudFullRes->points[i];
                const double range_sq = src_pt.x * src_pt.x + src_pt.y * src_pt.y + src_pt.z * src_pt.z;
                if (range_sq < blind_max_sq)
                {
                    filtered->points.push_back(src_pt);
                }
            }
            filtered->width = filtered->points.size();
            filtered->height = 1;
            filtered->is_dense = true;
            cloud_for_transform = filtered;
            size = static_cast<int>(cloud_for_transform->points.size());
        }

        bool transformed_on_gpu = false;
#ifdef FASTLIO_USE_CUDA
        if (gpu_point_transformer && gpu_point_transformer->available() && size > 0)
        {
            const Eigen::Matrix3d rot_world_body_d = state_point.rot.toRotationMatrix();
            const Eigen::Matrix3d rot_body_lidar_d = state_point.offset_R_L_I.toRotationMatrix();
            const Eigen::Matrix3d rot_world_lidar_d = rot_world_body_d * rot_body_lidar_d;
            const Eigen::Vector3d t_body_lidar_d(state_point.offset_T_L_I[0], state_point.offset_T_L_I[1], state_point.offset_T_L_I[2]);
            const Eigen::Vector3d t_world_lidar_d = rot_world_body_d * t_body_lidar_d + Eigen::Vector3d(state_point.pos[0], state_point.pos[1], state_point.pos[2]);
            transformed_on_gpu = gpu_point_transformer->transform(*cloud_for_transform,
                                                                   *laserCloudWorld,
                                                                   rot_world_lidar_d.cast<float>(),
                                                                   t_world_lidar_d.cast<float>());
        }
#endif

        if (!transformed_on_gpu)
        {
            for (int i = 0; i < size; i++)
            {
                const PointType &src_pt = cloud_for_transform->points[i];

                PointType world_pt;
                RGBpointBodyToWorld(&src_pt, &world_pt);
                laserCloudWorld->points.push_back(world_pt);
            }
            laserCloudWorld->width = laserCloudWorld->points.size();
            laserCloudWorld->height = 1;
            laserCloudWorld->is_dense = true;
        }

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        // laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = odom_frame_id;
        pubLaserCloudFull->publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    /*
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
    */
}

void publish_frame_body(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body)
{
    PointCloudXYZI::Ptr cloud_for_transform = feats_undistort;
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody = publish_body_buffer;
    laserCloudIMUBody->clear();
    laserCloudIMUBody->reserve(size);
    const double blind_max_sq = publish_blind_max > 0.0 ? publish_blind_max * publish_blind_max : 0.0;

    if (publish_blind_max > 0.0)
    {
        PointCloudXYZI::Ptr filtered = publish_body_filter_buffer;
        filtered->clear();
        filtered->reserve(size);
        for (int i = 0; i < size; i++)
        {
            const PointType &src_pt = feats_undistort->points[i];
            const double range_sq = src_pt.x * src_pt.x + src_pt.y * src_pt.y + src_pt.z * src_pt.z;
            if (range_sq < blind_max_sq)
            {
                filtered->points.push_back(src_pt);
            }
        }
        filtered->width = filtered->points.size();
        filtered->height = 1;
        filtered->is_dense = true;
        cloud_for_transform = filtered;
        size = static_cast<int>(cloud_for_transform->points.size());
    }

    bool transformed_on_gpu = false;
#ifdef FASTLIO_USE_CUDA
    if (gpu_point_transformer && gpu_point_transformer->available() && size > 0)
    {
        const Eigen::Matrix3d rot_body_lidar_d = state_point.offset_R_L_I.toRotationMatrix();
        const Eigen::Vector3d t_body_lidar_d(state_point.offset_T_L_I[0], state_point.offset_T_L_I[1], state_point.offset_T_L_I[2]);
        transformed_on_gpu = gpu_point_transformer->transform(*cloud_for_transform,
                                                               *laserCloudIMUBody,
                                                               rot_body_lidar_d.cast<float>(),
                                                               t_body_lidar_d.cast<float>());
    }
#endif

    if (!transformed_on_gpu)
    {
        for (int i = 0; i < size; i++)
        {
            const PointType &src_pt = cloud_for_transform->points[i];

            PointType body_pt;
            RGBpointBodyLidarToIMU(&src_pt, &body_pt);
            laserCloudIMUBody->points.push_back(body_pt);
        }
        laserCloudIMUBody->width = laserCloudIMUBody->points.size();
        laserCloudIMUBody->height = 1;
        laserCloudIMUBody->is_dense = true;
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = body_frame_id;
    pubLaserCloudFull_body->publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_frontend_frame(rclcpp::Publisher<fast_lio::msg::FrontendFrame>::SharedPtr pubFrontendFrame)
{
    if (!pubFrontendFrame) {
        return;
    }

    PointCloudXYZI::Ptr cloud_for_transform = feats_undistort;
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody = publish_body_buffer;
    laserCloudIMUBody->clear();
    laserCloudIMUBody->reserve(size);
    const double blind_max_sq = publish_blind_max > 0.0 ? publish_blind_max * publish_blind_max : 0.0;

    if (publish_blind_max > 0.0)
    {
        PointCloudXYZI::Ptr filtered = publish_body_filter_buffer;
        filtered->clear();
        filtered->reserve(size);
        for (int i = 0; i < size; i++)
        {
            const PointType &src_pt = feats_undistort->points[i];
            const double range_sq = src_pt.x * src_pt.x + src_pt.y * src_pt.y + src_pt.z * src_pt.z;
            if (range_sq < blind_max_sq)
            {
                filtered->points.push_back(src_pt);
            }
        }
        filtered->width = filtered->points.size();
        filtered->height = 1;
        filtered->is_dense = true;
        cloud_for_transform = filtered;
        size = static_cast<int>(cloud_for_transform->points.size());
    }

    bool transformed_on_gpu = false;
#ifdef FASTLIO_USE_CUDA
    if (gpu_point_transformer && gpu_point_transformer->available() && size > 0)
    {
        const Eigen::Matrix3d rot_body_lidar_d = state_point.offset_R_L_I.toRotationMatrix();
        const Eigen::Vector3d t_body_lidar_d(state_point.offset_T_L_I[0], state_point.offset_T_L_I[1], state_point.offset_T_L_I[2]);
        transformed_on_gpu = gpu_point_transformer->transform(*cloud_for_transform,
                                                               *laserCloudIMUBody,
                                                               rot_body_lidar_d.cast<float>(),
                                                               t_body_lidar_d.cast<float>());
    }
#endif

    if (!transformed_on_gpu)
    {
        for (int i = 0; i < size; i++)
        {
            const PointType &src_pt = cloud_for_transform->points[i];

            PointType body_pt;
            RGBpointBodyLidarToIMU(&src_pt, &body_pt);
            laserCloudIMUBody->points.push_back(body_pt);
        }
        laserCloudIMUBody->width = laserCloudIMUBody->points.size();
        laserCloudIMUBody->height = 1;
        laserCloudIMUBody->is_dense = true;
    }

    fast_lio::msg::FrontendFrame frontend_frame;
    frontend_frame.header.stamp = get_ros_time(lidar_end_time);
    frontend_frame.header.frame_id = odom_frame_id;
    frontend_frame.scan_id = frontend_scan_id;
    frontend_frame.pose_in_odom.position.x = state_point.pos(0);
    frontend_frame.pose_in_odom.position.y = state_point.pos(1);
    frontend_frame.pose_in_odom.position.z = state_point.pos(2);
    frontend_frame.pose_in_odom.orientation.x = geoQuat.x;
    frontend_frame.pose_in_odom.orientation.y = geoQuat.y;
    frontend_frame.pose_in_odom.orientation.z = geoQuat.z;
    frontend_frame.pose_in_odom.orientation.w = geoQuat.w;
    pcl::toROSMsg(*laserCloudIMUBody, frontend_frame.cloud_body);
    frontend_frame.cloud_body.header.stamp = frontend_frame.header.stamp;
    frontend_frame.cloud_body.header.frame_id = body_frame_id;
    cache_loop_closure_frontend_frame(
        frontend_frame.scan_id,
        lidar_end_time,
        loop_closure_pose_from_state(state_point),
        laserCloudIMUBody);
    pubFrontendFrame->publish(frontend_frame);
}

void publish_effect_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld = publish_effect_buffer;
    laserCloudWorld->clear();
    laserCloudWorld->resize(effct_feat_num);
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    laserCloudWorld->width = laserCloudWorld->points.size();
    laserCloudWorld->height = 1;
    laserCloudWorld->is_dense = true;
    sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = get_ros_time(lidar_end_time);
    laserCloudFullRes3.header.frame_id = odom_frame_id;
    pubLaserCloudEffect->publish(laserCloudFullRes3);
}

void publish_map(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap)
{
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                            &laserCloudWorld->points[i]);
    }
    *pcl_wait_pub += *laserCloudWorld;

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);
    // laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = odom_frame_id;
    pubLaserCloudMap->publish(laserCloudmsg);

    // sensor_msgs::msg::PointCloud2 laserCloudMap;
    // pcl::toROSMsg(*featsFromMap, laserCloudMap);
    // laserCloudMap.header.stamp = get_ros_time(lidar_end_time);
    // laserCloudMap.header.frame_id = odom_frame_id;
    // pubLaserCloudMap->publish(laserCloudMap);
}

void save_to_pcd()
{
    pcl::PCDWriter pcd_writer;
    pcd_writer.writeBinary(map_file_path, *pcl_wait_pub);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped, std::unique_ptr<tf2_ros::TransformBroadcaster> & tf_br)
{
    odomAftMapped.header.frame_id = odom_frame_id;
    odomAftMapped.child_frame_id = body_frame_id;
    odomAftMapped.header.stamp = get_ros_time(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped->publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    geometry_msgs::msg::TransformStamped trans;
    trans.header.frame_id = odom_frame_id;
    trans.header.stamp = odomAftMapped.header.stamp;
    trans.child_frame_id = body_frame_id;
    trans.transform.translation.x = odomAftMapped.pose.pose.position.x;
    trans.transform.translation.y = odomAftMapped.pose.pose.position.y;
    trans.transform.translation.z = odomAftMapped.pose.pose.position.z;
    trans.transform.rotation.w = odomAftMapped.pose.pose.orientation.w;
    trans.transform.rotation.x = odomAftMapped.pose.pose.orientation.x;
    trans.transform.rotation.y = odomAftMapped.pose.pose.orientation.y;
    trans.transform.rotation.z = odomAftMapped.pose.pose.orientation.z;
    tf_br->sendTransform(trans);

    if (loop_closure_enabled)
    {
        geometry_msgs::msg::TransformStamped map_to_odom;
        if (loop_closure_frontend_feedback_enable)
        {
            map_to_odom.header.frame_id = map_frame_id;
            map_to_odom.child_frame_id = odom_frame_id;
            map_to_odom.transform.translation.x = 0.0;
            map_to_odom.transform.translation.y = 0.0;
            map_to_odom.transform.translation.z = 0.0;
            map_to_odom.transform.rotation.x = 0.0;
            map_to_odom.transform.rotation.y = 0.0;
            map_to_odom.transform.rotation.z = 0.0;
            map_to_odom.transform.rotation.w = 1.0;
        }
        else
        {
            std::lock_guard<std::mutex> lock(loop_closure_tf_mutex);
            map_to_odom = loop_closure_map_to_odom_tf;
        }
        map_to_odom.header.stamp = odomAftMapped.header.stamp;
        map_to_odom.header.frame_id = map_frame_id;
        map_to_odom.child_frame_id = odom_frame_id;
        tf_br->sendTransform(map_to_odom);
    }
}

void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = get_ros_time(lidar_end_time); // ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = odom_frame_id;

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath->publish(path);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    bool transformed_on_gpu = false;
#ifdef FASTLIO_USE_CUDA
    if (gpu_point_transformer && gpu_point_transformer->available())
    {
        const double transform_start = omp_get_wtime();
        const Eigen::Matrix3d rot_world_body_d = s.rot.toRotationMatrix();
        const Eigen::Matrix3d rot_body_lidar_d = s.offset_R_L_I.toRotationMatrix();
        const Eigen::Matrix3d rot_world_lidar_d = rot_world_body_d * rot_body_lidar_d;
        const Eigen::Vector3d t_body_lidar_d(s.offset_T_L_I[0], s.offset_T_L_I[1], s.offset_T_L_I[2]);
        const Eigen::Vector3d t_world_lidar_d = rot_world_body_d * t_body_lidar_d + Eigen::Vector3d(s.pos[0], s.pos[1], s.pos[2]);

        transformed_on_gpu = gpu_point_transformer->transform(*feats_down_body,
                                                               *feats_down_world,
                                                               rot_world_lidar_d.cast<float>(),
                                                               t_world_lidar_d.cast<float>());
        stage_transform_time += omp_get_wtime() - transform_start;
    }
#endif

    if (!transformed_on_gpu)
    {
        const double transform_start_cpu = omp_get_wtime();
        #ifdef MP_EN
            #pragma omp parallel for
        #endif
        for (int i = 0; i < feats_down_size; i++)
        {
            PointType &point_body  = feats_down_body->points[i];
            PointType &point_world = feats_down_world->points[i];
            V3D p_body(point_body.x, point_body.y, point_body.z);
            V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
            point_world.x = p_global(0);
            point_world.y = p_global(1);
            point_world.z = p_global(2);
            point_world.intensity = point_body.intensity;
        }
        stage_transform_time += omp_get_wtime() - transform_start_cpu;
    }

    const bool perform_knn_search = ekfom_data.converge;
    double knn_stage_local = 0.0;

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 
        V3D p_body(point_body.x, point_body.y, point_body.z);

        static thread_local vector<float> pointSearchSqDis;
        if (pointSearchSqDis.size() != NUM_MATCH_POINTS)
        {
            pointSearchSqDis.resize(NUM_MATCH_POINTS);
        }

        auto &points_near = Nearest_Points[i];

        if (perform_knn_search)
        {
            /** Find the closest surfaces in the map **/
            const double knn_start = omp_get_wtime();
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            knn_stage_local += omp_get_wtime() - knn_start;
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    stage_knn_time += knn_stage_local;
    kdtree_search_time += knn_stage_local;
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        std::cerr << "No Effective Points!" << std::endl;
        // ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, state_ikfom::DOF);
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        auto H_i = ekfom_data.h_x.row(i);
        H_i(0) = norm_p.x;
        H_i(1) = norm_p.y;
        H_i(2) = norm_p.z;
        H_i.segment<3>(3) = A.transpose();
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            H_i.segment<3>(6) = B.transpose();
            H_i.segment<3>(9) = C.transpose();
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }

    solve_time += omp_get_wtime() - solve_start_;
}

class LaserMappingNode : public rclcpp::Node
{
public:
    LaserMappingNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : Node("laser_mapping", options)
    {
        this->declare_parameter<bool>("publish.path_en", true);
        this->declare_parameter<bool>("publish.effect_map_en", false);
        this->declare_parameter<bool>("publish.map_en", false);
        this->declare_parameter<bool>("publish.scan_publish_en", true);
        this->declare_parameter<bool>("publish.dense_publish_en", true);
        this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
        this->declare_parameter<double>("publish.blind_max", 0.0);
        this->declare_parameter<int>("max_iteration", 4);
        this->declare_parameter<string>("map_file_path", "");
        this->declare_parameter<string>("common.lid_topic", "/livox/lidar");
        this->declare_parameter<string>("common.imu_topic", "/livox/imu");
        this->declare_parameter<bool>("common.time_sync_en", false);
        this->declare_parameter<double>("common.time_offset_lidar_to_imu", 0.0);
        this->declare_parameter<double>("filter_size_corner", 0.5);
        this->declare_parameter<double>("filter_size_surf", 0.5);
        this->declare_parameter<double>("filter_size_map", 0.5);
        this->declare_parameter<double>("cube_side_length", 200.);
        this->declare_parameter<float>("mapping.det_range", 300.);
        this->declare_parameter<double>("mapping.fov_degree", 180.);
        this->declare_parameter<double>("mapping.gyr_cov", 0.1);
        this->declare_parameter<double>("mapping.acc_cov", 0.1);
        this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
        this->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
        this->declare_parameter<double>("mapping.process_noise.nv", 0.1);
        this->declare_parameter<double>("mapping.process_noise.nw", 0.1);
        this->declare_parameter<double>("mapping.process_noise.nbg", 0.0001);
        this->declare_parameter<double>("mapping.process_noise.nba", 0.0001);
        this->declare_parameter<double>("mapping.process_noise.nb_dvl", 0.0001);
        this->declare_parameter<double>("mapping.process_noise.nb_pressure", 0.0001);
        this->declare_parameter<double>("preprocess.blind", 0.01);
        this->declare_parameter<int>("preprocess.lidar_type", AVIA);
        this->declare_parameter<int>("preprocess.scan_line", 16);
        this->declare_parameter<int>("preprocess.timestamp_unit", US);
        this->declare_parameter<int>("preprocess.scan_rate", 10);
        this->declare_parameter<int>("point_filter_num", 2);
        this->declare_parameter<bool>("feature_extract_enable", false);
        this->declare_parameter<bool>("runtime_pos_log_enable", false);
        this->declare_parameter<bool>("mapping.extrinsic_est_en", true);
        this->declare_parameter<bool>("pcd_save.pcd_save_en", false);
        this->declare_parameter<int>("pcd_save.interval", -1);
        this->declare_parameter<vector<double>>("mapping.extrinsic_T", vector<double>());
        this->declare_parameter<vector<double>>("mapping.extrinsic_R", vector<double>());
        this->declare_parameter<bool>("dvl.enable", false);
        this->declare_parameter<string>("dvl.topic", "/auv/dvl");
        this->declare_parameter<vector<double>>("dvl.extrinsic_T", vector<double>());
        this->declare_parameter<vector<double>>("dvl.extrinsic_R", vector<double>());
        this->declare_parameter<double>("dvl.covariance_floor_std", 0.01);
        this->declare_parameter<double>("dvl.min_speed", 0.0);
        this->declare_parameter<bool>("dvl.hold_last_measurement", false);
        this->declare_parameter<double>("dvl.hold_max_age_sec", 0.25);
        this->declare_parameter<bool>("pressure.enable", false);
        this->declare_parameter<string>("pressure.topic", "/auv/pressure/scaled2");
        this->declare_parameter<vector<double>>("pressure.extrinsic_T", vector<double>());
        this->declare_parameter<double>("pressure.covariance_floor_std", 4.09);
        this->declare_parameter<double>("pressure.reference_pressure_pa", 101325.0);
        this->declare_parameter<bool>("loop_closure.enable", false);
        this->declare_parameter<string>("loop_closure.frontend_frame_topic", frontend_frame_topic);
        this->declare_parameter<string>("loop_closure.keyframe_id_topic", loop_closure_keyframe_id_topic);
        this->declare_parameter<string>("loop_closure.optimized_keyframes_topic", loop_closure_optimized_keyframes_topic);
        this->declare_parameter<string>("loop_closure.map_to_odom_topic", loop_closure_map_to_odom_topic);
        this->declare_parameter<int>("loop_closure.frontend_frame_stride", 1);
        this->declare_parameter<int>("loop_closure.frontend_frame_cache_size", 256);
        this->declare_parameter<bool>("loop_closure.frontend_feedback_enable", false);
        this->declare_parameter<double>("loop_closure.feedback_min_translation", 0.10);
        this->declare_parameter<double>("loop_closure.feedback_min_yaw_deg", 0.5);
        this->declare_parameter<double>("loop_closure.feedback_pose_position_cov", 1.0e-4);
        this->declare_parameter<double>("loop_closure.feedback_pose_rotation_cov", 1.0e-4);
        this->declare_parameter<double>("loop_closure.feedback_cross_cov_scale", 0.25);
        this->declare_parameter<int>("loop_closure.feedback_rebuild_search_num", 10);
        this->declare_parameter<bool>("dynamics.enable", true);
        this->declare_parameter<string>("dynamics.config_name", "default_value");
        this->declare_parameter<string>("dynamics.config_path", "");
        this->declare_parameter<string>("dynamics.forces_topic", "/auv/forces_desired_stamped");
        this->declare_parameter<double>("dynamics.model_trust", 1.0);
        this->declare_parameter<bool>("dynamics.thruster_meas_en", false);
        this->declare_parameter<double>("dynamics.thruster_acc_cov", 0.1);
    this->declare_parameter<string>("frames.map_frame", map_frame_id);
    this->declare_parameter<string>("frames.odom_frame", odom_frame_id);
    this->declare_parameter<string>("frames.body_frame", body_frame_id);

        this->get_parameter_or<bool>("publish.path_en", path_en, true);
        this->get_parameter_or<bool>("publish.effect_map_en", effect_pub_en, false);
        this->get_parameter_or<bool>("publish.map_en", map_pub_en, false);
        this->get_parameter_or<bool>("publish.scan_publish_en", scan_pub_en, true);
        this->get_parameter_or<bool>("publish.dense_publish_en", dense_pub_en, true);
        this->get_parameter_or<bool>("publish.scan_bodyframe_pub_en", scan_body_pub_en, true);
        this->get_parameter_or<double>("publish.blind_max", publish_blind_max, 0.0);
        this->get_parameter_or<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
        this->get_parameter_or<string>("map_file_path", map_file_path, "");
        this->get_parameter_or<string>("common.lid_topic", lid_topic, "/livox/lidar");
        this->get_parameter_or<string>("common.imu_topic", imu_topic,"/livox/imu");
        this->get_parameter_or<bool>("common.time_sync_en", time_sync_en, false);
        this->get_parameter_or<double>("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
        this->get_parameter_or<double>("filter_size_corner",filter_size_corner_min,0.5);
        this->get_parameter_or<double>("filter_size_surf",filter_size_surf_min,0.5);
        this->get_parameter_or<double>("filter_size_map",filter_size_map_min,0.5);
        this->get_parameter_or<double>("cube_side_length",cube_len,200.f);
        this->get_parameter_or<float>("mapping.det_range",DET_RANGE,300.f);
        this->get_parameter_or<double>("mapping.fov_degree",fov_deg,180.f);
        this->get_parameter_or<double>("mapping.gyr_cov",gyr_cov,0.1);
        this->get_parameter_or<double>("mapping.acc_cov",acc_cov,0.1);
        this->get_parameter_or<double>("mapping.b_gyr_cov",b_gyr_cov,0.0001);
        this->get_parameter_or<double>("mapping.b_acc_cov",b_acc_cov,0.0001);
        double proc_nv = acc_cov;
        double proc_nw = gyr_cov;
        double proc_nbg = b_gyr_cov;
        double proc_nba = b_acc_cov;
        double proc_nb_dvl = b_acc_cov;
        double proc_nb_pressure = b_acc_cov;
        this->get_parameter_or<double>("mapping.process_noise.nv", proc_nv, acc_cov);
        this->get_parameter_or<double>("mapping.process_noise.nw", proc_nw, gyr_cov);
        this->get_parameter_or<double>("mapping.process_noise.nbg", proc_nbg, b_gyr_cov);
        this->get_parameter_or<double>("mapping.process_noise.nba", proc_nba, b_acc_cov);
        this->get_parameter_or<double>("mapping.process_noise.nb_dvl", proc_nb_dvl, b_acc_cov);
        this->get_parameter_or<double>("mapping.process_noise.nb_pressure", proc_nb_pressure, b_acc_cov);
        this->get_parameter_or<double>("preprocess.blind", p_pre->blind, 0.01);
        this->get_parameter_or<int>("preprocess.lidar_type", p_pre->lidar_type, AVIA);
        this->get_parameter_or<int>("preprocess.scan_line", p_pre->N_SCANS, 16);
        this->get_parameter_or<int>("preprocess.timestamp_unit", p_pre->time_unit, US);
        this->get_parameter_or<int>("preprocess.scan_rate", p_pre->SCAN_RATE, 10);
        this->get_parameter_or<int>("point_filter_num", p_pre->point_filter_num, 2);
        this->get_parameter_or<bool>("feature_extract_enable", p_pre->feature_enabled, false);
        this->get_parameter_or<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
        this->get_parameter_or<bool>("mapping.extrinsic_est_en", extrinsic_est_en, true);
        this->get_parameter_or<bool>("pcd_save.pcd_save_en", pcd_save_en, false);
        this->get_parameter_or<int>("pcd_save.interval", pcd_save_interval, -1);
        this->get_parameter_or<vector<double>>("mapping.extrinsic_T", extrinT, vector<double>());
        this->get_parameter_or<vector<double>>("mapping.extrinsic_R", extrinR, vector<double>());
        this->get_parameter_or<bool>("dvl.enable", dvl_enabled, false);
        this->get_parameter_or<string>("dvl.topic", dvl_topic, "/auv/dvl");
        this->get_parameter_or<vector<double>>("dvl.extrinsic_T", dvl_extrinT, vector<double>());
        this->get_parameter_or<vector<double>>("dvl.extrinsic_R", dvl_extrinR, vector<double>());
        this->get_parameter_or<double>("dvl.covariance_floor_std", dvl_cov_floor_std, 0.01);
        this->get_parameter_or<double>("dvl.min_speed", dvl_min_speed, 0.0);
        this->get_parameter_or<bool>("dvl.hold_last_measurement", dvl_hold_enabled, false);
        this->get_parameter_or<double>("dvl.hold_max_age_sec", dvl_hold_max_age_sec, 0.25);
        this->get_parameter_or<bool>("pressure.enable", pressure_enabled, false);
        this->get_parameter_or<string>("pressure.topic", pressure_topic, "/auv/pressure/scaled2");
        this->get_parameter_or<vector<double>>("pressure.extrinsic_T", pressure_extrinT, vector<double>());
        this->get_parameter_or<double>("pressure.covariance_floor_std", pressure_cov_floor_std, 4.09);
        this->get_parameter_or<double>("pressure.reference_pressure_pa", pressure_reference_pa, 101325.0);
        this->get_parameter_or<bool>("loop_closure.enable", loop_closure_enabled, false);
        this->get_parameter_or<string>("loop_closure.frontend_frame_topic", frontend_frame_topic, frontend_frame_topic);
        this->get_parameter_or<string>("loop_closure.keyframe_id_topic", loop_closure_keyframe_id_topic, loop_closure_keyframe_id_topic);
        this->get_parameter_or<string>("loop_closure.optimized_keyframes_topic", loop_closure_optimized_keyframes_topic, loop_closure_optimized_keyframes_topic);
        this->get_parameter_or<string>("loop_closure.map_to_odom_topic", loop_closure_map_to_odom_topic, loop_closure_map_to_odom_topic);
        this->get_parameter_or<int>("loop_closure.frontend_frame_stride", loop_closure_frontend_frame_stride, 1);
        int frontend_frame_cache_size_param = static_cast<int>(loop_closure_frontend_frame_cache_size);
        this->get_parameter_or<int>("loop_closure.frontend_frame_cache_size", frontend_frame_cache_size_param, frontend_frame_cache_size_param);
        loop_closure_frontend_frame_cache_size = static_cast<size_t>(std::max(16, frontend_frame_cache_size_param));
        this->get_parameter_or<bool>("loop_closure.frontend_feedback_enable", loop_closure_frontend_feedback_enable, false);
        this->get_parameter_or<double>("loop_closure.feedback_min_translation", loop_closure_feedback_min_translation, 0.10);
        this->get_parameter_or<double>("loop_closure.feedback_min_yaw_deg", loop_closure_feedback_min_yaw_deg, 0.5);
        this->get_parameter_or<double>("loop_closure.feedback_pose_position_cov", loop_closure_feedback_pose_position_cov, 1.0e-4);
        this->get_parameter_or<double>("loop_closure.feedback_pose_rotation_cov", loop_closure_feedback_pose_rotation_cov, 1.0e-4);
        this->get_parameter_or<double>("loop_closure.feedback_cross_cov_scale", loop_closure_feedback_cross_cov_scale, 0.25);
        this->get_parameter_or<int>("loop_closure.feedback_rebuild_search_num", loop_closure_feedback_rebuild_search_num, 10);
        loop_closure_feedback_cross_cov_scale = std::clamp(loop_closure_feedback_cross_cov_scale, 0.0, 1.0);
        loop_closure_feedback_rebuild_search_num = std::max(0, loop_closure_feedback_rebuild_search_num);
        this->get_parameter_or<bool>("dynamics.enable", dynamics_enabled_, true);
        this->get_parameter_or<string>("dynamics.config_name", dynamics_config_name_, "default_value");
        this->get_parameter_or<string>("dynamics.config_path", dynamics_config_path_, "");
        this->get_parameter_or<string>("dynamics.forces_topic", dynamics_forces_topic_, "/auv/forces_desired_stamped");
        this->get_parameter_or<double>("dynamics.model_trust", dynamics_model_trust_, 1.0);
        this->get_parameter_or<bool>("dynamics.thruster_meas_en", thruster_meas_en_, false);
        this->get_parameter_or<double>("dynamics.thruster_acc_cov", thruster_acc_cov_, 0.1);
        this->get_parameter_or<string>("frames.map_frame", map_frame_id, map_frame_id);
        this->get_parameter_or<string>("frames.odom_frame", odom_frame_id, odom_frame_id);
        this->get_parameter_or<string>("frames.body_frame", body_frame_id, body_frame_id);
        loop_closure_frontend_frame_stride = std::max(1, loop_closure_frontend_frame_stride);
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
    #endif

        RCLCPP_INFO(this->get_logger(), "p_pre->lidar_type %d", p_pre->lidar_type);

        path.header.stamp = this->get_clock()->now();
        path.header.frame_id = odom_frame_id;

        // /*** variables definition ***/
        // int effect_feat_num = 0, frame_num = 0;
        // double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
        // bool flg_EKF_converged, EKF_stop_flg = 0;

        FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
        HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

        _featsArray.reset(new PointCloudXYZI());

        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
#ifdef FASTLIO_USE_CUDA
    gpu_downsampler_surf = std::make_unique<fastlio::gpu::VoxelDownsampler>();
    if (gpu_downsampler_surf && !gpu_downsampler_surf->available())
    {
        RCLCPP_WARN(this->get_logger(), "GPU voxel downsampler unavailable, falling back to CPU VoxelGrid");
    }
    else if (gpu_downsampler_surf)
    {
        gpu_downsampler_surf->set_leaf_size(static_cast<float>(filter_size_surf_min));
        RCLCPP_INFO(this->get_logger(), "GPU voxel downsampler initialized (leaf: %.3fm)", filter_size_surf_min);
    }

    gpu_point_transformer = std::make_unique<fastlio::gpu::PointTransformer>();
    if (gpu_point_transformer && !gpu_point_transformer->available())
    {
        RCLCPP_WARN(this->get_logger(), "GPU point transformer unavailable, using CPU transforms");
    }
    else if (gpu_point_transformer)
    {
        RCLCPP_INFO(this->get_logger(), "GPU point transformer initialized for correspondence/map updates");
    }
#endif
        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));

        Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
        p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
        p_imu->set_process_noise(V3D(proc_nv, proc_nv, proc_nv), V3D(proc_nw, proc_nw, proc_nw),
                     V3D(proc_nbg, proc_nbg, proc_nbg), V3D(proc_nba, proc_nba, proc_nba),
                     V3D(proc_nb_dvl, proc_nb_dvl, proc_nb_dvl), proc_nb_pressure);
        p_imu->set_dvl_params(dvl_cov_floor_std, dvl_min_speed);
        p_imu->set_dvl_hold(dvl_hold_enabled, dvl_hold_max_age_sec);
        p_imu->set_dynamics_trust(dynamics_model_trust_);
        p_imu->set_thruster_meas(thruster_meas_en_,
                                 V3D(thruster_acc_cov_, thruster_acc_cov_, thruster_acc_cov_));

        if (dvl_extrinT.size() != 3)
        {
            dvl_extrinT.assign(3, 0.0);
        }
        if (dvl_extrinR.size() != 9)
        {
            dvl_extrinR.assign(9, 0.0);
            dvl_extrinR[0] = dvl_extrinR[4] = dvl_extrinR[8] = 1.0;
        }
        Eigen::Vector3d r_bd_b(dvl_extrinT[0], dvl_extrinT[1], dvl_extrinT[2]);
        Eigen::Matrix3d R_b_d;
        R_b_d << dvl_extrinR[0], dvl_extrinR[1], dvl_extrinR[2],
                 dvl_extrinR[3], dvl_extrinR[4], dvl_extrinR[5],
                 dvl_extrinR[6], dvl_extrinR[7], dvl_extrinR[8];
        set_dvl_mount(r_bd_b, R_b_d);

        if (pressure_extrinT.size() != 3)
        {
            pressure_extrinT.assign(3, 0.0);
        }
        Eigen::Vector3d r_bp_b(pressure_extrinT[0], pressure_extrinT[1], pressure_extrinT[2]);
        set_pressure_mount(r_bp_b);
        pressure_ref_initialized = false;
        pressure_ref_sum = 0.0;
        pressure_ref_count = 0;

        if (dynamics_enabled_)
        {
            try
            {
                if (dynamics_config_path_.empty())
                {
                    throw std::runtime_error("dynamics.config_path is empty");
                }

                libconfig::Config config;
                config.readFile(dynamics_config_path_.c_str());

                dynamics_model_ = std::make_shared<mvm::UnderwaterVehicleModel>(config, dynamics_config_name_);
                fastlio::dynamics::set_model(dynamics_model_);
                RCLCPP_INFO(this->get_logger(), "Dynamics model loaded: %s", dynamics_config_path_.c_str());
            }
            catch (const std::exception &ex)
            {
                dynamics_enabled_ = false;
                RCLCPP_WARN(this->get_logger(), "Dynamics model disabled: %s", ex.what());
            }
        }

        fill(epsi, epsi + state_ikfom::DOF, 0.001);
        kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

        /*** debug record ***/
        // FILE *fp;
        string pos_log_dir = root_dir + "/Log/pos_log.txt";
        fp = fopen(pos_log_dir.c_str(),"w");

        // ofstream fout_pre, fout_out, fout_dbg;
        if (runtime_pos_log)
        {
            fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
            fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
            fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
            if (fout_pre && fout_out)
                cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
            else
                cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;
        }

        /*** ROS subscribe initialization ***/
        if (p_pre->lidar_type == AVIA)
        {
#ifdef FASTLIO_HAS_LIVOX
            sub_pcl_livox_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(lid_topic, 20, livox_pcl_cbk);
#else
            RCLCPP_FATAL(this->get_logger(), "Livox LiDAR support is disabled in this build. Rebuild with livox_ros_driver2 installed or choose a different lidar_type.");
            throw std::runtime_error("FAST_LIO was built without Livox support");
#endif
        }
        else
        {
            sub_pcl_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, rclcpp::SensorDataQoS(), standard_pcl_cbk);
        }
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(imu_topic, rclcpp::SensorDataQoS(), imu_cbk);
        sub_thrusters_ = this->create_subscription<sensor_msgs::msg::JointState>(dynamics_forces_topic_, 50, thruster_cbk);
        if (dvl_enabled)
        {
            sub_dvl_ = this->create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(dvl_topic, rclcpp::SensorDataQoS(), dvl_cbk);
        }
        if (pressure_enabled)
        {
            sub_pressure_ = this->create_subscription<sensor_msgs::msg::FluidPressure>(pressure_topic, 50, pressure_cbk);
        }
        pubLaserCloudFull_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 20);
        pubLaserCloudFull_body_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 20);
        pubLaserCloudEffect_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 20);
        pubLaserCloudMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", 20);
        pubOdomAftMapped_ = this->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 20);
        pubPath_ = this->create_publisher<nav_msgs::msg::Path>("/path", 20);
        if (loop_closure_enabled)
        {
            pubFrontendFrame_ = this->create_publisher<fast_lio::msg::FrontendFrame>(frontend_frame_topic, 20);
            {
                std::lock_guard<std::mutex> lock(loop_closure_tf_mutex);
                loop_closure_map_to_odom_tf.header.frame_id = map_frame_id;
                loop_closure_map_to_odom_tf.child_frame_id = odom_frame_id;
                loop_closure_map_to_odom_tf.transform.translation.x = 0.0;
                loop_closure_map_to_odom_tf.transform.translation.y = 0.0;
                loop_closure_map_to_odom_tf.transform.translation.z = 0.0;
                loop_closure_map_to_odom_tf.transform.rotation.x = 0.0;
                loop_closure_map_to_odom_tf.transform.rotation.y = 0.0;
                loop_closure_map_to_odom_tf.transform.rotation.z = 0.0;
                loop_closure_map_to_odom_tf.transform.rotation.w = 1.0;
                loop_closure_map_to_odom_available = true;
            }
            auto map_to_odom_qos = rclcpp::QoS(1).transient_local();
            sub_loop_closure_map_to_odom_ =
                this->create_subscription<geometry_msgs::msg::TransformStamped>(
                    loop_closure_map_to_odom_topic,
                    map_to_odom_qos,
                    loop_closure_map_to_odom_cbk);
            if (loop_closure_frontend_feedback_enable)
            {
                sub_loop_closure_keyframe_id_ =
                    this->create_subscription<std_msgs::msg::UInt32>(
                        loop_closure_keyframe_id_topic,
                        20,
                        loop_closure_keyframe_id_cbk);
                sub_loop_closure_optimized_keyframes_ =
                    this->create_subscription<fast_lio::msg::OptimizedKeyframes>(
                        loop_closure_optimized_keyframes_topic,
                        rclcpp::QoS(1).transient_local(),
                        loop_closure_optimized_keyframes_cbk);
                reset_loop_closure_map_to_odom_identity();
            }
        }
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    static_tf_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
    if (!loop_closure_enabled)
    {
        geometry_msgs::msg::TransformStamped map_to_camera;
        map_to_camera.header.stamp = this->now();
        map_to_camera.header.frame_id = map_frame_id;
        map_to_camera.child_frame_id = odom_frame_id;
        map_to_camera.transform.translation.x = 0.0;
        map_to_camera.transform.translation.y = 0.0;
        map_to_camera.transform.translation.z = 0.0;
        map_to_camera.transform.rotation.x = 0.0;
        map_to_camera.transform.rotation.y = 0.0;
        map_to_camera.transform.rotation.z = 0.0;
        map_to_camera.transform.rotation.w = 1.0;
        static_tf_broadcaster_->sendTransform(map_to_camera);
    }

        //------------------------------------------------------------------------------------------------------
        processing_worker_running_.store(true, std::memory_order_release);
        processing_thread_ = std::thread(&LaserMappingNode::processing_worker_loop, this);

        auto map_period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0));
        map_pub_timer_ = rclcpp::create_timer(this, this->get_clock(), map_period_ms, std::bind(&LaserMappingNode::map_publish_callback, this));

        map_save_srv_ = this->create_service<std_srvs::srv::Trigger>("map_save", std::bind(&LaserMappingNode::map_save_callback, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "Node init finished.");
    }

    ~LaserMappingNode()
    {
        processing_worker_running_.store(false, std::memory_order_release);
        process_wakeup_requested.store(true, std::memory_order_release);
        sig_buffer.notify_all();
        if (processing_thread_.joinable())
        {
            processing_thread_.join();
        }
        if (fout_out.is_open()) fout_out.close();
        if (fout_pre.is_open()) fout_pre.close();
        if (fout_dbg.is_open()) fout_dbg.close();
        if (fp != nullptr) fclose(fp);
    }

    void processing_worker_loop()
    {
        while (processing_worker_running_.load(std::memory_order_acquire) && rclcpp::ok() && !flg_exit)
        {
            std::unique_lock<std::mutex> lock(mtx_buffer);
            sig_buffer.wait(lock, [] {
                return flg_exit || process_wakeup_requested.load(std::memory_order_acquire);
            });
            process_wakeup_requested.store(false, std::memory_order_release);
            lock.unlock();

            process_available_scans();
        }
    }

    void process_available_scans()
    {
        while(sync_packages(Measures))
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                return;
            }

            double t0,t1,t2,t3,t5;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            t0 = omp_get_wtime();
            stage_downsample_time = 0.0;
            stage_transform_time = 0.0;
            stage_knn_time = 0.0;
            stage_map_add_time = 0.0;
            kdtree_search_time = 0.0;
            kdtree_incremental_time = 0.0;

            p_imu->Process(Measures, kf, feats_undistort);

            if (pressure_enabled && !Measures.pressure.empty())
            {
                for (const auto &pressure_msg : Measures.pressure)
                {
                    const double p_abs = pressure_msg->fluid_pressure;
                    if (!std::isfinite(p_abs))
                    {
                        continue;
                    }

                    if (!pressure_ref_initialized)
                    {
                        pressure_ref_sum += p_abs;
                        pressure_ref_count++;
                        if (pressure_ref_count >= kPressureRefInitSamples)
                        {
                            pressure_reference_pa = pressure_ref_sum / static_cast<double>(pressure_ref_count);
                            pressure_ref_initialized = true;
                            RCLCPP_INFO(this->get_logger(), "Pressure reference initialized at %.3f Pa (%d samples)",
                                        pressure_reference_pa, pressure_ref_count);
                        }
                        continue;
                    }

                    const state_ikfom state_pre_pressure = kf.get_x();
                    const Eigen::Vector3d grav_world(state_pre_pressure.grav[0], state_pre_pressure.grav[1], state_pre_pressure.grav[2]);
                    const double grav_mag = std::max(1e-3, grav_world.norm());
                    const double pressure_to_depth = 1.0 / (kPressureDensityFreshWater * grav_mag);
                    const double depth = (p_abs - pressure_reference_pa) * pressure_to_depth;
                    if (!std::isfinite(depth))
                    {
                        continue;
                    }

                    set_pressure_cov(pressure_cov_floor_std * pressure_cov_floor_std);

                    double z_arr[1] = {depth};
                    vect1 z_pressure(z_arr, 1);
                    kf.update_iterated_dyn_runtime_share(z_pressure, h_pressure_share);
                }
            }

            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "No point, skip this scan!");
                return;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            const double downsample_stage_start = omp_get_wtime();
#ifdef FASTLIO_USE_CUDA
            static bool logged_gpu_downsample_once = false;
            static bool logged_gpu_downsample_fallback = false;
            bool gpu_filtered = false;
            if (gpu_downsampler_surf && gpu_downsampler_surf->available())
            {
                gpu_filtered = gpu_downsampler_surf->filter(*feats_undistort, *feats_down_body);
                if (gpu_filtered && !logged_gpu_downsample_once)
                {
                    RCLCPP_INFO(this->get_logger(), "CUDA voxel downsampler active during filtering.");
                    logged_gpu_downsample_once = true;
                }
                else if (!gpu_filtered && !logged_gpu_downsample_fallback)
                {
                    RCLCPP_WARN(this->get_logger(), "CUDA voxel downsampler filter() failed; falling back to CPU path.");
                    logged_gpu_downsample_fallback = true;
                }
            }
            if (!gpu_filtered)
#endif
            {
                downSizeFilterSurf.setInputCloud(feats_undistort);
                downSizeFilterSurf.filter(*feats_down_body);
            }
            stage_downsample_time = omp_get_wtime() - downsample_stage_start;
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/
            if(ikdtree.Root_Node == nullptr)
            {
                RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                return;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "No point, skip this scan!");
                return;
            }
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            if (runtime_pos_log)
            {
                fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;
            }

            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            for (auto &neighbors : Nearest_Points)
            {
                if (neighbors.capacity() < static_cast<size_t>(NUM_MATCH_POINTS))
                {
                    neighbors.reserve(NUM_MATCH_POINTS);
                }
            }
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped_, tf_broadcaster_);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath_);
            if (scan_pub_en)      publish_frame_world(pubLaserCloudFull_);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body_);
            if (loop_closure_enabled && (frontend_scan_id % static_cast<uint32_t>(loop_closure_frontend_frame_stride) == 0)) {
                publish_frontend_frame(pubFrontendFrame_);
            }
            if (loop_closure_enabled && loop_closure_frontend_feedback_enable) {
                maybe_apply_loop_closure_frontend_feedback(this->get_logger());
            }
            if (effect_pub_en) publish_effect_world(pubLaserCloudEffect_);
            // if (map_pub_en) publish_map(pubLaserCloudMap_);

            ++frontend_scan_id;

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
        }
    }

    void map_publish_callback()
    {
        if (map_pub_en) publish_map(pubLaserCloudMap_);
    }

    void map_save_callback(std_srvs::srv::Trigger::Request::ConstSharedPtr req, std_srvs::srv::Trigger::Response::SharedPtr res)
    {
        RCLCPP_INFO(this->get_logger(), "Saving map to %s...", map_file_path.c_str());
        if (pcd_save_en)
        {
            save_to_pcd();
            res->success = true;
            res->message = "Map saved.";
        }
        else
        {
            res->success = false;
            res->message = "Map save disabled.";
        }
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    rclcpp::Publisher<fast_lio::msg::FrontendFrame>::SharedPtr pubFrontendFrame_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_thrusters_;
    rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr sub_dvl_;
    rclcpp::Subscription<sensor_msgs::msg::FluidPressure>::SharedPtr sub_pressure_;
    rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr sub_loop_closure_map_to_odom_;
    rclcpp::Subscription<std_msgs::msg::UInt32>::SharedPtr sub_loop_closure_keyframe_id_;
    rclcpp::Subscription<fast_lio::msg::OptimizedKeyframes>::SharedPtr sub_loop_closure_optimized_keyframes_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;
#ifdef FASTLIO_HAS_LIVOX
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;
#endif

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
    std::thread processing_thread_;
    std::atomic<bool> processing_worker_running_{false};
    rclcpp::TimerBase::SharedPtr map_pub_timer_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_srv_;

    bool dynamics_enabled_ = false;
    std::string dynamics_config_name_;
    std::string dynamics_config_path_;
    std::string dynamics_forces_topic_;
    double dynamics_model_trust_ = 1.0;
    bool thruster_meas_en_ = false;
    double thruster_acc_cov_ = 0.1;
    std::shared_ptr<mvm::UnderwaterVehicleModel> dynamics_model_;

    bool effect_pub_en = false, map_pub_en = false;
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    double epsi[state_ikfom::DOF] = {0.001};

    FILE *fp = nullptr;
    ofstream fout_pre, fout_out, fout_dbg;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    signal(SIGINT, SigHandle);

#ifdef FASTLIO_USE_CUDA
    auto node = std::make_shared<LaserMappingNode>();
    const unsigned int hw_threads = std::max(1u, std::thread::hardware_concurrency());
    const size_t executor_threads = std::min<size_t>(2, hw_threads);
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), executor_threads);
    executor.add_node(node);
    executor.spin();
#else
    rclcpp::spin(std::make_shared<LaserMappingNode>());
#endif

    if (rclcpp::ok())
        rclcpp::shutdown();
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
