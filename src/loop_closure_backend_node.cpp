#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <rclcpp/rclcpp.hpp>
#include <fast_lio/msg/frontend_frame.hpp>
#include <fast_lio/msg/optimized_keyframes.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/u_int32.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

using PointType = pcl::PointXYZINormal;

namespace
{

struct Pose6D
{
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
};

struct KeyframeData
{
  uint32_t scan_id = 0;
  builtin_interfaces::msg::Time stamp;
  double stamp_sec = 0.0;
  Pose6D raw_pose;
  pcl::PointCloud<PointType>::Ptr cloud_body;
};

struct IcpJob
{
  std::size_t history_idx = 0;
  std::size_t current_idx = 0;
  builtin_interfaces::msg::Time history_stamp;
  builtin_interfaces::msg::Time current_stamp;
  Pose6D history_pose;
  Pose6D current_pose;
  pcl::PointCloud<PointType>::Ptr history_cloud;
  pcl::PointCloud<PointType>::Ptr current_cloud;
};

struct IcpResult
{
  bool accepted = false;
  double fitness = std::numeric_limits<double>::infinity();
  double correction_translation = 0.0;
  double correction_yaw_deg = 0.0;
  std::string reject_reason = "uninitialized";
  gtsam::Pose3 relative_pose;
  pcl::PointCloud<PointType>::Ptr registered_cloud;
};

Pose6D pose_from_msg(const geometry_msgs::msg::Pose &pose_msg)
{
  Pose6D pose;
  pose.x = pose_msg.position.x;
  pose.y = pose_msg.position.y;
  pose.z = pose_msg.position.z;

  tf2::Quaternion quat(
    pose_msg.orientation.x,
    pose_msg.orientation.y,
    pose_msg.orientation.z,
    pose_msg.orientation.w);
  tf2::Matrix3x3(quat).getRPY(pose.roll, pose.pitch, pose.yaw);
  return pose;
}

geometry_msgs::msg::Pose pose_to_msg(const Pose6D &pose)
{
  geometry_msgs::msg::Pose pose_msg;
  pose_msg.position.x = pose.x;
  pose_msg.position.y = pose.y;
  pose_msg.position.z = pose.z;

  tf2::Quaternion quat;
  quat.setRPY(pose.roll, pose.pitch, pose.yaw);
  pose_msg.orientation.x = quat.x();
  pose_msg.orientation.y = quat.y();
  pose_msg.orientation.z = quat.z();
  pose_msg.orientation.w = quat.w();
  return pose_msg;
}

Pose6D diff_pose(const Pose6D &from, const Pose6D &to)
{
  Eigen::Affine3f tf_from = pcl::getTransformation(
    static_cast<float>(from.x),
    static_cast<float>(from.y),
    static_cast<float>(from.z),
    static_cast<float>(from.roll),
    static_cast<float>(from.pitch),
    static_cast<float>(from.yaw));
  Eigen::Affine3f tf_to = pcl::getTransformation(
    static_cast<float>(to.x),
    static_cast<float>(to.y),
    static_cast<float>(to.z),
    static_cast<float>(to.roll),
    static_cast<float>(to.pitch),
    static_cast<float>(to.yaw));
  Eigen::Affine3f delta = tf_from.inverse() * tf_to;

  Pose6D out;
  float dx = 0.0f;
  float dy = 0.0f;
  float dz = 0.0f;
  float droll = 0.0f;
  float dpitch = 0.0f;
  float dyaw = 0.0f;
  pcl::getTranslationAndEulerAngles(delta, dx, dy, dz, droll, dpitch, dyaw);
  out.x = std::abs(dx);
  out.y = std::abs(dy);
  out.z = std::abs(dz);
  out.roll = std::abs(droll);
  out.pitch = std::abs(dpitch);
  out.yaw = std::abs(dyaw);
  return out;
}

double pose_translation_norm(const Pose6D &pose)
{
  return std::sqrt(pose.x * pose.x + pose.y * pose.y + pose.z * pose.z);
}

Eigen::Affine3f pose_to_affine(const Pose6D &pose)
{
  return pcl::getTransformation(
    static_cast<float>(pose.x),
    static_cast<float>(pose.y),
    static_cast<float>(pose.z),
    static_cast<float>(pose.roll),
    static_cast<float>(pose.pitch),
    static_cast<float>(pose.yaw));
}

gtsam::Pose3 pose_to_gtsam(const Pose6D &pose)
{
  return gtsam::Pose3(
    gtsam::Rot3::RzRyRx(pose.roll, pose.pitch, pose.yaw),
    gtsam::Point3(pose.x, pose.y, pose.z));
}

Pose6D pose_from_gtsam(const gtsam::Pose3 &pose)
{
  Pose6D out;
  out.x = pose.translation().x();
  out.y = pose.translation().y();
  out.z = pose.translation().z();
  out.roll = pose.rotation().roll();
  out.pitch = pose.rotation().pitch();
  out.yaw = pose.rotation().yaw();
  return out;
}

pcl::PointCloud<PointType>::Ptr transform_cloud(
  const pcl::PointCloud<PointType>::Ptr &cloud_body,
  const Pose6D &pose)
{
  pcl::PointCloud<PointType>::Ptr cloud_odom(new pcl::PointCloud<PointType>());
  pcl::transformPointCloud(*cloud_body, *cloud_odom, pose_to_affine(pose));
  return cloud_odom;
}

}  // namespace

class LoopClosureBackendNode : public rclcpp::Node
{
public:
  LoopClosureBackendNode()
  : Node("fastlio_loop_closure_backend")
  {
    declare_parameter<bool>("loop_closure.enable", false);
    declare_parameter<bool>("loop_closure.frontend_feedback_enable", false);
    declare_parameter<std::string>("loop_closure.frontend_frame_topic", "/fastlio/frontend_frame");
    declare_parameter<std::string>("loop_closure.keyframe_id_topic", "/fastlio/keyframe_id");
    declare_parameter<std::string>("loop_closure.optimized_keyframes_topic", "/fastlio/optimized_keyframes");
    declare_parameter<std::string>("loop_closure.map_to_odom_topic", "/fastlio/map_to_odom");
    declare_parameter<std::string>("frames.map_frame", std::string("World"));
    declare_parameter<std::string>("frames.odom_frame", std::string("camera_init"));
    declare_parameter<double>("loop_closure.keyframe_meter_gap", 1.0);
    declare_parameter<double>("loop_closure.keyframe_deg_gap", 10.0);
    declare_parameter<double>("loop_closure.history_keyframe_search_radius", 5.0);
    declare_parameter<double>("loop_closure.history_keyframe_search_time_diff", 30.0);
    declare_parameter<int>("loop_closure.history_keyframe_search_num", 10);
    declare_parameter<double>("loop_closure.loop_closure_frequency", 2.0);
    declare_parameter<double>("loop_closure.icp_process_frequency", 1.0);
    declare_parameter<double>("loop_closure.keyframe_downsample_leaf", 0.4);
    declare_parameter<double>("loop_closure.map_downsample_leaf", 0.4);
    declare_parameter<double>("loop_closure.icp_max_correspondence_distance", 15.0);
    declare_parameter<int>("loop_closure.icp_max_iterations", 100);
    declare_parameter<double>("loop_closure.loop_fitness_score_threshold", 0.3);
    declare_parameter<double>("loop_closure.max_loop_correction_translation", 1.5);
    declare_parameter<int>("loop_closure.graph_update_times", 2);
    declare_parameter<double>("loop_closure.odom_noise_rotation", 1e-6);
    declare_parameter<double>("loop_closure.odom_noise_translation", 1e-4);
    declare_parameter<double>("loop_closure.loop_noise_score", 0.5);

    get_parameter("loop_closure.enable", enabled_);
    get_parameter("loop_closure.frontend_feedback_enable", frontend_feedback_enable_);
    get_parameter("loop_closure.frontend_frame_topic", frontend_frame_topic_);
    get_parameter("loop_closure.keyframe_id_topic", keyframe_id_topic_);
    get_parameter("loop_closure.optimized_keyframes_topic", optimized_keyframes_topic_);
    get_parameter("loop_closure.map_to_odom_topic", map_to_odom_topic_);
    get_parameter("frames.map_frame", map_frame_);
    get_parameter("frames.odom_frame", odom_frame_);
    get_parameter("loop_closure.keyframe_meter_gap", keyframe_meter_gap_);
    get_parameter("loop_closure.keyframe_deg_gap", keyframe_deg_gap_);
    get_parameter("loop_closure.history_keyframe_search_radius", history_keyframe_search_radius_);
    get_parameter("loop_closure.history_keyframe_search_time_diff", history_keyframe_search_time_diff_);
    get_parameter("loop_closure.history_keyframe_search_num", history_keyframe_search_num_);
    get_parameter("loop_closure.loop_closure_frequency", loop_closure_frequency_);
    get_parameter("loop_closure.icp_process_frequency", icp_process_frequency_);
    get_parameter("loop_closure.keyframe_downsample_leaf", keyframe_downsample_leaf_);
    get_parameter("loop_closure.map_downsample_leaf", map_downsample_leaf_);
    get_parameter("loop_closure.icp_max_correspondence_distance", icp_max_correspondence_distance_);
    get_parameter("loop_closure.icp_max_iterations", icp_max_iterations_);
    get_parameter("loop_closure.loop_fitness_score_threshold", loop_fitness_score_threshold_);
    get_parameter("loop_closure.max_loop_correction_translation", max_loop_correction_translation_);
    get_parameter("loop_closure.graph_update_times", graph_update_times_);
    get_parameter("loop_closure.odom_noise_rotation", odom_noise_rotation_);
    get_parameter("loop_closure.odom_noise_translation", odom_noise_translation_);
    get_parameter("loop_closure.loop_noise_score", loop_noise_score_);

    keyframe_rad_gap_ = keyframe_deg_gap_ * M_PI / 180.0;

    if (!enabled_) {
      return;
    }

    init_pose_graph();

    sub_frontend_frame_ = create_subscription<fast_lio::msg::FrontendFrame>(
      frontend_frame_topic_, rclcpp::SensorDataQoS(),
      std::bind(&LoopClosureBackendNode::frontend_frame_cb, this, std::placeholders::_1));

    if (frontend_feedback_enable_) {
      pub_keyframe_id_ = create_publisher<std_msgs::msg::UInt32>(keyframe_id_topic_, 20);
      pub_optimized_keyframes_ = create_publisher<fast_lio::msg::OptimizedKeyframes>(
        optimized_keyframes_topic_, rclcpp::QoS(1).transient_local());
    } else {
      pub_map_to_odom_ = create_publisher<geometry_msgs::msg::TransformStamped>(
        map_to_odom_topic_, rclcpp::QoS(1).transient_local());
    }

    const auto loop_period = std::chrono::duration<double>(1.0 / std::max(0.1, loop_closure_frequency_));
    loop_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::milliseconds>(loop_period),
      std::bind(&LoopClosureBackendNode::detect_loop_candidates, this));

    const auto icp_period = std::chrono::duration<double>(1.0 / std::max(0.1, icp_process_frequency_));
    icp_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::milliseconds>(icp_period),
      std::bind(&LoopClosureBackendNode::process_icp_queue, this));

    if (!frontend_feedback_enable_) {
      publish_map_to_odom_identity();
    }
  }

private:
  void init_pose_graph()
  {
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam_ = std::make_unique<gtsam::ISAM2>(parameters);

    gtsam::Vector6 prior_noise_vector;
    prior_noise_vector << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
    prior_noise_ = gtsam::noiseModel::Diagonal::Variances(prior_noise_vector);

    gtsam::Vector6 odom_noise_vector;
    odom_noise_vector << odom_noise_rotation_, odom_noise_rotation_, odom_noise_rotation_,
      odom_noise_translation_, odom_noise_translation_, odom_noise_translation_;
    odom_noise_ = gtsam::noiseModel::Diagonal::Variances(odom_noise_vector);

    gtsam::Vector6 loop_noise_vector;
    loop_noise_vector << loop_noise_score_, loop_noise_score_, loop_noise_score_,
      loop_noise_score_, loop_noise_score_, loop_noise_score_;
    robust_loop_noise_ = gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Cauchy::Create(1.0),
      gtsam::noiseModel::Diagonal::Variances(loop_noise_vector));
  }

  Pose6D get_pose_locked(std::size_t idx, bool optimized) const
  {
    if (optimized && idx < optimized_poses_.size()) {
      return optimized_poses_[idx];
    }
    return keyframes_[idx].raw_pose;
  }

  void frontend_frame_cb(const fast_lio::msg::FrontendFrame::SharedPtr msg)
  {
    if (!msg->header.frame_id.empty() && msg->header.frame_id != odom_frame_) {
      std::lock_guard<std::mutex> lock(mutex_);
      odom_frame_ = msg->header.frame_id;
    }

    const Pose6D pose = pose_from_msg(msg->pose_in_odom);

    std::lock_guard<std::mutex> lock(mutex_);
    if (!have_prev_frame_) {
      prev_frame_pose_ = pose;
      have_prev_frame_ = true;
      translation_accumulated_ = 0.0;
      rotation_accumulated_ = 0.0;
      add_keyframe_locked(*msg, pose);
      return;
    }

    const Pose6D delta = diff_pose(prev_frame_pose_, pose);
    translation_accumulated_ += pose_translation_norm(delta);
    rotation_accumulated_ += delta.roll + delta.pitch + delta.yaw;
    prev_frame_pose_ = pose;

    if (translation_accumulated_ > keyframe_meter_gap_ || rotation_accumulated_ > keyframe_rad_gap_) {
      translation_accumulated_ = 0.0;
      rotation_accumulated_ = 0.0;
      add_keyframe_locked(*msg, pose);
    }
  }

  void add_keyframe_locked(const fast_lio::msg::FrontendFrame &msg, const Pose6D &pose)
  {
    pcl::PointCloud<PointType>::Ptr cloud_body(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(msg.cloud_body, *cloud_body);

    pcl::VoxelGrid<PointType> downsampler;
    downsampler.setLeafSize(
      static_cast<float>(keyframe_downsample_leaf_),
      static_cast<float>(keyframe_downsample_leaf_),
      static_cast<float>(keyframe_downsample_leaf_));
    downsampler.setInputCloud(cloud_body);

    pcl::PointCloud<PointType>::Ptr cloud_body_ds(new pcl::PointCloud<PointType>());
    downsampler.filter(*cloud_body_ds);

    KeyframeData keyframe;
    keyframe.scan_id = msg.scan_id;
    keyframe.stamp = msg.header.stamp;
    keyframe.stamp_sec = rclcpp::Time(msg.header.stamp).seconds();
    keyframe.raw_pose = pose;
    keyframe.cloud_body = cloud_body_ds;
    keyframes_.push_back(keyframe);

    if (pub_keyframe_id_) {
      std_msgs::msg::UInt32 keyframe_id_msg;
      keyframe_id_msg.data = msg.scan_id;
      pub_keyframe_id_->publish(keyframe_id_msg);
    }

    add_pose_graph_keyframe_locked();
  }

  void add_pose_graph_keyframe_locked()
  {
    const std::size_t curr_idx = keyframes_.size() - 1;
    const gtsam::Pose3 pose_to = pose_to_gtsam(keyframes_[curr_idx].raw_pose);

    if (!posegraph_initialized_) {
      graph_.add(gtsam::PriorFactor<gtsam::Pose3>(0, pose_to, prior_noise_));
      initial_estimate_.insert(0, pose_to);
      posegraph_initialized_ = true;
    } else {
      const std::size_t prev_idx = curr_idx - 1;
      const gtsam::Pose3 pose_from = pose_to_gtsam(keyframes_[prev_idx].raw_pose);
      graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
        prev_idx,
        curr_idx,
        pose_from.between(pose_to),
        odom_noise_));
      initial_estimate_.insert(curr_idx, pose_to);
    }

    run_isam2_locked();
    publish_optimized_keyframes_locked();
    if (!frontend_feedback_enable_) {
      publish_map_to_odom_locked();
    }
  }

  void run_isam2_locked()
  {
    if (!isam_ || (graph_.empty() && initial_estimate_.empty())) {
      return;
    }

    isam_->update(graph_, initial_estimate_);
    isam_->update();
    for (int i = graph_update_times_; i > 0; --i) {
      isam_->update();
    }

    graph_.resize(0);
    initial_estimate_.clear();
    current_estimate_ = isam_->calculateEstimate();

    optimized_poses_.resize(keyframes_.size());
    for (std::size_t idx = 0; idx < keyframes_.size(); ++idx) {
      if (current_estimate_.exists(idx)) {
        optimized_poses_[idx] = pose_from_gtsam(current_estimate_.at<gtsam::Pose3>(idx));
      } else {
        optimized_poses_[idx] = keyframes_[idx].raw_pose;
      }
    }
  }

  void detect_loop_candidates()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (keyframes_.size() < 2) {
      return;
    }

    const std::size_t current_idx = keyframes_.size() - 1;
    if (pending_loop_index_container_.count(current_idx) > 0 ||
      accepted_loop_index_container_.count(current_idx) > 0)
    {
      return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr key_positions(new pcl::PointCloud<pcl::PointXYZ>());
    key_positions->reserve(keyframes_.size());
    for (std::size_t idx = 0; idx < keyframes_.size(); ++idx) {
      const Pose6D pose = keyframes_[idx].raw_pose;
      key_positions->push_back(pcl::PointXYZ(
        static_cast<float>(pose.x),
        static_cast<float>(pose.y),
        static_cast<float>(pose.z)));
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(key_positions);

    std::vector<int> indices;
    std::vector<float> sq_dists;
    kdtree.radiusSearch(key_positions->back(), history_keyframe_search_radius_, indices, sq_dists, 0);

    int selected_idx = -1;
    for (const int idx : indices) {
      if (idx < 0 || static_cast<std::size_t>(idx) == current_idx) {
        continue;
      }
      const double dt = std::abs(keyframes_[idx].stamp_sec - keyframes_[current_idx].stamp_sec);
      if (dt > history_keyframe_search_time_diff_) {
        selected_idx = idx;
        break;
      }
    }

    if (selected_idx < 0) {
      return;
    }

    pending_loop_index_container_[current_idx] = static_cast<std::size_t>(selected_idx);
    loop_icp_queue_.push_back(std::make_pair(static_cast<std::size_t>(selected_idx), current_idx));
  }

  pcl::PointCloud<PointType>::Ptr build_nearby_cloud_locked(std::size_t root_idx, int search_num) const
  {
    pcl::PointCloud<PointType>::Ptr near_keyframes(new pcl::PointCloud<PointType>());
    if (keyframes_.empty()) {
      return near_keyframes;
    }

    const int start_idx = std::max<int>(0, static_cast<int>(root_idx) - search_num);
    const int end_idx = std::min<int>(
      static_cast<int>(keyframes_.size()) - 1,
      static_cast<int>(root_idx) + search_num);

    for (int idx = start_idx; idx <= end_idx; ++idx) {
      *near_keyframes += *transform_cloud(keyframes_[idx].cloud_body, get_pose_locked(static_cast<std::size_t>(idx), true));
    }

    if (near_keyframes->empty()) {
      return near_keyframes;
    }

    pcl::VoxelGrid<PointType> downsampler;
    downsampler.setLeafSize(
      static_cast<float>(map_downsample_leaf_),
      static_cast<float>(map_downsample_leaf_),
      static_cast<float>(map_downsample_leaf_));
    pcl::PointCloud<PointType>::Ptr near_keyframes_ds(new pcl::PointCloud<PointType>());
    downsampler.setInputCloud(near_keyframes);
    downsampler.filter(*near_keyframes_ds);
    return near_keyframes_ds;
  }

  IcpJob make_icp_job_locked(std::size_t history_idx, std::size_t current_idx) const
  {
    IcpJob job;
    job.history_idx = history_idx;
    job.current_idx = current_idx;
    job.history_stamp = keyframes_[history_idx].stamp;
    job.current_stamp = keyframes_[current_idx].stamp;
    job.history_pose = get_pose_locked(history_idx, true);
    job.current_pose = get_pose_locked(current_idx, true);
    job.history_cloud = build_nearby_cloud_locked(history_idx, history_keyframe_search_num_);
    job.current_cloud = build_nearby_cloud_locked(current_idx, 0);
    return job;
  }

  IcpResult do_icp(const IcpJob &job)
  {
    IcpResult result;
    result.registered_cloud.reset(new pcl::PointCloud<PointType>());

    if (!job.current_cloud || !job.history_cloud || job.current_cloud->empty() || job.history_cloud->empty()) {
      result.reject_reason = "empty_cloud";
      return result;
    }

    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(icp_max_correspondence_distance_);
    icp.setMaximumIterations(icp_max_iterations_);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);
    icp.setInputSource(job.current_cloud);
    icp.setInputTarget(job.history_cloud);
    icp.align(*result.registered_cloud);

    result.fitness = icp.getFitnessScore();
    if (!icp.hasConverged()) {
      result.reject_reason = "not_converged";
      return result;
    }
    if (result.fitness > loop_fitness_score_threshold_) {
      result.reject_reason = "fitness";
      return result;
    }

    Eigen::Affine3f correction;
    correction.matrix() = icp.getFinalTransformation();
    float cx = 0.0f;
    float cy = 0.0f;
    float cz = 0.0f;
    float croll = 0.0f;
    float cpitch = 0.0f;
    float cyaw = 0.0f;
    pcl::getTranslationAndEulerAngles(correction, cx, cy, cz, croll, cpitch, cyaw);
    result.correction_translation = std::sqrt(
      static_cast<double>(cx) * static_cast<double>(cx) +
      static_cast<double>(cy) * static_cast<double>(cy) +
      static_cast<double>(cz) * static_cast<double>(cz));
    result.correction_yaw_deg = std::abs(static_cast<double>(cyaw)) * 180.0 / M_PI;
    if (max_loop_correction_translation_ > 0.0 &&
      result.correction_translation > max_loop_correction_translation_)
    {
      result.reject_reason = "correction_translation";
      return result;
    }
    Eigen::Affine3f current_affine = pose_to_affine(job.current_pose);
    Eigen::Affine3f corrected_affine = correction * current_affine;

    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float roll = 0.0f;
    float pitch = 0.0f;
    float yaw = 0.0f;
    pcl::getTranslationAndEulerAngles(corrected_affine, x, y, z, roll, pitch, yaw);

    Pose6D corrected_pose;
    corrected_pose.x = x;
    corrected_pose.y = y;
    corrected_pose.z = z;
    corrected_pose.roll = roll;
    corrected_pose.pitch = pitch;
    corrected_pose.yaw = yaw;

    result.relative_pose = pose_to_gtsam(corrected_pose).between(pose_to_gtsam(job.history_pose));
    result.accepted = true;
    result.reject_reason = "accepted";
    return result;
  }

  void process_icp_queue()
  {
    IcpJob job;
    bool have_job = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!loop_icp_queue_.empty()) {
        const auto loop_pair = loop_icp_queue_.front();
        loop_icp_queue_.pop_front();
        job = make_icp_job_locked(loop_pair.first, loop_pair.second);
        have_job = true;
      }
    }

    if (!have_job) {
      return;
    }

    IcpResult result = do_icp(job);

    std::lock_guard<std::mutex> lock(mutex_);
    pending_loop_index_container_.erase(job.current_idx);

    if (!result.accepted || accepted_loop_index_container_.count(job.current_idx) > 0) {
      return;
    }

    graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
      job.current_idx,
      job.history_idx,
      result.relative_pose,
      robust_loop_noise_));
    accepted_loop_index_container_[job.current_idx] = job.history_idx;

    run_isam2_locked();
    publish_optimized_keyframes_locked();
    if (!frontend_feedback_enable_) {
      publish_map_to_odom_locked();
    }
  }

  builtin_interfaces::msg::Time node_now_msg() const
  {
    const int64_t now_ns = get_clock()->now().nanoseconds();
    builtin_interfaces::msg::Time stamp;
    stamp.sec = static_cast<int32_t>(now_ns / 1000000000LL);
    stamp.nanosec = static_cast<uint32_t>(now_ns % 1000000000LL);
    return stamp;
  }

  void publish_optimized_keyframes_locked()
  {
    if (!pub_optimized_keyframes_ || optimized_poses_.size() != keyframes_.size()) {
      return;
    }

    fast_lio::msg::OptimizedKeyframes msg;
    msg.header.frame_id = map_frame_;
    if (keyframes_.empty()) {
      msg.header.stamp = node_now_msg();
    } else {
      msg.header.stamp = keyframes_.back().stamp;
    }
    msg.scan_ids.reserve(keyframes_.size());
    msg.poses_in_map.reserve(optimized_poses_.size());
    for (std::size_t idx = 0; idx < keyframes_.size(); ++idx) {
      msg.scan_ids.push_back(keyframes_[idx].scan_id);
      msg.poses_in_map.push_back(pose_to_msg(optimized_poses_[idx]));
    }
    pub_optimized_keyframes_->publish(msg);
  }

  void publish_map_to_odom_identity()
  {
    if (!pub_map_to_odom_) {
      return;
    }

    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = node_now_msg();
    tf_msg.header.frame_id = map_frame_;
    tf_msg.child_frame_id = odom_frame_;
    tf_msg.transform.translation.x = 0.0;
    tf_msg.transform.translation.y = 0.0;
    tf_msg.transform.translation.z = 0.0;
    tf_msg.transform.rotation.x = 0.0;
    tf_msg.transform.rotation.y = 0.0;
    tf_msg.transform.rotation.z = 0.0;
    tf_msg.transform.rotation.w = 1.0;
    pub_map_to_odom_->publish(tf_msg);
  }

  void publish_map_to_odom_locked()
  {
    if (!pub_map_to_odom_) {
      return;
    }

    if (keyframes_.empty() || optimized_poses_.size() != keyframes_.size()) {
      publish_map_to_odom_identity();
      return;
    }

    const Eigen::Affine3f raw_pose = pose_to_affine(keyframes_.back().raw_pose);
    const Eigen::Affine3f optimized_pose = pose_to_affine(optimized_poses_.back());
    const Eigen::Affine3f map_to_odom = optimized_pose * raw_pose.inverse();

    Eigen::Quaternionf quat(map_to_odom.rotation());
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = keyframes_.back().stamp;
    tf_msg.header.frame_id = map_frame_;
    tf_msg.child_frame_id = odom_frame_;
    tf_msg.transform.translation.x = map_to_odom.translation().x();
    tf_msg.transform.translation.y = map_to_odom.translation().y();
    tf_msg.transform.translation.z = map_to_odom.translation().z();
    tf_msg.transform.rotation.x = quat.x();
    tf_msg.transform.rotation.y = quat.y();
    tf_msg.transform.rotation.z = quat.z();
    tf_msg.transform.rotation.w = quat.w();
    pub_map_to_odom_->publish(tf_msg);
  }

  bool enabled_ = false;
  bool frontend_feedback_enable_ = false;
  std::string map_frame_;
  std::string frontend_frame_topic_;
  std::string keyframe_id_topic_;
  std::string optimized_keyframes_topic_;
  std::string map_to_odom_topic_;
  std::string odom_frame_;

  double keyframe_meter_gap_ = 1.0;
  double keyframe_deg_gap_ = 10.0;
  double keyframe_rad_gap_ = 10.0 * M_PI / 180.0;
  double history_keyframe_search_radius_ = 5.0;
  double history_keyframe_search_time_diff_ = 30.0;
  int history_keyframe_search_num_ = 10;
  double loop_closure_frequency_ = 2.0;
  double icp_process_frequency_ = 1.0;
  double keyframe_downsample_leaf_ = 0.4;
  double map_downsample_leaf_ = 0.4;
  double icp_max_correspondence_distance_ = 15.0;
  int icp_max_iterations_ = 100;
  double loop_fitness_score_threshold_ = 0.3;
  double max_loop_correction_translation_ = 1.5;
  int graph_update_times_ = 2;
  double odom_noise_rotation_ = 1e-6;
  double odom_noise_translation_ = 1e-4;
  double loop_noise_score_ = 0.5;

  std::mutex mutex_;
  std::vector<KeyframeData> keyframes_;
  std::vector<Pose6D> optimized_poses_;
  std::map<std::size_t, std::size_t> pending_loop_index_container_;
  std::map<std::size_t, std::size_t> accepted_loop_index_container_;
  std::deque<std::pair<std::size_t, std::size_t>> loop_icp_queue_;

  bool have_prev_frame_ = false;
  bool posegraph_initialized_ = false;
  Pose6D prev_frame_pose_;
  double translation_accumulated_ = 0.0;
  double rotation_accumulated_ = 0.0;

  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values initial_estimate_;
  gtsam::Values current_estimate_;
  std::unique_ptr<gtsam::ISAM2> isam_;
  gtsam::noiseModel::Diagonal::shared_ptr prior_noise_;
  gtsam::noiseModel::Diagonal::shared_ptr odom_noise_;
  gtsam::noiseModel::Base::shared_ptr robust_loop_noise_;

  rclcpp::Subscription<fast_lio::msg::FrontendFrame>::SharedPtr sub_frontend_frame_;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr pub_keyframe_id_;
  rclcpp::Publisher<fast_lio::msg::OptimizedKeyframes>::SharedPtr pub_optimized_keyframes_;
  rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr pub_map_to_odom_;

  rclcpp::TimerBase::SharedPtr loop_timer_;
  rclcpp::TimerBase::SharedPtr icp_timer_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LoopClosureBackendNode>());
  rclcpp::shutdown();
  return 0;
}
