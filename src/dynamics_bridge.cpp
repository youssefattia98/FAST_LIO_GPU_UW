#include "dynamics_bridge.hpp"

#include <mutex>
#include <iostream>
#include "underwater_vehicle_model.hpp"

namespace fastlio::dynamics {

namespace {
std::mutex g_mutex;
std::shared_ptr<mvm::UnderwaterVehicleModel> g_model;
Eigen::VectorXd g_thruster_forces;
}

void set_model(const std::shared_ptr<mvm::UnderwaterVehicleModel>& model)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  g_model = model;
}

bool has_model()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  return static_cast<bool>(g_model);
}

void set_thruster_forces(const Eigen::VectorXd& forces)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  g_thruster_forces = forces;
}

bool compute_body_accel(const Eigen::Matrix<double, 6, 1>& vel_body,
                        const Eigen::Matrix<double, 6, 1>& pose_world,
                        Eigen::Matrix<double, 6, 1>& accel_body)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_model)
  {
    return false;
  }

  g_model->UpdateModel(vel_body, pose_world);

  Eigen::VectorXd forces = g_thruster_forces;
  if (forces.size() == 0)
  {
    forces = Eigen::VectorXd::Zero(g_model->GetNumThrusters());
  }
  else if (static_cast<size_t>(forces.size()) != g_model->GetNumThrusters())
  {
    std::cerr << "[fast_lio] Thruster forces size mismatch: got " << forces.size()
              << ", expected " << g_model->GetNumThrusters() << ". Using zeros." << std::endl;
    forces = Eigen::VectorXd::Zero(g_model->GetNumThrusters());
  }

  accel_body = g_model->ComputeAcceleration(forces);
  return true;
}

} // namespace fastlio::dynamics
