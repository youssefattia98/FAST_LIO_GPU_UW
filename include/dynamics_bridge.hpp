#pragma once

#include <Eigen/Dense>
#include <memory>

namespace mvm {
class UnderwaterVehicleModel;
}

namespace fastlio::dynamics {

void set_model(const std::shared_ptr<mvm::UnderwaterVehicleModel>& model);
bool has_model();
void set_thruster_forces(const Eigen::VectorXd& forces);
bool compute_body_accel(const Eigen::Matrix<double, 6, 1>& vel_body,
                        const Eigen::Matrix<double, 6, 1>& pose_world,
                        Eigen::Matrix<double, 6, 1>& accel_body);

} // namespace fastlio::dynamics
