#pragma once

#ifdef FASTLIO_USE_CUDA

#include <memory>
#include <Eigen/Core>
#include "preprocess.h"

namespace fastlio
{
namespace gpu
{
class PointTransformer
{
public:
  PointTransformer();
  ~PointTransformer();

  PointTransformer(const PointTransformer &) = delete;
  PointTransformer &operator=(const PointTransformer &) = delete;

  bool available() const;

  bool transform(const PointCloudXYZI &input,
                 PointCloudXYZI &output,
                 const Eigen::Matrix3f &rotation,
                 const Eigen::Vector3f &translation);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gpu
}  // namespace fastlio

#endif  // FASTLIO_USE_CUDA
