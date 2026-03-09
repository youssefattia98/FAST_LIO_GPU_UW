#include "gpu/point_transform_gpu.hpp"

#ifdef FASTLIO_USE_CUDA

#include "gpu/fastlio_cuda_utils.hpp"

#include <cuda_runtime.h>
#include <rclcpp/rclcpp.hpp>

#include <vector>
#include <stdexcept>

namespace fastlio
{
namespace gpu
{
namespace
{
struct PackedPoint
{
  float x;
  float y;
  float z;
  float intensity;
  float curvature;
};

__global__ void transform_points_kernel(const PackedPoint *input,
                                        PackedPoint *output,
                                        size_t total_points,
                                        const float *rotation,
                                        const float *translation)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_points)
  {
    return;
  }

  const PackedPoint p = input[idx];
  PackedPoint out;
  out.x = rotation[0] * p.x + rotation[1] * p.y + rotation[2] * p.z + translation[0];
  out.y = rotation[3] * p.x + rotation[4] * p.y + rotation[5] * p.z + translation[1];
  out.z = rotation[6] * p.x + rotation[7] * p.y + rotation[8] * p.z + translation[2];
  out.intensity = p.intensity;
  out.curvature = p.curvature;
  output[idx] = out;
}
}  // namespace

struct PointTransformer::Impl
{
  Impl()
  {
    auto err = cudaStreamCreate(&stream_);
    available_ = (err == cudaSuccess);
    if (!available_)
    {
      RCLCPP_ERROR(rclcpp::get_logger("fast_lio.gpu"), "Failed to create CUDA stream for point transformer: %s", cudaGetErrorString(err));
    }
  }

  ~Impl()
  {
    release();
    if (stream_ != nullptr)
    {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  bool available() const { return available_; }

  bool transform(const PointCloudXYZI &input,
                 PointCloudXYZI &output,
                 const Eigen::Matrix3f &rotation,
                 const Eigen::Vector3f &translation)
  {
    if (!available_)
    {
      return false;
    }

    const size_t total_points = input.size();
    if (total_points == 0)
    {
      output.clear();
      return true;
    }

    try
    {
      ensure_capacity(total_points);
    }
    catch (const std::exception &ex)
    {
      RCLCPP_ERROR(rclcpp::get_logger("fast_lio.gpu"), "Point transformer CUDA allocation failed: %s", ex.what());
      available_ = false;
      return false;
    }

    host_input_.resize(total_points);
    for (size_t i = 0; i < total_points; ++i)
    {
      const PointType &pt = input.points[i];
      host_input_[i] = PackedPoint{pt.x, pt.y, pt.z, pt.intensity, pt.curvature};
    }

    FASTLIO_CUDA_CHECK(cudaMemcpyAsync(d_input_, host_input_.data(), total_points * sizeof(PackedPoint), cudaMemcpyHostToDevice, stream_));

    float h_rotation[9] = {
      rotation(0, 0), rotation(0, 1), rotation(0, 2),
      rotation(1, 0), rotation(1, 1), rotation(1, 2),
      rotation(2, 0), rotation(2, 1), rotation(2, 2)};
    float h_translation[3] = {translation(0), translation(1), translation(2)};

    FASTLIO_CUDA_CHECK(cudaMemcpyAsync(d_rotation_, h_rotation, sizeof(h_rotation), cudaMemcpyHostToDevice, stream_));
    FASTLIO_CUDA_CHECK(cudaMemcpyAsync(d_translation_, h_translation, sizeof(h_translation), cudaMemcpyHostToDevice, stream_));

    const int threads = 256;
    const int blocks = static_cast<int>((total_points + threads - 1) / threads);
    transform_points_kernel<<<blocks, threads, 0, stream_>>>(d_input_, d_output_, total_points, d_rotation_, d_translation_);
    FASTLIO_CUDA_CHECK(cudaGetLastError());

    host_output_.resize(total_points);
    FASTLIO_CUDA_CHECK(cudaMemcpyAsync(host_output_.data(), d_output_, total_points * sizeof(PackedPoint), cudaMemcpyDeviceToHost, stream_));
    FASTLIO_CUDA_CHECK(cudaStreamSynchronize(stream_));

    output.clear();
    output.reserve(total_points);
    for (size_t i = 0; i < total_points; ++i)
    {
      const PackedPoint &pt = host_output_[i];
      PointType out_pt;
      out_pt.x = pt.x;
      out_pt.y = pt.y;
      out_pt.z = pt.z;
      out_pt.intensity = pt.intensity;
      out_pt.curvature = pt.curvature;
      out_pt.normal_x = 0.0f;
      out_pt.normal_y = 0.0f;
      out_pt.normal_z = 0.0f;
      output.push_back(out_pt);
    }

    output.width = static_cast<uint32_t>(output.size());
    output.height = 1;
    output.is_dense = true;
    return true;
  }

private:
  void ensure_capacity(size_t desired)
  {
    if (desired <= capacity_)
    {
      return;
    }
    release();
    capacity_ = desired;
    FASTLIO_CUDA_CHECK(cudaMalloc(&d_input_, capacity_ * sizeof(PackedPoint)));
    FASTLIO_CUDA_CHECK(cudaMalloc(&d_output_, capacity_ * sizeof(PackedPoint)));
    FASTLIO_CUDA_CHECK(cudaMalloc(&d_rotation_, 9 * sizeof(float)));
    FASTLIO_CUDA_CHECK(cudaMalloc(&d_translation_, 3 * sizeof(float)));
  }

  void release()
  {
    if (d_input_)
    {
      cudaFree(d_input_);
      d_input_ = nullptr;
    }
    if (d_output_)
    {
      cudaFree(d_output_);
      d_output_ = nullptr;
    }
    if (d_rotation_)
    {
      cudaFree(d_rotation_);
      d_rotation_ = nullptr;
    }
    if (d_translation_)
    {
      cudaFree(d_translation_);
      d_translation_ = nullptr;
    }
    capacity_ = 0;
  }

  cudaStream_t stream_{nullptr};
  bool available_{true};
  size_t capacity_{0};

  PackedPoint *d_input_{nullptr};
  PackedPoint *d_output_{nullptr};
  float *d_rotation_{nullptr};
  float *d_translation_{nullptr};

  std::vector<PackedPoint> host_input_;
  std::vector<PackedPoint> host_output_;
};

PointTransformer::PointTransformer() : impl_(std::make_unique<Impl>()) {}
PointTransformer::~PointTransformer() = default;

bool PointTransformer::available() const
{
  return impl_ && impl_->available();
}

bool PointTransformer::transform(const PointCloudXYZI &input,
                                 PointCloudXYZI &output,
                                 const Eigen::Matrix3f &rotation,
                                 const Eigen::Vector3f &translation)
{
  if (!impl_)
  {
    return false;
  }
  return impl_->transform(input, output, rotation, translation);
}

}  // namespace gpu
}  // namespace fastlio

#endif  // FASTLIO_USE_CUDA
