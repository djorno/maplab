#ifndef CERES_ERROR_TERMS_BALM_ERROR_TERM_H_
#define CERES_ERROR_TERMS_BALM_ERROR_TERM_H_

#include <vector>

#include <Eigen/Core>
#include <ceres/sized_cost_function.h>
#include <glog/logging.h>
#include <vi-map/point-cluster.h>

#include <ceres-error-terms/common.h>

namespace ceres_error_terms {
// Note: this error term accepts rotations expressed as quaternions
// in JPL convention [x, y, z, w]. This convention corresponds to the internal
// coefficient storage of Eigen so you can directly pass pointer to your
// Eigen quaternion data, e.g. your_eigen_quaternion.coeffs().data().
class BALMErrorTerm : public ceres::SizedCostFunction<
                          balmblocks::kResidualSize, balmblocks::kPoseSize> {
 public:
  BALMErrorTerm(
      vi_map::PointCluster point_cluster, const aslam::Transformation& T_G_M,
      const aslam::Transformation& T_I_S)
      : point_cluster_(point_cluster), T_G_M_(T_G_M), T_I_S_(T_I_S) {
    CHECK_GE(point_cluster_.N, 0);
    CHECK(point_cluster_.P.allFinite());
    CHECK(point_cluster_.v.allFinite());
    CHECK(point_cluster_.P.isApprox(point_cluster.P.transpose()));
  }

  virtual ~BALMErrorTerm() {}

  virtual bool Evaluate(
      double const* const* parameters, double* residuals_ptr,
      double** jacobians) const;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  enum { kIdxPose };
  mutable vi_map::PointCluster point_cluster_;
  const aslam::Transformation T_G_M_;
  const aslam::Transformation T_I_S_;
};

}  // namespace ceres_error_terms

#endif  // CERES_ERROR_TERMS_INERTIAL_ERROR_TERM_H_
