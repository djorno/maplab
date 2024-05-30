#ifndef CERES_ERROR_TERMS_BALM_ERROR_TERM_H_
#define CERES_ERROR_TERMS_BALM_ERROR_TERM_H_

#include <vector>

#include <Eigen/Core>
#include <ceres/cost_function.h>
#include <glog/logging.h>
// #include <vi-map/point-cluster.h>
#include <ceres-error-terms/balm-voxhess.h>

#include <ceres-error-terms/common.h>

namespace ceres_error_terms {
// Note: this error term accepts rotations expressed as quaternions
// in JPL convention [x, y, z, w]. This convention corresponds to the internal
// coefficient storage of Eigen so you can directly pass pointer to your
// Eigen quaternion data, e.g. your_eigen_quaternion.coeffs().data().

class BALMErrorTerm : public ceres::CostFunction {
 public:
  BALMErrorTerm(
      const VoxHess& voxhess, const aslam::Transformation& T_I_S,
      const aslam::Transformation& T_G_M,
      const std::vector<std::vector<std::pair<size_t, size_t>>> feature_indices)
      : feature_indices_(feature_indices),
        num_poses_(feature_indices.size()),
        T_I_S_(T_I_S),
        T_G_M_(T_G_M),
        num_features_(voxhess.plvec_voxels.size()) {
    // for ceres::CostFunction
    set_num_residuals(balmblocks::kResidualSize);
    std::vector<int32_t>* parameter_block_sizes =
        mutable_parameter_block_sizes();
    parameter_block_sizes->resize(num_poses_, balmblocks::kPoseSize);

    lmbd_.resize(num_features_);
    U_.resize(num_features_);
    uk_.resize(num_features_);
    sig_transformed_.resize(num_features_);
    for (size_t i = 0; i < num_features_; i++) {
      voxhess_atoms_.push_back(VoxHessAtom(voxhess, i));
    }
    LOG(INFO) << "num features in BEC: " << num_features_;
  }

  virtual ~BALMErrorTerm() {}

  virtual bool Evaluate(
      double const* const* parameters, double* residuals_ptr,
      double** jacobians) const;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  const size_t get_num_features() const {
    return num_features_;
  }

  const double get_num_obs_for_feature(const size_t feat_ind) const {
    return voxhess_atoms_[feat_ind].index.size();
  }

  const std::vector<size_t>& get_index(const size_t feat_ind) const {
    return voxhess_atoms_[feat_ind].index;
  }

  const PointCluster& get_sig(const size_t feat_ind) const {
    return voxhess_atoms_[feat_ind].sig;
  }

  const std::vector<PointCluster>& get_sig_origin(const size_t feat_ind) const {
    return voxhess_atoms_[feat_ind].sig_origin;
  }

  const double get_coeff(const size_t feat_ind) const {
    return voxhess_atoms_[feat_ind].coeff;
  }

  const Eigen::Vector3d get_uk(const size_t feat_ind) const {
    return uk_[feat_ind];
  }

  const Eigen::Vector3d get_vBar(const size_t feat_ind) const {
    return sig_transformed_[feat_ind].v / sig_transformed_[feat_ind].N;
  }

  const int get_NN(const size_t feat_ind) const {
    return sig_transformed_[feat_ind].N;
  }

 private:
  enum { kIdxPose };
  const std::vector<std::vector<std::pair<size_t, size_t>>> feature_indices_;
  std::vector<VoxHessAtom> voxhess_atoms_;
  mutable std::vector<PointCluster> sig_transformed_;
  const size_t num_poses_;
  const size_t num_features_;
  const aslam::Transformation T_I_S_;
  const aslam::Transformation T_G_M_;
  mutable std::vector<Eigen::Vector3d> lmbd_;
  mutable std::vector<Eigen::Matrix3d> U_;
  mutable std::vector<Eigen::Vector3d> uk_;
  const double sigma_inv = 1.0 / 0.8;  // taken from VisualReprojectionError

  // The representation for Jacobian computed by this object.
  typedef Eigen::Matrix<
      double, balmblocks::kResidualSize, balmblocks::kPoseSize, Eigen::RowMajor>
      PoseJacobian;
};

}  // namespace ceres_error_terms

#include "ceres-error-terms/balm-error-term-inl.h"

#endif  // CERES_ERROR_TERMS_INERTIAL_ERROR_TERM_H_