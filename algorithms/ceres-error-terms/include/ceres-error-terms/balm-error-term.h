#ifndef CERES_ERROR_TERMS_BALM_ERROR_TERM_H_
#define CERES_ERROR_TERMS_BALM_ERROR_TERM_H_

#include <vector>

#include <Eigen/Core>
#include <ceres-error-terms/balm-voxhess.h>
#include <ceres/cost_function.h>
#include <glog/logging.h>

#include <ceres-error-terms/common.h>

namespace ceres_error_terms {
// Note: this error term accepts rotations expressed as quaternions
// in JPL convention [x, y, z, w]. This convention corresponds to the internal
// coefficient storage of Eigen so you can directly pass pointer to your
// Eigen quaternion data, e.g. your_eigen_quaternion.coeffs().data().
class BALMEvaluationCallback : public ceres::EvaluationCallback {
 public:
  BALMEvaluationCallback(
      const VoxHess& voxhess, const std::vector<double*> xs,
      const aslam::Transformation T_I_S, const aslam::Transformation T_G_M)
      : xs_(xs),
        T_I_S_(T_I_S),
        T_G_M_(T_G_M),
        num_features_(voxhess.plvec_voxels.size()) {
    lmbd_.resize(num_features_);
    U_.resize(num_features_);
    sig_transformed_.resize(num_features_);
    feature_planes_.resize(num_features_);

    // subdivide voxhess into atoms
    for (size_t i = 0; i < num_features_; i++) {
      voxhess_atoms_.push_back(VoxHessAtom(voxhess, i));
      voxhess_atoms_[i].generate_original_planes();
    }

    LOG(INFO) << "num features in BEC: " << num_features_;
  }

  void PrepareForEvaluation(
      bool evaluate_jacobians, bool new_evaluation_point) {
    if (new_evaluation_point) {
      double balm_residual = 0.0;
      for (size_t i = 0; i < num_features_; i++) {
        voxhess_atoms_[i].evaluate_plane(
            xs_, feature_planes_[i], T_I_S_, T_G_M_);
        // Optional evaluation of the original balm residual. not needed for the
        // optimization
        balm_residual += voxhess_atoms_[i].evaluate_residual(
            xs_, sig_transformed_[i], lmbd_[i], U_[i], T_I_S_, T_G_M_);
      }
      LOG(INFO) << "BALM Residual: " << balm_residual;
    }
    evaluated_ = true;
  }

  bool prepared_for_evaluation() const {
    return evaluated_;
  }

  const size_t get_num_features() const {
    return num_features_;
  }

  const double get_num_obs_for_feature(const size_t feat_ind) const {
    return voxhess_atoms_[feat_ind].index.size();
  }

  const aslam::Transformation& get_T_I_S() const {
    return T_I_S_;
  }
  const aslam::Transformation& get_T_G_M() const {
    return T_G_M_;
  }

  const std::vector<PointCluster>& get_sig_origin(const size_t feat_ind) const {
    return voxhess_atoms_[feat_ind].sig_origin;
  }

  const Eigen::Vector3d get_vBar(const size_t feat_ind) const {
    return feature_planes_[feat_ind].p;
  }

  const std::vector<Eigen::Vector3d> get_vBar_i(
      const std::vector<std::pair<size_t, size_t>> feature_index) const {
    std::vector<Eigen::Vector3d> vBar;
    vBar.reserve(feature_index.size());
    for (const auto& pair : feature_index) {
      size_t feat_ind = pair.first;
      vBar.push_back(get_vBar(feat_ind));
    }
    return vBar;
  }

  const std::vector<BALMPlane> get_feature_planes(
      const std::vector<std::pair<size_t, size_t>> feature_index) const {
    std::vector<BALMPlane> planes;
    planes.reserve(feature_index.size());
    for (const auto& pair : feature_index) {
      size_t feat_ind = pair.first;
      size_t pose_ind = pair.second;
      planes.push_back(feature_planes_[feat_ind]);
    }
    return planes;
  }

  std::vector<BALMPlane> get_original_planes_ij(
      const std::vector<std::pair<size_t, size_t>>& feature_index) const {
    std::vector<BALMPlane> planes;
    planes.reserve(feature_index.size());
    for (const auto& pair : feature_index) {
      size_t feat_ind = pair.first;
      size_t pose_ind = pair.second;
      BALMPlane plane;
      plane = voxhess_atoms_[feat_ind].original_planes[pose_ind];
      CHECK(std::abs(plane.n.norm() - 1.0) <= 1e-6)
          << "plane.n.norm(): " << plane.n.norm();
      planes.push_back(plane);
    }
    return planes;
  }

  const double get_N_i(const size_t feat_ind) const {
    return sig_transformed_[feat_ind].N;
  }

  const double get_N_ij(const size_t feat_ind, const size_t pose_ind) const {
    return voxhess_atoms_[feat_ind].sig_origin[pose_ind].N;
  }

 private:
  const size_t num_features_;
  std::vector<VoxHessAtom> voxhess_atoms_;
  std::vector<PointCluster> sig_transformed_;
  const std::vector<double*> xs_;
  std::vector<Eigen::Vector3d> lmbd_;
  std::vector<Eigen::Matrix3d> U_;
  const aslam::Transformation T_I_S_;
  const aslam::Transformation T_G_M_;
  std::vector<BALMPlane> feature_planes_;
  bool evaluated_ = false;
};

class BALMErrorTerm : public ceres::CostFunction {
 public:
  BALMErrorTerm(
      const std::shared_ptr<BALMEvaluationCallback> evaluation_callback,
      const size_t i,
      const std::vector<std::pair<size_t, size_t>> feature_index)
      : i_(i),
        evaluation_callback_(evaluation_callback),
        feature_index_(feature_index),
        T_I_S_(evaluation_callback->get_T_I_S()),
        T_G_M_(evaluation_callback->get_T_G_M()),
        residual_size_(feature_index.size() * balmblocks::kResidualSize) {
    CHECK_NOTNULL(evaluation_callback.get());
    // for ceres::CostFunction
    set_num_residuals(residual_size_);
    std::vector<int32_t>* parameter_block_sizes =
        mutable_parameter_block_sizes();
    parameter_block_sizes->resize(1, balmblocks::kPoseSize);
    //////////////////////////
    // get the original observed planes in local lidar frame
    original_planes_ij =
        evaluation_callback_->get_original_planes_ij(feature_index_);
    CHECK(original_planes_ij.size() == feature_index.size())
        << "size mismatch" << original_planes_ij.size() << " "
        << feature_index.size();
  }

  virtual ~BALMErrorTerm() {}

  virtual bool Evaluate(
      double const* const* parameters, double* residuals_ptr,
      double** jacobians) const;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  enum { kIdxPose };
  const std::shared_ptr<BALMEvaluationCallback> evaluation_callback_;
  const std::vector<std::pair<size_t, size_t>> feature_index_;
  std::vector<BALMPlane> original_planes_ij;
  const size_t i_;
  const int residual_size_;
  const aslam::Transformation T_I_S_;
  const aslam::Transformation T_G_M_;
  const double sigma_inv = 1.0 / 0.8;  // taken from VisualReprojectionError

  // The representation for Jacobian computed by this object.
  typedef Eigen::Matrix<
      double, Eigen::Dynamic, balmblocks::kPoseSize, Eigen::RowMajor>
      PoseJacobian;
};

}  // namespace ceres_error_terms

#include "ceres-error-terms/balm-error-term-inl.h"

#endif  // CERES_ERROR_TERMS_INERTIAL_ERROR_TERM_H_