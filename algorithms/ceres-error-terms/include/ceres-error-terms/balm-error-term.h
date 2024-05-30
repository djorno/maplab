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
    residual_.resize(num_features_);
    uk_.resize(num_features_);
    sig_transformed_.resize(num_features_);

    // subdivide voxhess into atoms
    for (size_t i = 0; i < num_features_; i++) {
      voxhess_atoms_.push_back(VoxHessAtom(voxhess, i));
    }
    LOG(INFO) << "num features in BEC: " << num_features_;
  }

  void PrepareForEvaluation(
      bool evaluate_jacobians, bool new_evaluation_point) final {
    if (new_evaluation_point) {
      for (size_t i = 0; i < num_features_; i++) {
        double res = voxhess_atoms_[i].evaluate_residual(
            xs_, sig_transformed_[i], lmbd_[i], U_[i], T_I_S_, T_G_M_);
        residual_[i] = res;
        uk_[i] = U_[i].col(0);
      }
    }

    LOG(INFO) << "Total residual sum: "
              << std::accumulate(residual_.begin(), residual_.end(), 0.0);
  }

  const double get_residual(const size_t feat_ind) const {
    return residual_[feat_ind];
  }

  const size_t get_num_features() const {
    return num_features_;
  }

  const double get_total_residual() const {
    return std::accumulate(residual_.begin(), residual_.end(), 0.0);
  }

  const double get_num_obs_for_feature(const size_t feat_ind) const {
    return voxhess_atoms_[feat_ind].index.size();
  }

  const std::vector<double> get_accum_res_for_features(
      const std::vector<std::pair<size_t, size_t>>& feature_index) const {
    std::vector<double> res_vec;
    for (const auto& pair : feature_index) {
      size_t feat_ind = pair.first;
      double num_obs = get_num_obs_for_feature(feat_ind);
      CHECK_GT(num_obs, 0.0) << "num_obs is 0";
      res_vec.push_back(residual_[feat_ind] / num_obs);
    }
    return res_vec;
  }

  const aslam::Transformation get_T_I_S() const {
    return T_I_S_;
  }
  const aslam::Transformation get_T_G_M() const {
    return T_G_M_;
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
  const size_t num_features_;
  std::vector<double> residual_;
  std::vector<VoxHessAtom> voxhess_atoms_;
  std::vector<PointCluster> sig_transformed_;
  const std::vector<double*> xs_;
  std::vector<Eigen::Vector3d> lmbd_;
  std::vector<Eigen::Matrix3d> U_;
  std::vector<Eigen::Vector3d> uk_;
  const aslam::Transformation T_I_S_;
  const aslam::Transformation T_G_M_;
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
        residual_size_(feature_index.size()) {
    CHECK_NOTNULL(evaluation_callback.get());
    set_num_residuals(residual_size_);
    std::vector<int32_t>* parameter_block_sizes =
        mutable_parameter_block_sizes();
    parameter_block_sizes->resize(1, balmblocks::kPoseSize);
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