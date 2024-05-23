#ifndef CERES_ERROR_TERMS_BALM_ERROR_TERM_H_
#define CERES_ERROR_TERMS_BALM_ERROR_TERM_H_

#include <vector>

#include <Eigen/Core>
#include <ceres/sized_cost_function.h>
#include <glog/logging.h>
// #include <vi-map/point-cluster.h>
#include <ceres-error-terms/balm-voxhess.h>

#include <ceres-error-terms/common.h>

namespace ceres_error_terms {
// Note: this error term accepts rotations expressed as quaternions
// in JPL convention [x, y, z, w]. This convention corresponds to the internal
// coefficient storage of Eigen so you can directly pass pointer to your
// Eigen quaternion data, e.g. your_eigen_quaternion.coeffs().data().
class BALMEvaluationCallback : public ceres::EvaluationCallback {
 public:
  BALMEvaluationCallback(
      const VoxHess& voxhess, const std::vector<double*>& xs,
      const aslam::Transformation T_I_S, const aslam::Transformation T_G_M)
      : xs_(xs), T_I_S_(T_I_S), T_G_M_(T_G_M) {
    size_t num_features = voxhess.plvec_voxels.size();
    lmbd_.reserve(num_features);
    U_.reserve(num_features);
    residual_.reserve(num_features);
    uk_.reserve(num_features);

    // subdivide voxhess into atoms
    for (size_t i = 0; i < num_features; i++) {
      voxhess_atoms_.push_back(VoxHessAtom(voxhess, i));
    }
  }

  void PrepareForEvaluation(
      bool evaluate_jacobians, bool new_evaluation_point) final {
    if (new_evaluation_point) {
      for (size_t i = 0; i < voxhess_atoms_.size(); i++) {
        residual_[i] = voxhess_atoms_[i].evaluate_residual(
            xs_, sig_transformed_[i], lmbd_[i], U_[i], T_I_S_, T_G_M_);
        uk_[i] = U_[i].col(0);
      }
    }
  }

  const double residual(const size_t feat_ind) const {
    return residual_[feat_ind];
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
  std::vector<double> residual_;
  std::vector<VoxHessAtom> voxhess_atoms_;
  std::vector<PointCluster> sig_transformed_;
  const std::vector<double*>& xs_;
  std::vector<Eigen::Vector3d> lmbd_;
  std::vector<Eigen::Matrix3d> U_;
  std::vector<Eigen::Vector3d> uk_;
  const aslam::Transformation T_I_S_;
  const aslam::Transformation T_G_M_;
};

class BALMErrorTerm : public ceres::SizedCostFunction<
                          balmblocks::kResidualSize, balmblocks::kPoseSize> {
 public:
  BALMErrorTerm(
      const std::shared_ptr<BALMEvaluationCallback>& evaluation_callback,
      const size_t sig_i, const size_t feat_num)
      : sig_i_(sig_i),
        feat_num_(feat_num),
        evaluation_callback_(evaluation_callback),
        T_I_S_(evaluation_callback->get_T_I_S()),
        T_G_M_(evaluation_callback->get_T_G_M()) {}

  virtual ~BALMErrorTerm() {}

  virtual bool Evaluate(
      double const* const* parameters, double* residuals_ptr,
      double** jacobians) const;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  enum { kIdxPose };
  const std::shared_ptr<BALMEvaluationCallback>& evaluation_callback_;
  const size_t sig_i_;
  const size_t feat_num_;
  const aslam::Transformation T_I_S_;
  const aslam::Transformation T_G_M_;

  // The representation for Jacobian computed by this object.
  typedef Eigen::Matrix<
      double, balmblocks::kResidualSize, balmblocks::kPoseSize, Eigen::RowMajor>
      PoseJacobian;
};

}  // namespace ceres_error_terms

#endif  // CERES_ERROR_TERMS_INERTIAL_ERROR_TERM_H_