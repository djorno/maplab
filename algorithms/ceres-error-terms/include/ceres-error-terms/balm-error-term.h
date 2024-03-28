#ifndef CERES_ERROR_TERMS_BALM_ERROR_TERM_H_
#define CERES_ERROR_TERMS_BALM_ERROR_TERM_H_

#include <numeric>
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

// This implements an evaluation callback to pre-compute the residuals for every
// BALM feature at once.
// TODO(gtonetti): Ceres only supports a single evaluation callback. Maplab
// should therefore support a general MaplabEvaluationCallback object that owns
// each evaluation callback that a user wants to add
class BALMEvaluationCallback : public ceres::EvaluationCallback {
 public:
  BALMEvaluationCallback(
      VoxHess&& voxhess, const std::vector<double*> poses_M_I,
      const aslam::Transformation T_I_S, const aslam::Transformation T_G_M)
      : num_features_(voxhess.getNumFeatures()),
        voxhess_(std::move(voxhess)),
        xs_(poses_M_I),
        T_I_S_(T_I_S),
        T_G_M_(T_G_M) {
    // mostly legacy code from original balm residual calculation
    lmbd_.resize(num_features_);
    U_.resize(num_features_);
    cluster_G.resize(num_features_);
    feature_planes_.resize(num_features_);

    // subdivide voxhess into features
    for (size_t i = 0; i < num_features_; i++) {
      auto& feature = voxhess_.getFeatureMutable(i);
      feature.generateOriginalPlanes();
      num_total_observations_ += feature.index.size();
      N_total_ += feature.coeff;
    }
    N_mean = N_total_ / num_total_observations_;
    LOG(INFO) << "N_mean: " << N_mean;

    LOG(INFO) << "num features in BEC: " << num_features_;
  }

  void PrepareForEvaluation(
      bool evaluate_jacobians, bool new_evaluation_point) {
    if (new_evaluation_point) {
      double balm_residual = 0.0;
      for (size_t i = 0; i < num_features_; i++) {
        feature_planes_[i] =
            voxhess_.getFeature(i).evaluatePlane(xs_, T_I_S_, T_G_M_);
        // Optional evaluation of the original balm residual. not needed for the
        // optimization

        balm_residual += voxhess_.getFeature(i).evaluateResidual(
            xs_, cluster_G[i], lmbd_[i], U_[i], T_I_S_, T_G_M_);
      }
      VLOG(3) << "BALM Residual: " << balm_residual
              << " Num features: " << num_features_;
    }
    evaluated_ = true;
  }

  bool preparedForEvaluation() const {
    return evaluated_;
  }

  size_t getNumFeatures() const {
    return num_features_;
  }

  double getNumObsForFeature(const size_t feat_ind) const {
    return voxhess_.getFeature(feat_ind).index.size();
  }

  const aslam::Transformation& get_T_I_S() const {
    return T_I_S_;
  }
  const aslam::Transformation& get_T_G_M() const {
    return T_G_M_;
  }

  const std::vector<PointCluster>& getClusterInSensorFrame(
      const size_t feat_ind) const {
    return voxhess_.getFeature(feat_ind).observation_clusters_S;
  }

  const Eigen::Vector3d& getMeanPlanePoint(const size_t feat_ind) const {
    return feature_planes_[feat_ind].p;
  }

  std::vector<BALMPlane> getFeaturePlanes(
      const std::vector<std::pair<size_t, size_t>> feature_index) const {
    std::vector<BALMPlane> planes;
    planes.reserve(feature_index.size());
    for (const auto& pair : feature_index) {
      const size_t feat_ind = pair.first;
      planes.push_back(feature_planes_[feat_ind]);
    }
    return planes;
  }

  std::vector<BALMPlane> getOriginalPlanesij(
      const std::vector<std::pair<size_t, size_t>>& feature_index) const {
    std::vector<BALMPlane> planes;
    planes.reserve(feature_index.size());
    for (const auto& pair : feature_index) {
      const size_t feat_ind = pair.first;
      const size_t pose_ind = pair.second;
      const auto& plane_vec = voxhess_.getFeature(feat_ind).planes_S;
      CHECK_LT(pose_ind, plane_vec.size());
      auto plane = plane_vec[pose_ind];
      CHECK(std::abs(plane.n.norm() - 1.0) <= 1e-6)
          << "plane.n.norm(): " << plane.n.norm();
      planes.emplace_back(std::move(plane));
    }
    return planes;
  }

  double get_N_i(const size_t feat_ind) const {
    return cluster_G[feat_ind].N;
  }

  double get_N_ij(const size_t feat_ind, const size_t pose_ind) const {
    return voxhess_.getFeature(feat_ind).observation_clusters_S[pose_ind].N;
  }

  double get_N_mean() const {
    return N_mean;
  }

  const VoxHess& getVoxHess() const {
    return voxhess_;
  }

 private:
  // number of total balm features
  const size_t num_features_;
  // total number of points in all features
  double N_total_ = 0;
  // total number of planes in all features
  double num_total_observations_ = 0;
  // mean number of points per feature observation
  double N_mean = 0.0;
  VoxHess voxhess_;
  std::vector<PointCluster> cluster_G;
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
      : evaluation_callback_(evaluation_callback),
        feature_index_(feature_index),
        i_(i),
        residual_size_(feature_index.size() * balmblocks::kResidualSize),
        T_I_S_(evaluation_callback->get_T_I_S()),
        T_G_M_(evaluation_callback->get_T_G_M()) {
    CHECK_NOTNULL(evaluation_callback.get());
    // for ceres::CostFunction
    set_num_residuals(residual_size_);
    std::vector<int32_t>* parameter_block_sizes =
        mutable_parameter_block_sizes();
    parameter_block_sizes->resize(1, balmblocks::kPoseSize);
    //////////////////////////
    // get the original observed planes in local lidar frame
    original_planes_ij_ =
        evaluation_callback_->getOriginalPlanesij(feature_index_);
    CHECK(original_planes_ij_.size() == feature_index.size())
        << "size mismatch" << original_planes_ij_.size() << " "
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
  std::vector<BALMPlane> original_planes_ij_;
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