#ifndef CERES_ERROR_TERMS_BALM_ERROR_TERM_INL_H_
#define CERES_ERROR_TERMS_BALM_ERROR_TERM_INL_H_

#include <ceres-error-terms/balm-voxhess.h>
#include <cmath>
#include <maplab-common/geometry.h>
#include <maplab-common/quaternion-math.h>

#include "ceres-error-terms/common.h"

namespace ceres_error_terms {

bool BALMErrorTerm::Evaluate(
    double const* const* parameters, double* residuals,
    double** jacobians) const {
  // Coordinate frames:
  //  G = global
  //  M = mission of the keyframe vertex, expressed in G
  //  I = IMU position of the keyframe vertex, expressed in M
  //  S = LiDAR position, expressed in I
  // LOG(INFO) << "BALMErrorTerm::Evaluate";
  JplQuaternionParameterization quat_parametrization;
  // LOG(INFO) << "CP 1";
  CHECK(evaluation_callback_ != nullptr);
  CHECK(parameters != nullptr);

  CHECK(residual_size_ == feature_index_.size() * 6);
  Eigen::Map<Eigen::VectorXd> residuals_vec(residuals, residual_size_);

  // Evaluate is being called before the first evaluation callback. The
  // necessary data is not available.
  if (evaluation_callback_->prepared_for_evaluation() == false) {
    LOG(WARNING) << "Not prepared for evaluation";
    residuals_vec.setZero();
    return true;
  }

  // Unpack parameter blocks.
  Eigen::Map<const Eigen::Quaterniond> q_I_M(parameters[kIdxPose]);
  Eigen::Map<const Eigen::Vector3d> p_M_I(
      parameters[kIdxPose] + balmblocks::kOrientationBlockSize);

  Eigen::Matrix3d R_I_M;
  common::toRotationMatrixJPL(q_I_M.coeffs(), &R_I_M);
  aslam::Quaternion q_I_M_HML(R_I_M);

  aslam::Transformation T_M_I(p_M_I, q_I_M_HML.inverse());

  const aslam::Transformation& T_G_M = evaluation_callback_->get_T_G_M();
  const aslam::Transformation& T_I_S = evaluation_callback_->get_T_I_S();

  aslam::Transformation T_M_S = T_M_I * T_I_S;
  Eigen::Matrix3d R_M_S = T_M_S.getRotationMatrix();
  Eigen::Vector3d p_M_S = T_M_S.getPosition();

  // vector containing the full feature planes for all features observed by the
  // current pose. Ordered as in feature_index_
  const std::vector<BALMPlane>& feature_planes_i =
      evaluation_callback_->get_feature_planes(feature_index_);

  // get normalization factor N_mean
  const double N_mean = evaluation_callback_->get_N_mean();

  for (size_t i = 0; i < feature_index_.size(); i++) {
    const double N_ij = evaluation_callback_->get_N_ij(
        feature_index_[i].first, feature_index_[i].second);
    const double N_i = evaluation_callback_->get_N_i(feature_index_[i].first);

    const double sign =
        (feature_planes_i[i].n.dot(R_M_S * original_planes_ij[i].n) < 0) ? -1.0
                                                                         : 1.0;
    const Eigen::Vector3d& uk_i = sign * feature_planes_i[i].n;
    const Eigen::Vector3d& uk0_ij = original_planes_ij[i].n;
    const Eigen::Vector3d& pBar_i = feature_planes_i[i].p;
    const Eigen::Vector3d& pBar0_ij = original_planes_ij[i].p;

    double sigmainv = original_planes_ij[i].sigmainv;
    // check normals
    CHECK(std::abs(uk_i.norm() - 1.0) <= 1e-6);
    CHECK(std::abs(uk0_ij.norm() - 1.0) <= 1e-6);
    CHECK_GT(uk_i.dot(R_M_S * uk0_ij), 0.0)
        << "Normal direction selection not working";
    // if the normals are identical, the residuals are zero and numerical issues
    // can occur in the gradient.
    if (std::abs(uk_i.dot(R_M_S * uk0_ij) - 1.0) <= 1e-6) {
      sigmainv = 0.0;
    } else {
      CHECK_LE(std::abs(uk_i.dot(R_M_S * uk0_ij)), 1)
          << "uk_i.dot(uk_ij): " << uk_i.dot(R_M_S * uk0_ij) << " i: " << i;
      CHECK_GE(acos(uk_i.dot(R_M_S * uk0_ij)), 0.0);
    }
    CHECK_GE(sigmainv, 0.0) << "sigmainv: " << sigmainv;

    ////////////////////////////// RESIDUALS ////////////////////////////////
    // Visually: r_q = (10 * N_ij * sigmainv) * common::skew(uk_i) * R_M_S*
    // uk0_ij* acos(uk_i.dot(R_M_S * uk0_ij));
    Eigen::Vector3d r_q =
        (sigmainv != 0.0)
            ? ((10 * N_ij * sigmainv) * common::skew(uk_i) * R_M_S * uk0_ij *
               acos(uk_i.dot(R_M_S * uk0_ij)))
                  .eval()
            : Eigen::Vector3d::Zero();  // Set r_q to zero if sigmainv is zero

    Eigen::Vector3d r_p = (10 * N_ij) * uk_i * uk_i.transpose() *
                          (R_M_S * pBar0_ij + p_M_S - pBar_i);
    /////////////////////////////////////////////////////////////////////////

    CHECK(i * 6 + 6 <= residual_size_);
    CHECK(r_q.allFinite()) << "r_q: " << r_q;
    CHECK(r_p.allFinite()) << "r_p: " << r_p;

    residuals_vec.segment<3>(i * 6) = r_q;
    residuals_vec.segment<3>(i * 6 + 3) = r_p;
  }

  if (jacobians) {
    Eigen::Matrix<double, Eigen::Dynamic, 6> J_res_full_wrt_T_M_I;
    J_res_full_wrt_T_M_I.resize(residual_size_, 6);
    J_res_full_wrt_T_M_I.setZero();

    Eigen::Matrix<double, 3, 3> R_S_I = T_I_S.getRotationMatrix().transpose();

    // loop over all features observed by the current pose
    for (size_t i = 0; i < feature_index_.size(); i++) {
      const std::pair<size_t, size_t>& pair = feature_index_[i];
      size_t feat_num = pair.first;
      size_t sig_i = pair.second;
      CHECK(feat_num < evaluation_callback_->get_num_features());
      CHECK(sig_i < evaluation_callback_->get_sig_origin(feat_num).size());
      const double N_ij = evaluation_callback_->get_N_ij(feat_num, sig_i);
      const double N_i = evaluation_callback_->get_N_i(feat_num);

      const double sign =
          (feature_planes_i[i].n.dot(R_M_S * original_planes_ij[i].n) < 0)
              ? -1.0
              : 1.0;
      const Eigen::Vector3d& uk_i = sign * feature_planes_i[i].n;
      const Eigen::Vector3d& uk0_ij = original_planes_ij[i].n;
      const Eigen::Vector3d& pBar_i = feature_planes_i[i].p;
      const Eigen::Vector3d& pBar0_ij = original_planes_ij[i].p;

      double sigmainv = original_planes_ij[i].sigmainv;
      if (std::abs(uk_i.dot(R_M_S * uk0_ij) - 1.0) <= 1e-6) {
        sigmainv = 0.0;
      }
      CHECK_GE(sigmainv, 0.0);

      ///////// computation of the Jacobian of r_p wrt T_M_S
      Eigen::Matrix<double, 3, 3> J_r_p_wrt_q_M_S = (10 * N_ij) * -uk_i *
                                                    uk_i.transpose() * R_M_S *
                                                    common::skew(pBar0_ij);
      Eigen::Matrix<double, 3, 3> J_r_p_wrt_p_M_S =
          (10 * N_ij) * uk_i * uk_i.transpose();
      ////////// computation of the Jacobian of r_q wrt T_M_S
      // Visually;       -(10 * N_ij * sigmainv) * common::skew(uk_i) * R_M_S*
      // common::skew(uk0_ij) * acos(uk_i.dot(R_M_S * uk0_ij)) + (10 * N_ij *
      // sigmainv) * common::skew(uk_i) * R_M_S* uk0_ij* uk_i.transpose() *
      // R_M_S* common::skew(uk0_ij) / sqrt(1.0 + 1e-6 - pow(uk_i.dot(R_M_S *
      // uk0_ij), 2));
      Eigen::Matrix<double, 3, 3> J_r_q_wrt_q_M_S =
          (sigmainv != 0.0)
              ? ((10 * N_ij * sigmainv) *
                 (-common::skew(uk_i) * R_M_S * common::skew(uk0_ij) *
                      acos(uk_i.dot(R_M_S * uk0_ij))

                  + common::skew(uk_i) * R_M_S * uk0_ij * uk_i.transpose() *
                        R_M_S * common::skew(uk0_ij) /
                        sqrt(1.0 + 1e-6 - pow(uk_i.dot(R_M_S * uk0_ij), 2))))
                    .eval()
              : Eigen::Matrix<double, 3, 3>::Zero();  // Zero matrix if
                                                      // sigmainv is zero
      Eigen::Matrix<double, 3, 3> J_r_q_wrt_p_M_S = Eigen::Matrix3d::Zero();

      Eigen::Matrix<double, balmblocks::kResidualSize, 6> J_res_wrt_T_M_S;
      J_res_wrt_T_M_S.setZero();
      J_res_wrt_T_M_S.block<3, 3>(0, 0) = J_r_q_wrt_q_M_S;
      J_res_wrt_T_M_S.block<3, 3>(0, 3) = J_r_q_wrt_p_M_S;
      J_res_wrt_T_M_S.block<3, 3>(3, 3) = J_r_p_wrt_p_M_S;
      J_res_wrt_T_M_S.block<3, 3>(3, 0) = J_r_p_wrt_q_M_S;
      ////////////// TRANSFORM JACOBIAN TO T_M_I //////////////////
      Eigen::Matrix<double, 3, 3> J_p_M_S_wrt_p_M_I =
          Eigen::Matrix3d::Identity();
      Eigen::Matrix<double, 3, 3> J_p_M_S_wrt_q_M_I =
          common::skew(T_M_I.getRotationMatrix() * T_I_S.getPosition());
      Eigen::Matrix<double, 3, 3> J_q_M_S_wrt_p_M_I = Eigen::Matrix3d::Zero();
      Eigen::Matrix<double, 3, 3> J_q_M_S_wrt_q_M_I = R_S_I;

      Eigen::Matrix<double, 6, 6> J_T_M_S_wrt_T_M_I =
          Eigen::Matrix<double, 6, 6>::Identity();
      J_T_M_S_wrt_T_M_I.block<3, 3>(0, 0) = J_q_M_S_wrt_q_M_I;
      J_T_M_S_wrt_T_M_I.block<3, 3>(3, 3) = J_p_M_S_wrt_p_M_I;
      J_T_M_S_wrt_T_M_I.block<3, 3>(3, 0) = J_p_M_S_wrt_q_M_I;
      J_T_M_S_wrt_T_M_I.block<3, 3>(0, 3) = J_q_M_S_wrt_p_M_I;

      Eigen::Matrix<double, balmblocks::kResidualSize, 6> J_res_wrt_T_M_I =
          J_res_wrt_T_M_S * J_T_M_S_wrt_T_M_I;

      ////////////////////////////////////////////////////////////
      CHECK(i * balmblocks::kResidualSize + 6 <= residual_size_);
      J_res_full_wrt_T_M_I.block<6, 6>(balmblocks::kResidualSize * i, 0) =
          J_res_wrt_T_M_I;
    }
    if (jacobians[kIdxPose]) {
      Eigen::Map<PoseJacobian> J(
          jacobians[kIdxPose], residual_size_, balmblocks::kPoseSize);

      // Compute the parametrization jacobian. Since ceres expects the jacobian
      // wrt the parameters and we have the jacobian wrt the SE(3) retraction,
      // we must apply the inverse of the parametrization jacobian to our
      // residual jacobian. Factor 4 is due to the internal computation in
      // ComputeJacobian
      Eigen::Matrix<double, 4, 3, Eigen::RowMajor> J_quat_local_param;
      quat_parametrization.ComputeJacobian(
          q_I_M.coeffs().data(), J_quat_local_param.data());

      J.setZero();
      J.leftCols(balmblocks::kOrientationBlockSize) =
          J_res_full_wrt_T_M_I.leftCols(balmblocks::kOrientationBlockSize) * 4 *
          J_quat_local_param.transpose();
      J.rightCols(balmblocks::kPositionBlockSize) =
          J_res_full_wrt_T_M_I.rightCols(balmblocks::kPositionBlockSize);
    } else {
      LOG(INFO) << "Jacobian not set";
    }
  }
  return true;
}

}  // namespace ceres_error_terms

#endif  // CERES_ERROR_TERMS_BALM_ERROR_TERM_INL_H_