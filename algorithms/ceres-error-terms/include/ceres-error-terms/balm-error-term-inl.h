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

  // compute the residuals
  std::vector<Eigen::Vector3d> uk_ij_vec;
  std::vector<Eigen::Vector3d> pBar_ij_vec;

  const std::vector<Eigen::Vector3d> uk_i_vec =
      evaluation_callback_->get_uk_i(feature_index_);
  const std::vector<Eigen::Vector3d> pBar_i_vec =
      evaluation_callback_->get_vBar_i(feature_index_);

  evaluation_callback_->evaluate_single_plane(
      feature_index_, parameters[kIdxPose], uk_ij_vec, pBar_ij_vec);
  // get residual vector
  CHECK(residual_size_ == feature_index_.size() * 6);
  Eigen::Map<Eigen::VectorXd> residuals_eigen(residuals, residual_size_);

  for (size_t i = 0; i < feature_index_.size(); i++) {
    // CHECK_GE(uk_i_vec[i].dot(uk_ij_vec[i]), -1)
    //     << "uk_i: " << uk_i_vec[i].norm() << " uk_ij: " <<
    //     uk_ij_vec[i].norm();
    // CHECK_LE(uk_i_vec[i].dot(uk_ij_vec[i]), 1)
    //     << "uk_i: " << uk_i_vec[i].norm() << " uk_ij: " <<
    //     uk_ij_vec[i].norm();
    const double N_ij = evaluation_callback_->get_N_ij(
        feature_index_[i].first, feature_index_[i].second);
    const double N_i = evaluation_callback_->get_N_i(feature_index_[i].first);
    if (N_i == 0) {
      LOG(WARNING) << "N_i is zero";
      residuals_eigen.segment<6>(i * 6).setZero();
      continue;
    }
    CHECK(N_ij / N_i <= 1.0) << "N_ij/N_i: " << N_ij / N_i;

    Eigen::Vector3d r_q = common::skew(uk_i_vec[i]) * uk_ij_vec[i];
    //  *
    //                       acos(uk_i_vec[i].dot(uk_ij_vec[i]));
    Eigen::Vector3d r_p =  // uk_i_vec[i] * uk_i_vec[i].transpose() *
        (pBar_ij_vec[i] - pBar_i_vec[i]);
    CHECK(i * 6 + 3 < residual_size_);
    // CHECK(r_q.allFinite()) << "r_q: " << r_q;
    // CHECK(r_p.allFinite()) << "r_p: " << r_p;
    if (!r_q.allFinite()) {
      r_q.setZero();
      LOG(WARNING) << "r_q is not finite";
    }
    if (!r_p.allFinite()) {
      r_p.setZero();
      LOG(WARNING) << "r_p is not finite";
    }
    residuals_eigen.segment<3>(i * 6) = r_q;
    residuals_eigen.segment<3>(i * 6 + 3) = r_p;
  }

  if (jacobians) {
    // Unpack parameter blocks.
    Eigen::Map<const Eigen::Quaterniond> q_I_M_JPL(parameters[kIdxPose]);
    Eigen::Map<const Eigen::Vector3d> p_M_I(
        parameters[kIdxPose] + balmblocks::kOrientationBlockSize);
    // from active to passive and from JPL to Hamilton (no inversion needed)
    Eigen::Quaterniond q_M_I = q_I_M_JPL;

    aslam::Transformation T_M_I(p_M_I, q_M_I);

    aslam::Transformation T_G_M = evaluation_callback_->get_T_G_M();
    aslam::Transformation T_I_S = evaluation_callback_->get_T_I_S();
    // LOG(INFO) << "CP 3";
    Eigen::Matrix<double, Eigen::Dynamic, 6> J_res_full_wrt_T_M_I;
    J_res_full_wrt_T_M_I.resize(residual_size_, 6);
    J_res_full_wrt_T_M_I.setZero();
    // Transform T_M_I to T_G_S for BALM jacobian calculation
    aslam::Transformation T_G_S = T_G_M * T_M_I * T_I_S;

    Eigen::Matrix<double, 3, 3> R_S_I = T_I_S.getRotationMatrix().transpose();
    Eigen::Matrix<double, 3, 3> R_I_G = T_M_I.getRotationMatrix().transpose() *
                                        T_G_M.getRotationMatrix().transpose();

    // Evaluate the jacobian as in bavoxel.h
    size_t res_number = 0;
    for (const auto& pair : feature_index_) {
      size_t feat_num = pair.first;
      size_t sig_i = pair.second;
      CHECK(feat_num < evaluation_callback_->get_num_features());
      CHECK(sig_i < evaluation_callback_->get_sig_origin(feat_num).size());
      const double N_ij = evaluation_callback_->get_N_ij(feat_num, sig_i);
      const double N_i = evaluation_callback_->get_N_i(feat_num);
      const Eigen::Vector3d uk_i = uk_i_vec[res_number];
      const Eigen::Vector3d uk_ij = uk_ij_vec[res_number];
      const Eigen::Vector3d pBar_i = pBar_i_vec[res_number];
      const Eigen::Vector3d pBar_ij = pBar_ij_vec[res_number];
      ///////// computation of the Jacobian of r_p wrt T_G_S
      Eigen::Matrix<double, 3, 3> J_r_p_wrt_q_G_S = common::skew(pBar_ij);
      Eigen::Matrix<double, 3, 3> J_r_p_wrt_p_G_S =
          Eigen::Matrix3d::Identity();  // uk_i * uk_i.transpose();

      ////////// computation of the Jacobian of r_q wrt T_G_S
      // Eigen::Matrix<double, 3, 3> J_r_q_wrt_q_G_S =
      //     (-acos(uk_i.dot(uk_ij)) * common::skew(uk_i) +
      //      common::skew(uk_i) * uk_ij * uk_i.transpose() *
      //          (1 / (1 - std::pow(uk_i.dot(uk_ij), 2)))) *
      //     common::skew(uk_ij);
      Eigen::Matrix<double, 3, 3> J_r_q_wrt_q_G_S =
          -common::skew(uk_i) * common::skew(uk_ij);
      Eigen::Matrix<double, 3, 3> J_r_q_wrt_p_G_S = Eigen::Matrix3d::Zero();

      Eigen::Matrix<double, balmblocks::kResidualSize, 6> J_res_wrt_T_G_S;
      J_res_wrt_T_G_S.setZero();
      J_res_wrt_T_G_S.block<3, 3>(0, 0) = J_r_q_wrt_q_G_S;
      J_res_wrt_T_G_S.block<3, 3>(0, 3) = J_r_q_wrt_p_G_S;
      J_res_wrt_T_G_S.block<3, 3>(3, 3) = J_r_p_wrt_p_G_S;
      J_res_wrt_T_G_S.block<3, 3>(3, 0) = J_r_p_wrt_q_G_S;
      ////////////// TRANSFORM JACOBIAN TO T_M_I //////////////////
      Eigen::Matrix<double, 3, 3> J_p_M_S_wrt_p_M_I =
          Eigen::Matrix3d::Identity();
      Eigen::Matrix<double, 3, 3> J_p_M_S_wrt_q_M_I = R_S_I;
      // common::skew(T_M_I.getRotationMatrix() * T_I_S.getPosition());

      //   Eigen::Matrix<double, 3, 3> J_q_M_S_wrt_p_M_I =
      //   Eigen::Matrix3d::Zero(); Eigen::Matrix<double, 3, 3>
      //   J_q_M_S_wrt_q_M_I = R_S_I;
      Eigen::Matrix<double, 3, 3> J_q_M_S_wrt_p_M_I = Eigen::Matrix3d::Zero();
      Eigen::Matrix<double, 3, 3> J_q_M_S_wrt_q_M_I =
          -common::skew(T_M_I.getRotationMatrix() * T_I_S.getPosition());

      Eigen::Matrix<double, 6, 6> J_T_M_S_wrt_T_M_I =
          Eigen::Matrix<double, 6, 6>::Zero();
      J_T_M_S_wrt_T_M_I.block<3, 3>(0, 0) = J_q_M_S_wrt_q_M_I;
      J_T_M_S_wrt_T_M_I.block<3, 3>(3, 3) = J_p_M_S_wrt_p_M_I;
      J_T_M_S_wrt_T_M_I.block<3, 3>(3, 0) = J_p_M_S_wrt_q_M_I;
      J_T_M_S_wrt_T_M_I.block<3, 3>(0, 3) = J_q_M_S_wrt_p_M_I;

      Eigen::Matrix<double, 3, 3> J_r_p_wrt_q_M_I =
          J_r_p_wrt_q_G_S * J_p_M_S_wrt_q_M_I;
      Eigen::Matrix<double, 3, 3> J_r_p_wrt_p_M_I =
          J_r_p_wrt_p_G_S * J_p_M_S_wrt_p_M_I;
      Eigen::Matrix<double, 3, 3> J_r_q_wrt_q_M_I =
          J_r_q_wrt_q_G_S * J_q_M_S_wrt_q_M_I;
      Eigen::Matrix<double, 3, 3> J_r_q_wrt_p_M_I =
          J_r_q_wrt_p_G_S * J_q_M_S_wrt_p_M_I;

      // Eigen::Matrix<double, balmblocks::kResidualSize, 6> J_res_wrt_T_M_I =
      //     J_res_wrt_T_G_S * J_T_M_S_wrt_T_M_I;
      Eigen::Matrix<double, balmblocks::kResidualSize, 6> J_res_wrt_T_M_I =
          Eigen::Matrix<double, balmblocks::kResidualSize, 6>::Zero();
      J_res_wrt_T_M_I.block<3, 3>(0, 0) = J_r_q_wrt_q_M_I;
      J_res_wrt_T_M_I.block<3, 3>(0, 3) = J_r_q_wrt_p_M_I;
      J_res_wrt_T_M_I.block<3, 3>(3, 3) = J_r_p_wrt_p_M_I;
      J_res_wrt_T_M_I.block<3, 3>(3, 0) = J_r_p_wrt_q_M_I;

      ////////////////////////////////////////////////////////////
      CHECK(res_number < residual_size_);
      J_res_full_wrt_T_M_I.block<6, 6>(
          balmblocks::kResidualSize * res_number, 0) = J_res_wrt_T_G_S;
      //   LOG(INFO) << "Jac of i: " << i_ << " feat: " << feat_num
      //             << "jac =" << J_res_wrt_T_G_S;
      res_number++;
    }

    // LOG(INFO) << "CP 10"
    // LOG(INFO) << "CP 11";
    if (jacobians[kIdxPose]) {
      Eigen::Map<PoseJacobian> J(
          jacobians[kIdxPose], residual_size_, balmblocks::kPoseSize);

      Eigen::Matrix<double, 4, 3, Eigen::RowMajor> J_quat_local_param;
      quat_parametrization.ComputeJacobian(
          q_M_I.coeffs().data(), J_quat_local_param.data());
      J.setZero();
      J.leftCols(balmblocks::kOrientationBlockSize) =
          J_res_full_wrt_T_M_I.leftCols(balmblocks::kOrientationBlockSize) * 4 *
          J_quat_local_param.transpose();
      J.rightCols(balmblocks::kPositionBlockSize) =
          J_res_full_wrt_T_M_I.rightCols(balmblocks::kPositionBlockSize);
      // LOG(INFO) << "Jacobian mod i: " << i_ << " = " << J;
    } else {
      LOG(INFO) << "Jacobian not set";
    }
    // LOG(INFO) << "CP 15";
  }
  return true;
  // LOG(INFO) << "End of BALMErrorTerm::Evaluate";
}

}  // namespace ceres_error_terms

#endif  // CERES_ERROR_TERMS_BALM_ERROR_TERM_INL_H_