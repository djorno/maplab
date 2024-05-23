#ifndef CERES_ERROR_TERMS_BALM_ERROR_TERM_INL_H_
#define CERES_ERROR_TERMS_BALM_ERROR_TERM_INL_H_

#include <ceres-error-terms/balm-voxhess.h>
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
  //  LM = mission of the landmark-base vertex, expressed in G
  //  B = base vertex of the landmark, expressed in LM
  //  I = IMU position of the keyframe vertex, expressed in M
  //  C = LiDAR position, expressed in I
  LOG(INFO) << "BALMErrorTerm::Evaluate";
  LOG(INFO) << "feat_num_: " << feat_num_;
  JplQuaternionParameterization quat_parametrization;
  LOG(INFO) << "CP 1";
  *residuals = evaluation_callback_->get_residual(feat_num_);
  // Unpack parameter blocks.
  Eigen::Map<const Eigen::Quaterniond> q_I_M_JPL(parameters[kIdxPose]);
  Eigen::Map<const Eigen::Vector3d> p_M_I(
      parameters[kIdxPose] + balmblocks::kOrientationBlockSize);
  // from active to passive and from JPL to Hamilton (no inversion needed)
  LOG(INFO) << "CP 2";
  Eigen::Quaterniond q_M_I = q_I_M_JPL;

  aslam::Transformation T_M_I(p_M_I, q_M_I);

  aslam::Transformation T_G_M = evaluation_callback_->get_T_G_M();
  aslam::Transformation T_I_S = evaluation_callback_->get_T_I_S();
  LOG(INFO) << "CP 3";

  // Transform T_M_I to T_G_S for BALM jacobian calculation
  aslam::Transformation T_G_S = T_G_M * T_M_I * T_I_S;

  // Evaluate the jacobian as in bavoxel.h

  double coe = evaluation_callback_->get_coeff(feat_num_);
  Eigen::Vector3d uk = evaluation_callback_->get_uk(feat_num_);
  const Eigen::Vector3d& vBar = evaluation_callback_->get_vBar(feat_num_);
  LOG(INFO) << "CP 4";

  Eigen::Matrix3d Pi =
      evaluation_callback_->get_sig_origin(feat_num_)[sig_i_].P;
  Eigen::Vector3d vi =
      evaluation_callback_->get_sig_origin(feat_num_)[sig_i_].v;
  size_t Ni = evaluation_callback_->get_sig_origin(feat_num_)[sig_i_].N;
  LOG(INFO) << "CP 5";
  Eigen::Matrix3d Ri = T_G_S.getRotationMatrix();

  Eigen::Matrix3d vihat = common::skew(vi);
  Eigen::Vector3d RiTuk = Ri.transpose() * uk;
  Eigen::Matrix3d RiTukhat = common::skew(RiTuk);
  LOG(INFO) << "CP 6";

  Eigen::Vector3d PiRiTuk = Pi * RiTuk;

  Eigen::Vector3d ti_v = T_G_S.getPosition() - vBar;
  double ukTti_v = uk.dot(ti_v);

  Eigen::Matrix3d combo1 = hat(PiRiTuk) + vihat * ukTti_v;
  Eigen::Vector3d combo2 = Ri * vi + Ni * ti_v;

  int NN = evaluation_callback_->get_NN(feat_num_);

  Eigen::Matrix<double, 3, 6> Auk;
  Auk.block<3, 3>(0, 0) =
      (Ri * Pi + ti_v * vi.transpose()) * RiTukhat - Ri * combo1;
  Auk.block<3, 3>(0, 3) =
      combo2 * uk.transpose() + combo2.dot(uk) * Eigen::Matrix3d::Identity();
  Auk /= NN;
  LOG(INFO) << "CP 7";

  const Eigen::Matrix<double, 6, 1>& jjt = Auk.transpose() * uk;
  const Eigen::Matrix<double, balmblocks::kResidualSize, 6> J_res_wrt_T_G_S =
      coe * jjt.transpose();
  LOG(INFO) << "CP 8";

  const Eigen::Matrix<double, 3, 1> J_res_wrt_p_M_S =
      J_res_wrt_T_G_S.block<3, 1>(0, 3);
  const Eigen::Matrix<double, 3, 1> J_res_wrt_q_M_S =
      J_res_wrt_T_G_S.block<3, 1>(0, 0);

  Eigen::Matrix<double, 3, 3> R_S_I = T_I_S.getRotationMatrix().transpose();
  Eigen::Matrix<double, 3, 3> R_I_G = T_M_I.getRotationMatrix().transpose() *
                                      T_G_M.getRotationMatrix().transpose();

  Eigen::Matrix<double, 3, 3> J_p_M_S_wrt_p_M_I = Eigen::Matrix3d::Identity();
  Eigen::Matrix<double, 3, 3> J_p_M_S_wrt_q_M_I =
      common::skew(T_M_I.getRotationMatrix() * T_I_S.getPosition());

  Eigen::Matrix<double, 3, 3> J_q_M_S_wrt_p_M_I = Eigen::Matrix3d::Zero();
  Eigen::Matrix<double, 3, 3> J_q_M_S_wrt_q_M_I = Eigen::Matrix3d::Identity();
  LOG(INFO) << "CP 9";

  Eigen::Matrix<double, 6, 6> J_T_M_S_wrt_T_M_I =
      Eigen::Matrix<double, 6, 6>::Zero();
  J_T_M_S_wrt_T_M_I.block<3, 3>(0, 0) = J_q_M_S_wrt_q_M_I;
  J_T_M_S_wrt_T_M_I.block<3, 3>(3, 3) = J_p_M_S_wrt_p_M_I;
  J_T_M_S_wrt_T_M_I.block<3, 3>(3, 0) = J_p_M_S_wrt_q_M_I;
  J_T_M_S_wrt_T_M_I.block<3, 3>(0, 3) = J_q_M_S_wrt_p_M_I;

  Eigen::Matrix<double, 1, 6> J_res_wrt_T_M_I =
      J_res_wrt_T_G_S * J_T_M_S_wrt_T_M_I;

  Eigen::Matrix<double, 1, 3> J_res_wrt_q_M_I =
      J_res_wrt_T_M_I.block<1, 3>(0, 0);
  Eigen::Matrix<double, 1, 3> J_res_wrt_p_M_I =
      J_res_wrt_T_M_I.block<1, 3>(0, 3);

  LOG(INFO) << "CP 10";

  if (jacobians[kIdxPose]) {
    LOG(INFO) << "CP 11";
    Eigen::Map<PoseJacobian> J(jacobians[kIdxPose]);
    LOG(INFO) << "CP 12";
    Eigen::Matrix<double, 4, 3, Eigen::RowMajor> J_quat_local_param;
    LOG(INFO) << "CP 13";
    quat_parametrization.ComputeJacobian(
        q_M_I.coeffs().data(), J_quat_local_param.data());
    J.setZero();
    J.leftCols(balmblocks::kOrientationBlockSize) =
        J_res_wrt_q_M_I * J_quat_local_param.transpose();
    LOG(INFO) << "CP 14";
    J.rightCols(balmblocks::kPositionBlockSize) = J_res_wrt_p_M_I;
    LOG(INFO) << "CP 15";
  }
  LOG(INFO) << "End of BALMErrorTerm::Evaluate";
}

}  // namespace ceres_error_terms

#endif  // CERES_ERROR_TERMS_BALM_ERROR_TERM_INL_H_