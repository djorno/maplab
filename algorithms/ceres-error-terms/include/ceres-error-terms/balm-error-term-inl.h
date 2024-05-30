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
  JplQuaternionParameterization quat_parametrization;
  // LOG(INFO) << "CP 1";
  CHECK(parameters != nullptr);
  std::vector<const double*> xs_;
  for (size_t i = 0; i < num_poses_; i++) {
    xs_.push_back(parameters[i]);
  }

  if (residuals) {
    std::vector<double> residual_(num_features_);
    for (size_t i = 0; i < num_features_; i++) {
      double res = voxhess_atoms_[i].evaluate_residual(
          xs_, sig_transformed_[i], lmbd_[i], U_[i], T_I_S_, T_G_M_);
      residual_[i] = res;
      uk_[i] = U_[i].col(0);
    }
    *residuals = std::accumulate(residual_.begin(), residual_.end(), 0.0);
  }

  else {
    LOG(WARNING) << "Residuals not set";
  }

  if (jacobians) {
    for (size_t param_num = 0; param_num < num_poses_; param_num++) {
      std::vector<std::pair<size_t, size_t>> feature_index_ =
          feature_indices_[param_num];
      // Unpack parameter blocks.
      Eigen::Map<const Eigen::Quaterniond> q_I_M_JPL(xs_[param_num]);
      Eigen::Map<const Eigen::Vector3d> p_M_I(
          xs_[param_num] + balmblocks::kOrientationBlockSize);
      // from active to passive and from JPL to Hamilton (no inversion needed)
      Eigen::Quaterniond q_M_I = q_I_M_JPL;

      aslam::Transformation T_M_I(p_M_I, q_M_I);
      // LOG(INFO) << "CP 3";
      Eigen::Matrix<double, balmblocks::kResidualSize, 6> J_res_wrt_T_G_S =
          Eigen::Matrix<double, balmblocks::kResidualSize, 6>::Zero();
      // Transform T_M_I to T_G_S for BALM jacobian calculation
      aslam::Transformation T_G_S = T_G_M_ * T_M_I * T_I_S_;

      // Evaluate the jacobian as in bavoxel.h
      for (const auto& pair : feature_index_) {
        size_t feat_num = pair.first;
        size_t sig_i = pair.second;
        CHECK(feat_num < get_num_features());
        CHECK(sig_i < get_sig_origin(feat_num).size());

        double coe = get_coeff(feat_num);
        Eigen::Vector3d uk = get_uk(feat_num);
        const Eigen::Vector3d& vBar = get_vBar(feat_num);
        // LOG(INFO) << "CP 4";

        Eigen::Matrix3d Pi = get_sig_origin(feat_num)[sig_i].P;
        Eigen::Vector3d vi = get_sig_origin(feat_num)[sig_i].v;
        size_t ni = get_sig_origin(feat_num)[sig_i].N;
        // LOG(INFO) << "CP 5";
        Eigen::Matrix3d Ri = T_G_S.getRotationMatrix();

        Eigen::Matrix3d vihat = common::skew(vi);
        Eigen::Vector3d RiTuk = Ri.transpose() * uk;
        Eigen::Matrix3d RiTukhat = common::skew(RiTuk);
        // LOG(INFO) << "CP 6";

        Eigen::Vector3d PiRiTuk = Pi * RiTuk;

        Eigen::Vector3d ti_v = T_G_S.getPosition() - vBar;
        double ukTti_v = uk.dot(ti_v);

        Eigen::Matrix3d combo1 = hat(PiRiTuk) + vihat * ukTti_v;
        Eigen::Vector3d combo2 = Ri * vi + ni * ti_v;

        int NN = get_NN(feat_num);

        Eigen::Matrix<double, 3, 6> Auk;
        Auk.block<3, 3>(0, 0) =
            (Ri * Pi + ti_v * vi.transpose()) * RiTukhat - Ri * combo1;
        Auk.block<3, 3>(0, 3) = combo2 * uk.transpose() +
                                combo2.dot(uk) * Eigen::Matrix3d::Identity();
        Auk /= NN;
        // LOG(INFO) << "CP 7";

        const Eigen::Matrix<double, 6, 1>& jjt = Auk.transpose() * uk;
        J_res_wrt_T_G_S += coe * jjt.transpose();
      }
      // LOG(INFO) << "Jacobian i: " << param_num << " = " << J_res_wrt_T_G_S;

      const Eigen::Matrix<double, 3, 1> J_res_wrt_p_M_S =
          J_res_wrt_T_G_S.block<3, 1>(0, 3);
      const Eigen::Matrix<double, 3, 1> J_res_wrt_q_M_S =
          J_res_wrt_T_G_S.block<3, 1>(0, 0);

      Eigen::Matrix<double, 3, 3> R_S_I =
          T_I_S_.getRotationMatrix().transpose();
      Eigen::Matrix<double, 3, 3> R_I_G =
          T_M_I.getRotationMatrix().transpose() *
          T_G_M_.getRotationMatrix().transpose();

      Eigen::Matrix<double, 3, 3> J_p_M_S_wrt_p_M_I =
          Eigen::Matrix3d::Identity();
      Eigen::Matrix<double, 3, 3> J_p_M_S_wrt_q_M_I =
          common::skew(T_M_I.getRotationMatrix() * T_I_S_.getPosition());

      Eigen::Matrix<double, 3, 3> J_q_M_S_wrt_p_M_I = Eigen::Matrix3d::Zero();
      Eigen::Matrix<double, 3, 3> J_q_M_S_wrt_q_M_I = R_S_I;
      //  =
      //     Eigen::Matrix3d::Identity();
      // LOG(INFO) << "CP 9";

      Eigen::Matrix<double, 6, 6> J_T_M_S_wrt_T_M_I =
          Eigen::Matrix<double, 6, 6>::Zero();
      J_T_M_S_wrt_T_M_I.block<3, 3>(0, 0) = J_q_M_S_wrt_q_M_I;
      J_T_M_S_wrt_T_M_I.block<3, 3>(3, 3) = J_p_M_S_wrt_p_M_I;
      J_T_M_S_wrt_T_M_I.block<3, 3>(3, 0) = J_p_M_S_wrt_q_M_I;
      J_T_M_S_wrt_T_M_I.block<3, 3>(0, 3) = J_q_M_S_wrt_p_M_I;

      //////////////////// SIDEQUEST ///////////////////////////
      // Eigen::Matrix<double, 6, 6> adj = Eigen::Matrix<double, 6, 6>::Zero();

      // Eigen::Matrix<double, 3, 3> R_I_S = T_I_S_.getRotationMatrix();

      // adj.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
      // adj.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
      // adj.block<3, 3>(3, 0) =
      //     common::skew(T_I_S_.inverse().getPosition()) * R_S_I;

      // Eigen::Matrix<double, 1, 6> J_res_wrt_T_M_I = J_res_wrt_T_G_S * adj;
      /////////////////////////////////////////////////////////

      Eigen::Matrix<double, 1, 6> J_res_wrt_T_M_I =
          J_res_wrt_T_G_S * J_T_M_S_wrt_T_M_I;

      Eigen::Matrix<double, 1, 3> J_res_wrt_q_M_I =
          J_res_wrt_T_M_I.block<1, 3>(0, 0);
      Eigen::Matrix<double, 1, 3> J_res_wrt_p_M_I =
          J_res_wrt_T_M_I.block<1, 3>(0, 3);

      // LOG(INFO) << "CP 10"
      // LOG(INFO) << "CP 11";
      if (jacobians[param_num]) {
        Eigen::Map<PoseJacobian> J(jacobians[param_num]);
        // LOG(INFO) << "CP 12";
        Eigen::Matrix<double, 4, 3, Eigen::RowMajor> J_quat_local_param;
        // LOG(INFO) << "CP 13";
        quat_parametrization.ComputeJacobian(
            q_M_I.coeffs().data(), J_quat_local_param.data());
        J.setZero();
        J.leftCols(balmblocks::kOrientationBlockSize) =
            J_res_wrt_q_M_I * 4.0 * J_quat_local_param.transpose();
        // LOG(INFO) << "CP 14";
        J.rightCols(balmblocks::kPositionBlockSize) = J_res_wrt_p_M_I;
        // LOG(INFO) << "Jacobian mod i: " << param_num << " = "
        //           << J_res_wrt_T_M_I;

      } else {
        LOG(WARNING) << "Jacobian not set";
      }
    }
    // LOG(INFO) << "CP 15";
  }
  return true;
  // LOG(INFO) << "End of BALMErrorTerm::Evaluate";
}

}  // namespace ceres_error_terms

#endif  // CERES_ERROR_TERMS_BALM_ERROR_TERM_INL_H_