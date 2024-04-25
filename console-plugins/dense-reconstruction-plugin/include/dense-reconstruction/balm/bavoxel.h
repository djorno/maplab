#ifndef BALM_H_
#define BALM_H_

#include <Eigen/Core>
#include <aslam/common/pose-types.h>
#include <gflags/gflags.h>
#include <kindr/minimal/common.h>
#include <maplab-common/threading-helpers.h>
#include <resources-common/point-cloud.h>

#include "dense-reconstruction/balm/state-buffer.h"
#include "dense-reconstruction/balm/inertial-error-term.h"
#include "dense-reconstruction/balm/common.h"
#include "dense-reconstruction/balm/inertial-terms.h" 
#include "dense-reconstruction/balm/test-jac.h"

#include <Eigen/Core>
#include <aslam/common/pose-types.h>
#include <gflags/gflags.h>
#include <kindr/minimal/common.h>
#include <maplab-common/threading-helpers.h>
#include <resources-common/point-cloud.h>

#include <chrono>
#include <cstring>
#include <malloc.h>
#include <string>
#include <cstdint> 

#include <aslam/common/timer.h>
#include <aslam/common/memory.h>
#include <aslam/common/unique-id.h>
#include <glog/logging.h>
#include <vi-map/vi-map.h>
#include <console-common/console.h>
#include <dense-reconstruction/conversion-tools.h>
#include <dense-reconstruction/pmvs-file-utils.h>
#include <dense-reconstruction/pmvs-interface.h>
#include <dense-reconstruction/stereo-dense-reconstruction.h>
#include <depth-integration/depth-integration.h>
#include <gflags/gflags.h>
#include <map-manager/map-manager.h>
#include <maplab-common/conversions.h>
#include <maplab-common/file-system-tools.h>
#include <vi-map/unique-id.h>
#include <vi-map/vi-map.h>
#include <memory>

#include <ceres-error-terms/parameterization/quaternion-param-jpl.h>
#include <imu-integrator/imu-integrator.h>
#include <maplab-common/quaternion-math.h>
#include <iostream>


#include <thread>
#include <vector>

#define PLM(a)                     \
  std::vector<                     \
      Eigen::Matrix<double, a, a>, \
      Eigen::aligned_allocator<Eigen::Matrix<double, a, a>>>
#define PLV(a)                     \
  std::vector<                     \
      Eigen::Matrix<double, a, 1>, \
      Eigen::aligned_allocator<Eigen::Matrix<double, a, 1>>>

DECLARE_double(balm_voxel_size);
DECLARE_uint32(balm_max_layers);
DECLARE_uint32(balm_min_plane_points);
DECLARE_double(balm_max_eigen_value);

int win_size;

class OctoTreeNode;
typedef std::unordered_map<resources::VoxelPosition, OctoTreeNode*> SurfaceMap;

class PointCluster {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Matrix3d P;
  Eigen::Vector3d v;
  size_t N;

  PointCluster() {
    P.setZero();
    v.setZero();
    N = 0;
  }

  void push(const Eigen::Vector3d& vec) {
    N++;
    P += vec * vec.transpose();
    v += vec;
  }

  Eigen::Matrix3d cov() {
    Eigen::Vector3d center = v / N;
    return P / N - center * center.transpose();
  }

  PointCluster& operator+=(const PointCluster& sigv) {
    this->P += sigv.P;
    this->v += sigv.v;
    this->N += sigv.N;

    return *this;
  }

  PointCluster transform(const aslam::Transformation& T) const {
    const Eigen::Matrix3d R = T.getRotationMatrix();
    const Eigen::Vector3d p = T.getPosition();
    const Eigen::Matrix3d rp = R * v * p.transpose();

    PointCluster sigv;
    sigv.N = N;
    sigv.v = R * v + N * p;
    sigv.P =
        R * P * R.transpose() + rp + rp.transpose() + N * p * p.transpose();
    return sigv;
  }
};

Eigen::Matrix3d hat(const Eigen::Vector3d& v) {
  Eigen::Matrix3d Omega;
  Omega << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
  return Omega;
}

class VoxHess {
 public:
  std::vector<const std::vector<size_t>*> indices;
  std::vector<const PointCluster*> sig_vecs;
  std::vector<const std::vector<PointCluster>*> plvec_voxels;
  std::vector<double> coeffs;

  void push_voxel(
      const std::vector<size_t>* index,
      const std::vector<PointCluster>* sig_orig, const PointCluster* fix) {
    double coe = 0;
    for (const PointCluster& p : *sig_orig) {
      coe += p.N;
    }

    indices.emplace_back(index);
    sig_vecs.emplace_back(fix);
    plvec_voxels.emplace_back(sig_orig);
    coeffs.emplace_back(coe);
  }

  void acc_evaluate2(
      const aslam::TransformationVector& xs, int head, int end,
      Eigen::MatrixXd& Hess, Eigen::VectorXd& JacT, double& residual) {
    Hess.setZero();
    JacT.setZero();
    residual = 0;
    const int kk = 0;

    PLV(3) viRiTuk(win_size);
    PLM(3) viRiTukukT(win_size);

    std::vector<
        Eigen::Matrix<double, 3, 6>,
        Eigen::aligned_allocator<Eigen::Matrix<double, 3, 6>>>
        Auk(win_size);
    Eigen::Matrix3d umumT;

    for (int a = head; a < end; a++) {
      const std::vector<PointCluster>& sig_orig = *plvec_voxels[a];
      const std::vector<size_t>& index = *indices[a];
      double coe = coeffs[a];

      PointCluster sig = *sig_vecs[a];
      for (size_t sig_i = 0; sig_i < sig_orig.size(); ++sig_i) {
        const size_t i = index[sig_i];
        sig += sig_orig[sig_i].transform(xs[i]);
      }

      const Eigen::Vector3d& vBar = sig.v / sig.N;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(
          sig.P / sig.N - vBar * vBar.transpose());
      const Eigen::Vector3d& lmbd = saes.eigenvalues();
      const Eigen::Matrix3d& U = saes.eigenvectors();
      int NN = sig.N;

      Eigen::Vector3d u[3] = {U.col(0), U.col(1), U.col(2)};

      const Eigen::Vector3d& uk = u[kk];
      Eigen::Matrix3d ukukT = uk * uk.transpose();
      umumT.setZero();
      for (int i = 0; i < 3; i++) {
        if (i != kk) {
          umumT += 2.0 / (lmbd[kk] - lmbd[i]) * u[i] * u[i].transpose();
        }
      }

      for (size_t sig_i = 0; sig_i < sig_orig.size(); ++sig_i) {
        Eigen::Matrix3d Pi = sig_orig[sig_i].P;
        Eigen::Vector3d vi = sig_orig[sig_i].v;
        double ni = sig_orig[sig_i].N;

        const size_t i = index[sig_i];
        Eigen::Matrix3d Ri = xs[i].getRotationMatrix();

        Eigen::Matrix3d vihat = kindr::minimal::skewMatrix(vi);
        Eigen::Vector3d RiTuk = Ri.transpose() * uk;
        Eigen::Matrix3d RiTukhat = kindr::minimal::skewMatrix(RiTuk);

        Eigen::Vector3d PiRiTuk = Pi * RiTuk;
        viRiTuk[i] = vihat * RiTuk;
        viRiTukukT[i] = viRiTuk[i] * uk.transpose();

        Eigen::Vector3d ti_v = xs[i].getPosition() - vBar;
        double ukTti_v = uk.dot(ti_v);

        Eigen::Matrix3d combo1 = hat(PiRiTuk) + vihat * ukTti_v;
        Eigen::Vector3d combo2 = Ri * vi + ni * ti_v;
        Auk[i].block<3, 3>(0, 0) =
            (Ri * Pi + ti_v * vi.transpose()) * RiTukhat - Ri * combo1;
        Auk[i].block<3, 3>(0, 3) = combo2 * uk.transpose() +
                                   combo2.dot(uk) * Eigen::Matrix3d::Identity();
        Auk[i] /= NN;

        const Eigen::Matrix<double, 6, 1>& jjt = Auk[i].transpose() * uk;
        JacT.block<6, 1>(6 * i, 0) += coe * jjt;

        const Eigen::Matrix3d& HRt = 2.0 / NN * (1.0 - ni / NN) * viRiTukukT[i];
        Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[i];
        Hb.block<3, 3>(0, 0) +=
            2.0 / NN * (combo1 - RiTukhat * Pi) * RiTukhat -
            2.0 / NN / NN * viRiTuk[i] * viRiTuk[i].transpose() -
            0.5 * hat(jjt.block<3, 1>(0, 0));
        Hb.block<3, 3>(0, 3) += HRt;
        Hb.block<3, 3>(3, 0) += HRt.transpose();
        Hb.block<3, 3>(3, 3) += 2.0 / NN * (ni - ni * ni / NN) * ukukT;

        Hess.block<6, 6>(6 * i, 6 * i) += coe * Hb;
      }

      for (int sig_i = 0; sig_i < sig_orig.size() - 1; sig_i++) {
        const double ni = sig_orig[sig_i].N;
        const size_t i = index[sig_i];
        for (int sig_j = sig_i + 1; sig_j < sig_orig.size(); sig_j++) {
          const double nj = sig_orig[sig_j].N;
          const size_t j = index[sig_j];

          Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[j];
          Hb.block<3, 3>(0, 0) +=
              -2.0 / NN / NN * viRiTuk[i] * viRiTuk[j].transpose();
          Hb.block<3, 3>(0, 3) += -2.0 * nj / NN / NN * viRiTukukT[i];
          Hb.block<3, 3>(3, 0) +=
              -2.0 * ni / NN / NN * viRiTukukT[j].transpose();
          Hb.block<3, 3>(3, 3) += -2.0 * ni * nj / NN / NN * ukukT;

          Hess.block<6, 6>(6 * i, 6 * j) += coe * Hb;
        }
      }

      residual += coe * lmbd[kk];
    }

    for (int i = 1; i < win_size; i++) {
      for (int j = 0; j < i; j++) {
        Hess.block<6, 6>(6 * i, 6 * j) =
            Hess.block<6, 6>(6 * j, 6 * i).transpose();
      }
    }
  }

  void evaluate_only_residual(
      const aslam::TransformationVector& xs, double& residual) {
    residual = 0;
    int kk = 0;  // The kk-th lambda value

    for (int a = 0; a < plvec_voxels.size(); a++) {
      const std::vector<PointCluster>& sig_orig = *plvec_voxels[a];
      const std::vector<size_t>& index = *indices[a];

      PointCluster sig = *sig_vecs[a];
      for (size_t sig_i = 0; sig_i < sig_orig.size(); ++sig_i) {
        const size_t i = index[sig_i];
        sig += sig_orig[sig_i].transform(xs[i]);
      }

      Eigen::Vector3d vBar = sig.v / sig.N;
      Eigen::Matrix3d cmt = sig.P / sig.N - vBar * vBar.transpose();

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
      Eigen::Vector3d lmbd = saes.eigenvalues();

      residual += coeffs[a] * lmbd[kk];
    }
  }
};

class OctoTreeNode {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  size_t layer;
  std::vector<size_t> index;
  std::vector<PLV(3)> vec_orig, vec_tran;
  std::vector<PointCluster> sig_orig, sig_tran;
  PointCluster fix_point;

  OctoTreeNode* leaves[8];
  float voxel_center[3];
  float quater_length;

  OctoTreeNode() {
    layer = 0;

    vec_orig.resize(1);
    vec_tran.resize(1);
    sig_orig.resize(1);
    sig_tran.resize(1);

    for (size_t i = 0; i < 8; ++i) {
      leaves[i] = nullptr;
    }
  }

  bool judge_eigen() {
    PointCluster covMat;
    for (const PointCluster& p : sig_tran) {
      covMat += p;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat.cov());
    double decision = saes.eigenvalues()[0] / saes.eigenvalues()[1];

    return (decision < FLAGS_balm_max_eigen_value);
  }

  void cut_func() {
    for (size_t i = 0; i < index.size(); ++i) {
      for (size_t j = 0; j < vec_tran[i].size(); ++j) {
        const int x = vec_tran[i][j][0] > voxel_center[0];
        const int y = vec_tran[i][j][1] > voxel_center[1];
        const int z = vec_tran[i][j][2] > voxel_center[2];
        const int leafnum = (x << 2) + (y << 1) + z;

        if (leaves[leafnum] == nullptr) {
          leaves[leafnum] = new OctoTreeNode();
          leaves[leafnum]->voxel_center[0] =
              voxel_center[0] + (2 * x - 1) * quater_length;
          leaves[leafnum]->voxel_center[1] =
              voxel_center[1] + (2 * y - 1) * quater_length;
          leaves[leafnum]->voxel_center[2] =
              voxel_center[2] + (2 * z - 1) * quater_length;
          leaves[leafnum]->quater_length = quater_length / 2;
          leaves[leafnum]->layer = layer + 1;
          leaves[leafnum]->index.emplace_back(index[i]);
        } else if (leaves[leafnum]->index.back() != index[i]) {
          leaves[leafnum]->index.emplace_back(index[i]);
          leaves[leafnum]->vec_orig.emplace_back(PLV(3)());
          leaves[leafnum]->vec_tran.emplace_back(PLV(3)());
          leaves[leafnum]->sig_orig.emplace_back(PointCluster());
          leaves[leafnum]->sig_tran.emplace_back(PointCluster());
        }

        leaves[leafnum]->vec_orig.back().emplace_back(vec_orig[i][j]);
        leaves[leafnum]->vec_tran.back().emplace_back(vec_tran[i][j]);
        leaves[leafnum]->sig_orig.back().push(vec_orig[i][j]);
        leaves[leafnum]->sig_tran.back().push(vec_tran[i][j]);
      }
    }

    vec_orig = std::vector<PLV(3)>();
    vec_tran = std::vector<PLV(3)>();
  }

  bool recut(VoxHess* vox_opt, resources::PointCloud* points_G) {
    size_t point_size = 0;
    for (const PointCluster& p : sig_tran) {
      point_size += p.N;
    }

    const size_t num_pointclouds = index.size();
    if (num_pointclouds < 2 || point_size <= FLAGS_balm_min_plane_points) {
      return false;
    }

    if (judge_eigen()) {
      // Push planes into visualization if requested.
      if (points_G != nullptr) {
        const float intensity = 255.0 * rand() / (RAND_MAX + 1.0f);

        for (size_t i = 0; i < vec_tran.size(); ++i) {
          for (Eigen::Vector3d point : vec_tran[i]) {
            points_G->xyz.emplace_back(point.x());
            points_G->xyz.emplace_back(point.y());
            points_G->xyz.emplace_back(point.z());
            points_G->scalars.emplace_back(intensity);
          }
        }
      }

      // Deallocate unnecessary memory.
      vec_orig = std::vector<PLV(3)>();
      vec_tran = std::vector<PLV(3)>();
      sig_tran = std::vector<PointCluster>();

      // Push plane into optimization.
      vox_opt->push_voxel(&index, &sig_orig, &fix_point);

      return true;
    } else if (layer == FLAGS_balm_max_layers) {
      return false;
    }

    sig_orig = std::vector<PointCluster>();
    sig_tran = std::vector<PointCluster>();
    cut_func();

    bool keep = false;
    for (size_t i = 0; i < 8; ++i) {
      if (leaves[i] != nullptr) {
        if (leaves[i]->recut(vox_opt, points_G)) {
          keep = true;
        } else {
          delete leaves[i];
          leaves[i] = nullptr;
        }
      }
    }

    return keep;
  }

  ~OctoTreeNode() {
    for (size_t i = 0; i < 8; ++i) {
      if (leaves[i] != nullptr) {
        delete leaves[i];
      }
    }
  }
};

class BALM2 {
 public:
  BALM2() {}

  double divide_thread(
      const aslam::TransformationVector& x_stats, VoxHess& voxhess,
      Eigen::MatrixXd& Hess, Eigen::VectorXd& JacT) {
    size_t thd_num = common::getNumHardwareThreads();
    Hess.setZero();
    JacT.setZero();
    PLM(-1) hessians(thd_num);
    PLV(-1) jacobins(thd_num);

    for (size_t i = 0; i < thd_num; ++i) {
      hessians[i].resize(6 * win_size, 6 * win_size);
      jacobins[i].resize(6 * win_size);
    }

    size_t g_size = voxhess.plvec_voxels.size();
    if (g_size < thd_num) {
      thd_num = 1;
    }

    std::vector<double> resis(thd_num, 0);
    std::vector<std::thread*> mthreads(thd_num);
    double part = 1.0 * g_size / thd_num;
    for (size_t i = 0; i < thd_num; ++i) {
      mthreads[i] = new std::thread(
          &VoxHess::acc_evaluate2, &voxhess, x_stats, part * i, part * (i + 1),
          std::ref(hessians[i]), std::ref(jacobins[i]), std::ref(resis[i]));
    }

    double residual = 0;
    for (size_t i = 0; i < thd_num; ++i) {
      mthreads[i]->join();
      Hess += hessians[i];
      JacT += jacobins[i];
      residual += resis[i];
      delete mthreads[i];
    }

    return residual;
  }

  double only_residual(
      const aslam::TransformationVector& x_stats, VoxHess& voxhess) {
    double residual = 0;

    voxhess.evaluate_only_residual(x_stats, residual);
    return residual;
  }

  void damping_iter(aslam::TransformationVector& x_stats, VoxHess& voxhess, vi_map::VIMap& map) {
    //double u = 0.01, v = 2;
    double u = 0.02, v = 2;
    //double u = 20.48, v = 2;
    double lambda = 0.01;
    Eigen::MatrixXd D(6 * win_size, 6 * win_size),
        Hess(6 * win_size, 6 * win_size);
    Eigen::VectorXd JacT(6 * win_size), dxi(balm_error_terms::kFullResidualSize * win_size);
    Eigen::VectorXd JacT_full(balm_error_terms::kFullResidualSize * win_size);
    Eigen::MatrixXd Hess_balm(balm_error_terms::kFullResidualSize * win_size, balm_error_terms::kFullResidualSize * win_size),
        D_full(balm_error_terms::kFullResidualSize * win_size, balm_error_terms::kFullResidualSize * win_size);

    Eigen::MatrixXd Identity(balm_error_terms::kFullResidualSize * win_size, balm_error_terms::kFullResidualSize * win_size);
    JacT_full.setZero();
    Hess_balm.setZero();

    Eigen::MatrixXd Hess_inertial(balm_error_terms::kFullResidualSize * win_size, balm_error_terms::kFullResidualSize * win_size);
    Eigen::MatrixXd Hess_full(balm_error_terms::kFullResidualSize * win_size, balm_error_terms::kFullResidualSize * win_size);

    Eigen::VectorXd J_T_r(balm_error_terms::kFullResidualSize * win_size); 

    D.setIdentity();
    D_full.setIdentity();
    Identity.setIdentity();

    double residual1, residual2, q, residual_balm, residual_inertial;
    bool is_calc_hess = true;
    aslam::TransformationVector x_stats_temp = x_stats;

    Eigen::Matrix<double, Eigen::Dynamic, 1> temporary_states;
    temporary_states.resize(balm_error_terms::kStateSize * win_size, 1);

    // setup for inertial integration
    // create inertial residual block
    balm_error_terms::ResidualBlockSet inertial_residual_block;
    // create state buffer
    balm_error_terms::OptimizationStateBuffer buffer;
    int num_blocks_added = balm_error_terms::addInertialTermsForEdges(&map, inertial_residual_block, &buffer);
    LOG(INFO) << "Added " << num_blocks_added << " inertial residual blocks to the optimization problem.";
    // check buffer
    LOG(INFO) << "Checking buffer...";
    balm_error_terms::checkBuffer(&buffer, &map);
    LOG(INFO) << "Buffer checked.";

    // get vertex ids
    pose_graph::VertexIdList vertex_ids;
    map.getAllVertexIds(&vertex_ids);

    checkParameterizationQuaternionJPL(buffer, vertex_ids, &map);

    // create jacobians, residuals and parameters pointers
    Eigen::Matrix<double, Eigen::Dynamic, 1> residuals;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> jacobians;
    Eigen::Matrix<double, Eigen::Dynamic, 1> parameters;
    // allocate memory for parameters, residuals and jacobians
    const int num_vertices = map.numVertices();
    CHECK_EQ(num_vertices, win_size);
    residuals.resize(num_vertices * balm_error_terms::kFullResidualSize, 1);
    //residuals.setZero();
    jacobians.resize(num_vertices * balm_error_terms::kFullResidualSize, num_vertices * balm_error_terms::kUpdateSize);
    //jacobians.setZero();
    parameters.resize(num_vertices * balm_error_terms::kStateSize, 1);
    //parameters.setZero();
    
    // Optimization Loop
    for (int i = 0; i < 1; i++) {

      if (is_calc_hess) {
        // calculate the residuals and jacobians
        timing::TimerImpl jac_timer("balm_evaluate_inertial_jacobian");
        balm_error_terms::evaluateInertialJacobian(inertial_residual_block, &map, &buffer, jacobians, residuals, parameters);
        Hess_inertial = jacobians.transpose() * jacobians;
        residual_balm = divide_thread(x_stats, voxhess, Hess, JacT);
        residual_inertial = residuals.squaredNorm();
        residual1 = residual_balm + residual_inertial;
        LOG(INFO) << "Residual balm: " << residual_balm;
        LOG(INFO) << "Residual inertial: " << residual_inertial;
        LOG(INFO) << "Evaluated inertial jacobian in " << jac_timer.Stop() << " s.";
        // insert the JacT into jacobians
        timing::TimerImpl insert_timer("balm_insert_jacT_into_jacobians");
        for (int j = 0; j < num_vertices; j++) {
          JacT_full.block<3, 1>(balm_error_terms::kFullResidualSize * j, 0) = JacT.block<3, 1>(6 * j, 0);
          JacT_full.block<3, 1>(balm_error_terms::kFullResidualSize * j + 12, 0) = JacT.block<3, 1>(6 * j + 3, 0);

          Hess_balm.block<3, 3>(balm_error_terms::kFullResidualSize * j, balm_error_terms::kFullResidualSize * j) = Hess.block<3, 3>(6 * j, 6 * j);
          Hess_balm.block<3, 3>(balm_error_terms::kFullResidualSize * j + 12, balm_error_terms::kFullResidualSize * j + 12) = Hess.block<3, 3>(6 * j + 3, 6 * j + 3);
          Hess_balm.block<3, 3>(balm_error_terms::kFullResidualSize * j, balm_error_terms::kFullResidualSize * j + 12) = Hess.block<3, 3>(6 * j, 6 * j + 3);
          Hess_balm.block<3, 3>(balm_error_terms::kFullResidualSize * j + 12, balm_error_terms::kFullResidualSize * j) = Hess.block<3, 3>(6 * j + 3, 6 * j);
        }
        LOG(INFO) << "Inserted JacT into jacobians in " << insert_timer.Stop() << " s.";

      }
      timing::TimerImpl random_stuff("balm_evaluate_inertial_residual");
      Hess_full = Hess_inertial; //Hess_balm + Hess_inertial;
      D_full.diagonal() = Hess_full.diagonal();
      
      J_T_r = jacobians.transpose() * residuals;
      LOG(INFO) << "Random stuff in: " << random_stuff.Stop() << " s.";
      
      timing::TimerImpl checks_timer("balm_checks");
      LOG(INFO) << "Starting checks...";
      // check that all dimensions are correct
      LOG(INFO) << "Checking dimensions...";
      CHECK_EQ(Hess_balm.rows(), balm_error_terms::kFullResidualSize * win_size);
      CHECK_EQ(Hess_balm.cols(), balm_error_terms::kFullResidualSize * win_size);
      CHECK_EQ(JacT_full.rows(), balm_error_terms::kFullResidualSize * win_size);
      CHECK_EQ(JacT_full.cols(), 1);
      CHECK_EQ(D_full.rows(), balm_error_terms::kFullResidualSize * win_size);
      CHECK_EQ(D_full.cols(), balm_error_terms::kFullResidualSize * win_size);
      CHECK_EQ(residuals.rows(), balm_error_terms::kFullResidualSize * win_size);
      CHECK_EQ(residuals.cols(), 1);
      CHECK_EQ(jacobians.rows(), Hess_balm.rows());
      CHECK_EQ(jacobians.cols(), Hess_balm.cols());

      LOG(INFO) << "Checking matrix entries...";
      // check that all matrix entries are valid
      CHECK(residuals.allFinite());
      CHECK(jacobians.allFinite());
      CHECK(Hess_inertial.allFinite());
      CHECK(parameters.allFinite());
      CHECK(Hess_balm.allFinite());
      CHECK(JacT_full.allFinite());
      CHECK(J_T_r.allFinite());
      // check that all matrix entries are not nan
      LOG(INFO) << "Checking matrix entries for NaN...";
      CHECK(!residuals.hasNaN());
      CHECK(!jacobians.hasNaN());
      CHECK(!Hess_inertial.hasNaN());
      CHECK(!parameters.hasNaN());
      CHECK(!Hess_balm.hasNaN());
      CHECK(!JacT_full.hasNaN());
      CHECK(!J_T_r.hasNaN());

      LOG(INFO) << "Checks passed: " << checks_timer.Stop() << " s.";

      // solve the linear system
      LOG(INFO) << "Solving linear system...";
      //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(Hess_balm + Hess_inertial + u * D_full);
      //LOG(INFO) << "Eigenvalues: " << eigensolver.eigenvalues();
      //LOG(INFO) << "Eigenvectors: " << eigensolver.eigenvectors();
      //LOG(INFO) << "Max eigenvalue: " << eigensolver.eigenvalues().maxCoeff();
      //LOG(INFO) << "Min eigenvalue: " << eigensolver.eigenvalues().minCoeff();
      //LOG(INFO) << "Condition number: " << eigensolver.eigenvalues().maxCoeff() / eigensolver.eigenvalues().minCoeff();
      timing::TimerImpl it_timer("balm_solve_linear_system");
      dxi = (Hess_full + u * D_full).ldlt().solve(- J_T_r);//- JacT - J_T_r);
      LOG(INFO) << "Solved linear system in " << it_timer.Stop() << " s.";
      CHECK(!dxi.hasNaN());
      for (int j = 0; j < win_size; j++) {
        x_stats_temp[j].getRotation() =
            x_stats[j].getRotation() *
            aslam::Quaternion::exp(dxi.block<3, 1>(balm_error_terms::kFullResidualSize * j, 0));
        x_stats_temp[j].getPosition() =
            x_stats[j].getPosition() + dxi.block<3, 1>(balm_error_terms::kFullResidualSize * j + 12, 0);
        
        // save temporary states
        aslam::Transformation T;
        T.getRotation().toImplementation() = parameters.block<4, 1>(balm_error_terms::kStateSize * j, 0);
        //T.getPosition() = parameters.block<3, 1>(balm_error_terms::kStateSize * j + 4, 0);
        aslam::Transformation T_temp;
        T_temp.getRotation() = T.getRotation() * aslam::Quaternion::exp(dxi.block<3, 1>(balm_error_terms::kFullResidualSize * j, 0));
        //T_temp.getPosition() = T.getPosition() + dxi.block<3, 1>(balm_error_terms::kFullResidualSize * j + 12, 0);

        temporary_states.block<4, 1>(balm_error_terms::kStateSize * j, 0) = T_temp.getRotation().toImplementation().coeffs();
        temporary_states.block<12, 1>(balm_error_terms::kStateSize * j + 4, 0) = dxi.block<12, 1>(balm_error_terms::kFullResidualSize * j + 3, 0) + parameters.block<12, 1>(balm_error_terms::kStateSize * j + 4, 0);
      }
      double q1 = 0.5 * dxi.dot(u * D * dxi - JacT);

      residual_balm = only_residual(x_stats_temp, voxhess);
      balm_error_terms::calculateResiduals(inertial_residual_block, &map, residuals, temporary_states);
      
      residual_inertial = residuals.squaredNorm();
      residual2 = residual_balm + residual_inertial;

      LOG(INFO) << "Residual balm: " << residual_balm;
      LOG(INFO) << "Residual inertial: " << residual_inertial;

      q = (residual1 - residual2);
      printf(
          "iter%d: (%lf %lf) u: %lf v: %.1lf q: %.3lf %lf %lf\n", i, residual1,
          residual2, u, v, q / q1, q1, q);

      if (q > 0) {
        x_stats = x_stats_temp;

        q = q / q1;
        v = 2;
        q = 1 - pow(2 * q - 1, 3);
        constexpr double kOneThird = 1.0 / 3.0;
        u *= (q < kOneThird ? kOneThird : q);
        is_calc_hess = true;
      } else {
        u = u * v;
        v = 2 * v;
        is_calc_hess = false;
      }

      if (fabs(residual1 - residual2) / residual1 < 1e-6)
        break;
    }
  
  }
};

void cut_voxel(
    SurfaceMap& surface_map, const resources::PointCloud& points_S,
    const aslam::Transformation& T_G_S, size_t index, size_t num_scans) {
  resources::PointCloud points_G;
  points_G.appendTransformed(points_S, T_G_S);

  Eigen::Map<const Eigen::Matrix3Xd> xyz_S(
      points_S.xyz.data(), 3, points_S.size());
  Eigen::Map<const Eigen::Matrix3Xd> xyz_G(
      points_G.xyz.data(), 3, points_G.size());

  for (size_t i = 0; i < points_S.size(); ++i) {
    const Eigen::Vector3d pvec_orig = xyz_S.col(i);
    const Eigen::Vector3d pvec_tran = xyz_G.col(i);

    resources::VoxelPosition position(pvec_tran, FLAGS_balm_voxel_size);
    auto iter = surface_map.find(position);
    if (iter != surface_map.end()) {
      if (iter->second->index.back() != index) {
        iter->second->index.emplace_back(index);
        iter->second->vec_orig.emplace_back(PLV(3)());
        iter->second->vec_tran.emplace_back(PLV(3)());
        iter->second->sig_orig.emplace_back(PointCluster());
        iter->second->sig_tran.emplace_back(PointCluster());
      }

      iter->second->vec_orig.back().emplace_back(pvec_orig);
      iter->second->vec_tran.back().emplace_back(pvec_tran);
      iter->second->sig_orig.back().push(pvec_orig);
      iter->second->sig_tran.back().push(pvec_tran);
    } else {
      OctoTreeNode* node = new OctoTreeNode();

      node->index.emplace_back(index);
      node->vec_orig.back().emplace_back(pvec_orig);
      node->vec_tran.back().emplace_back(pvec_tran);
      node->sig_orig.back().push(pvec_orig);
      node->sig_tran.back().push(pvec_tran);

      node->voxel_center[0] = (0.5 + position.x) * FLAGS_balm_voxel_size;
      node->voxel_center[1] = (0.5 + position.y) * FLAGS_balm_voxel_size;
      node->voxel_center[2] = (0.5 + position.z) * FLAGS_balm_voxel_size;
      node->quater_length = FLAGS_balm_voxel_size / 4.0;
      node->layer = 0;

      surface_map[position] = node;
    }
  }
}

#endif  // BALM_H_
