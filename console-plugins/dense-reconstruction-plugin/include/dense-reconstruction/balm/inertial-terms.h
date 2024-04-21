#ifndef BALM_INERTIAL_H_
#define BALM_INERTIAL_H_

#include "dense-reconstruction/balm/state-buffer.h"
#include "dense-reconstruction/balm/inertial-error-term.h"
#include "dense-reconstruction/balm/common.h"

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

/*#include <ceres-error-terms/block-pose-prior-error-term-v2.h>
#include <ceres-error-terms/inertial-error-term.h>
#include <ceres-error-terms/landmark-common.h>
#include <ceres-error-terms/lidar-error-term.h>
#include <ceres-error-terms/pose-prior-error-term.h>
#include <ceres-error-terms/six-dof-block-pose-error-term-autodiff.h>
#include <ceres-error-terms/six-dof-block-pose-error-term-with-extrinsics-autodiff.h>
#include <ceres-error-terms/visual-error-term-factory.h>
#include <ceres-error-terms/visual-error-term.h>
#include <ceres/ceres.h>
#include <ceres-error-terms/ceres-signal-handler.h>
#include <ceres/iteration_callback.h>*/
#include <landmark-triangulation/pose-interpolator.h>
#include <maplab-common/progress-bar.h>
#include <vi-map-helpers/vi-map-queries.h>
#include <vi-map/landmark-quality-metrics.h>

#include <thread>
#include <vector>
#include <algorithm>



namespace balm_error_terms {

struct ResidualBlock {
    std::shared_ptr<InertialErrorTerm> cost_function;
    std::vector<double*> parameter_blocks;
    pose_graph::EdgeId edge_id;
};

struct ResidualBlockSet{
    std::vector<ResidualBlock> inertial_residual_blocks;
    void addInertialResidualBlock(
        std::shared_ptr<InertialErrorTerm> cost_function,
        std::vector<double*> parameter_blocks,
        pose_graph::EdgeId edge_id) {
        ResidualBlock residual_block;
        residual_block.cost_function = cost_function;
        residual_block.parameter_blocks = parameter_blocks;
        residual_block.edge_id = edge_id;
        inertial_residual_blocks.push_back(residual_block);
    }
    ResidualBlock& getInertialResidualBlocks(size_t i) {
        return inertial_residual_blocks[i];
    }
};

void InertialErrorTerm::IntegrateStateAndCovariance(
    const InertialState& current_state,
    const Eigen::Matrix<int64_t, 1, Eigen::Dynamic>& imu_timestamps,
    const Eigen::Matrix<double, 6, Eigen::Dynamic>& imu_data,
    InertialState* next_state, InertialStateCovariance* phi_accum,
    InertialStateCovariance* Q_accum) const {
  CHECK_NOTNULL(next_state);
  CHECK_NOTNULL(phi_accum);
  CHECK_NOTNULL(Q_accum);

  Eigen::Matrix<double, 2 * kImuReadingSize, 1>
      debiased_imu_readings;
  InertialStateCovariance phi;
  InertialStateCovariance new_phi_accum;
  InertialStateCovariance Q;
  InertialStateCovariance new_Q_accum;

  Q_accum->setZero();
  phi_accum->setIdentity();

  typedef Eigen::Matrix<double, kStateSize, 1>
      InertialStateVector;
  InertialStateVector current_state_vec, next_state_vec;
  current_state_vec = current_state.toVector();

  for (int i = 0; i < imu_data.cols() - 1; ++i) {
    CHECK_GE(imu_timestamps(0, i + 1), imu_timestamps(0, i))
        << "IMU measurements not properly ordered";

    const Eigen::Block<
        InertialStateVector, kGyroBiasBlockSize, 1>
        current_gyro_bias =
            current_state_vec.segment<kGyroBiasBlockSize>(
                kStateGyroBiasOffset);
    const Eigen::Block<
        InertialStateVector, kAccelBiasBlockSize, 1>
        current_accel_bias =
            current_state_vec.segment<kAccelBiasBlockSize>(
                kStateAccelBiasOffset);

    debiased_imu_readings << imu_data.col(i).segment<3>(
                                 kAccelReadingOffset) -
                                 current_accel_bias,
        imu_data.col(i).segment<3>(kGyroReadingOffset) -
            current_gyro_bias,
        imu_data.col(i + 1).segment<3>(kAccelReadingOffset) -
            current_accel_bias,
        imu_data.col(i + 1).segment<3>(kGyroReadingOffset) -
            current_gyro_bias;

    const double delta_time_seconds =
        (imu_timestamps(0, i + 1) - imu_timestamps(0, i)) *
        kNanoSecondsToSeconds;
    integrator_.integrate(
        current_state_vec, debiased_imu_readings, delta_time_seconds,
        &next_state_vec, &phi, &Q);

    current_state_vec = next_state_vec;
    new_Q_accum = phi * (*Q_accum) * phi.transpose() + Q;

    Q_accum->swap(new_Q_accum);
    new_phi_accum = phi * (*phi_accum);
    phi_accum->swap(new_phi_accum);
  }

  *next_state = InertialState::fromVector(next_state_vec);
}

bool InertialErrorTerm::Evaluate(
    const Eigen::Matrix<double, 2 * kStateSize, 1>& parameters, 
    Eigen::Matrix<double, kErrorStateSize, 1>& residuals,
    Eigen::Matrix<double, kErrorStateSize, kUpdateSize>& jacobian_from,
    Eigen::Matrix<double, kErrorStateSize, kUpdateSize>& jacobian_to, bool eval_jac) const {
  enum {
    kIdxPoseFrom,
    kIdxGyroBiasFrom,
    kIdxVelocityFrom,
    kIdxAccBiasFrom,
    kIdxPoseTo,
    kIdxGyroBiasTo,
    kIdxVelocityTo,
    kIdxAccBiasTo
  };
  // extract subvectors from parameters
  if (!eval_jac) {
    LOG(INFO) << "CP 6";
  }
    Eigen::Vector4d q_I_M_from = parameters.block<4, 1>(0, 0);
    Eigen::Vector3d b_g_from = parameters.block<3, 1>(kStateGyroBiasOffset, 0);
    Eigen::Vector3d v_M_I_from = parameters.block<3, 1>(kStateVelocityOffset, 0);
    Eigen::Vector3d b_a_from = parameters.block<3, 1>(kStateAccelBiasOffset, 0);
    Eigen::Vector3d p_M_I_from = parameters.block<3, 1>(kStatePositionOffset, 0);

    Eigen::Vector4d q_I_M_to = parameters.block<4, 1>(kStateSize, 0);
    Eigen::Vector3d b_g_to = parameters.block<3, 1>(kStateSize + kStateGyroBiasOffset, 0);
    Eigen::Vector3d v_M_I_to = parameters.block<3, 1>(kStateSize + kStateVelocityOffset, 0);
    Eigen::Vector3d b_a_to = parameters.block<3, 1>(kStateSize + kStateAccelBiasOffset, 0);
    Eigen::Vector3d p_M_I_to = parameters.block<3, 1>(kStateSize + kStatePositionOffset, 0);
    if (!eval_jac) {
    LOG(INFO) << "CP 7";
  }
    // print parameters with //LOG(INFO)
    //LOG(INFO) << "q_I_M_from: " << q_I_M_from;
    //LOG(INFO) << "b_g_from: " << b_g_from;
    //LOG(INFO) << "v_M_I_from: " << v_M_I_from;
    //LOG(INFO) << "b_a_from: " << b_a_from;
    //LOG(INFO) << "p_M_I_from: " << p_M_I_from;
    /*LOG(INFO) << "q_I_M_to: " << q_I_M_to;
    LOG(INFO) << "b_g_to: " << b_g_to;
    LOG(INFO) << "v_M_I_to: " << v_M_I_to;
    LOG(INFO) << "b_a_to: " << b_a_to;
    LOG(INFO) << "p_M_I_to: " << p_M_I_to;*/

    // Eigen::Map<const Eigen::Vector4d> q_I_M_from(parameters.data());
    // Eigen::Map<const Eigen::Vector3d> b_g_from(parameters.data() +
    // kStateGyroBiasOffset); Eigen::Map<const Eigen::Vector3d>
    // v_M_I_from(parameters.data() + kStateVelocityOffset);
    // Eigen::Map<const Eigen::Vector3d> b_a_from(parameters.data() +
    // kStateAccelBiasOffset); Eigen::Map<const Eigen::Vector3d>
    // p_M_I_from(parameters.data() + kStatePositionOffset);

    // Eigen::Map<const Eigen::Vector4d> q_I_M_to(parameters.data());
    // Eigen::Map<const Eigen::Vector3d> b_g_to(parameters.data() +
    // kStateGyroBiasOffset); Eigen::Map<const Eigen::Vector3d>
    // v_M_I_to(parameters.data() + kStateVelocityOffset);
    // Eigen::Map<const Eigen::Vector3d> b_a_to(parameters.data() +
    // kStateAccelBiasOffset); Eigen::Map<const Eigen::Vector3d>
    // p_M_I_to(parameters.data() + kStatePositionOffset);

    // Integrate the IMU measurements.
    InertialState begin_state;
    begin_state.q_I_M = q_I_M_from;
    begin_state.b_g = b_g_from;
    begin_state.v_M = v_M_I_from;
    begin_state.b_a = b_a_from;
    begin_state.p_M_I = p_M_I_from;
    // Reuse a previous integration if the linearization point hasn't changed.
    const bool cache_is_valid = false; //integration_cache_.valid &&
                                //(integration_cache_.begin_state == begin_state);
if (!eval_jac) {
    LOG(INFO) << "CP 8";
  }
    if (!cache_is_valid) {
      integration_cache_.begin_state = begin_state;
      if (!eval_jac) {
    LOG(INFO) << "CP 9";
  }
      IntegrateStateAndCovariance(
          integration_cache_.begin_state, imu_timestamps_, imu_data_,
          &integration_cache_.end_state, &integration_cache_.phi_accum,
          &integration_cache_.Q_accum);
        if (!eval_jac) {
    LOG(INFO) << "CP 10";
  }

      integration_cache_.L_cholesky_Q_accum.compute(integration_cache_.Q_accum);
      integration_cache_.valid = true;
  }/*
  LOG(INFO) << "q_I_M_to: " << q_I_M_to;
  LOG(INFO) << "q_I_M_to int: " << integration_cache_.end_state.q_I_M;
  LOG(INFO) << "b_g_to: " << b_g_to;
  LOG(INFO) << "b_g_to int: " << integration_cache_.end_state.b_g;
  LOG(INFO) << "v_M_I_to: " << v_M_I_to;
  LOG(INFO) << "v_M_I_to int: " << integration_cache_.end_state.v_M;
  LOG(INFO) << "b_a_to: " << b_a_to;
  LOG(INFO) << "b_a_to int: " << integration_cache_.end_state.b_a;
  LOG(INFO) << "p_M_I_to: " << p_M_I_to;
  LOG(INFO) << "p_M_I_to int: " << integration_cache_.end_state.p_M_I;
  */
  CHECK(integration_cache_.valid);
    if (!eval_jac) {
        LOG(INFO) << "CP 11";
    }

  if (true) {
    Eigen::Quaterniond quaternion_to;
    quaternion_to.coeffs() = q_I_M_to;

    Eigen::Quaterniond quaternion_integrated;
    quaternion_integrated.coeffs() = integration_cache_.end_state.q_I_M;

    if (!eval_jac) {
    LOG(INFO) << "CP 12";
  }

    Eigen::Vector4d delta_q;
    common::positiveQuaternionProductJPL(
        q_I_M_to, quaternion_integrated.inverse().coeffs(), delta_q);
    CHECK_GE(delta_q(3), 0.);

    if (!eval_jac) {
    LOG(INFO) << "CP 13";
  }

    residuals <<
        // While our quaternion representation is Hamilton, underlying memory
        // layout is JPL because of Eigen.
        2. * delta_q.head<3>(),
        b_g_to - integration_cache_.end_state.b_g,
        v_M_I_to - integration_cache_.end_state.v_M,
        b_a_to - integration_cache_.end_state.b_a,
        p_M_I_to - integration_cache_.end_state.p_M_I;

    if (!eval_jac) {
    LOG(INFO) << "CP 14";
  }
    integration_cache_.L_cholesky_Q_accum.matrixL().solveInPlace(residuals);
    if (!eval_jac) {
    LOG(INFO) << "CP 15";
  }
  } else {
    LOG(WARNING)
        << "Skipped residual calculation, since residual pointer was NULL";
  }

  if (eval_jac) {
    if (!cache_is_valid) {
      InertialJacobianType& J_end = integration_cache_.J_end;
      InertialJacobianType& J_begin = integration_cache_.J_begin;

      Eigen::Matrix<double, 4, 3, Eigen::RowMajor> theta_local_begin;
      Eigen::Matrix<double, 4, 3, Eigen::RowMajor> theta_local_end;
      // This is the jacobian lifting the error state to the state. JPL
      // quaternion
      // parameterization is used because our memory layout of quaternions is
      // JPL.
      ceres_error_terms::JplQuaternionParameterization parameterization;
      parameterization.ComputeJacobian(q_I_M_to.data(), theta_local_end.data());
      parameterization.ComputeJacobian(
          q_I_M_from.data(), theta_local_begin.data());

      // Calculate the Jacobian for the end of the edge:
      J_end.setZero();
      J_end = Eigen::Matrix<double, 15, 15>::Identity();
      
      J_begin.setZero();
      J_begin = integration_cache_.phi_accum;

      // Invert and apply by using backsolve.
      integration_cache_.L_cholesky_Q_accum.matrixL().solveInPlace(J_end);
      integration_cache_.L_cholesky_Q_accum.matrixL().solveInPlace(J_begin);
    }

    const InertialJacobianType& J_end = integration_cache_.J_end;
    const InertialJacobianType& J_begin = integration_cache_.J_begin;
    // copy jacobians
    jacobian_from = J_begin;
    jacobian_to = J_end;
  }
  if (!eval_jac) {
    LOG(INFO) << "CP 16";
  }
  return true;
}

int addInertialTermsForEdges(
    vi_map::VIMap* map, ResidualBlockSet& residual_block_set, OptimizationStateBuffer* buffer) {
    CHECK_NOTNULL(map);
    // get mission id
    vi_map::MissionIdList mission_ids;
    // checkpoin 1
    map->getAllMissionIds(&mission_ids);
    CHECK(!mission_ids.empty());
    const vi_map::MissionId& mission_id = mission_ids.front();
    // extract all edges
    pose_graph::EdgeIdList edges;
    map->getAllEdgeIdsInMissionAlongGraph(mission_id, pose_graph::Edge::EdgeType::kViwls, &edges);
    CHECK(!edges.empty());
    const vi_map::Imu& imu_sensor = map->getMissionImu(mission_id);
    const vi_map::ImuSigmas& imu_sigmas = imu_sensor.getImuSigmas();
    // construct optimization_state_buffer
    buffer->importKeyframePosesOfMissions(*map, {mission_id});

    // gravity magnitude
    const double gravity_magnitude = 9.81; // TODO: change later
        
    int num_residuals_added = 0;
    for (pose_graph::EdgeId edge_id : edges) {
        const vi_map::ViwlsEdge& inertial_edge =
            map->getEdgeAs<vi_map::ViwlsEdge>(edge_id);
        std::shared_ptr<InertialErrorTerm> inertial_term_cost(
            new InertialErrorTerm(
                inertial_edge.getImuData(), inertial_edge.getImuTimestamps(),
                imu_sigmas.gyro_noise_density,
                imu_sigmas.gyro_bias_random_walk_noise_density,
                imu_sigmas.acc_noise_density,
                imu_sigmas.acc_bias_random_walk_noise_density, gravity_magnitude));
        vi_map::Vertex& vertex_from = map->getVertex(inertial_edge.from());
        vi_map::Vertex& vertex_to = map->getVertex(inertial_edge.to());
        // vertex pose in JPL quaternion format
        double* vertex_from_q_IM__M_p_MI = buffer->get_vertex_q_IM__M_p_MI_JPL(inertial_edge.from());
        double* vertex_to_q_IM__M_p_MI = buffer->get_vertex_q_IM__M_p_MI_JPL(inertial_edge.to());
        // Add residual block
        residual_block_set.addInertialResidualBlock(
            inertial_term_cost, {vertex_from_q_IM__M_p_MI, vertex_from.getGyroBiasMutable(),
            vertex_from.get_v_M_Mutable(), vertex_from.getAccelBiasMutable(),
            vertex_from_q_IM__M_p_MI, vertex_to.getGyroBiasMutable(),
            vertex_to.get_v_M_Mutable(), vertex_to.getAccelBiasMutable()}, edge_id);
        ++num_residuals_added;
    }
    return num_residuals_added;

}

void allocateMemoryForPointerArrays(std::vector<double**>& jacobians_ptr, 
    std::vector<double*>& residuals_ptr, 
    std::vector<double**>& parameters_ptr, 
    size_t size) {
    enum {
        kIdxPoseFrom,
        kIdxGyroBiasFrom,
        kIdxVelocityFrom,
        kIdxAccBiasFrom,
        kIdxPoseTo,
        kIdxGyroBiasTo,
        kIdxVelocityTo,
        kIdxAccBiasTo
    };
    jacobians_ptr.resize(size);
    residuals_ptr.resize(size);
    parameters_ptr.resize(size);
    for (size_t i = 0; i < size; ++i) {
        jacobians_ptr[i] = new double*[8];
        CHECK(jacobians_ptr[i] != NULL);
        residuals_ptr[i] = new double[kErrorStateSize];
        CHECK(residuals_ptr[i] != NULL);
        parameters_ptr[i] = new double*[8];
        CHECK(parameters_ptr[i] != NULL);
        jacobians_ptr[i][kIdxPoseFrom] = new double[kErrorStateSize * kStatePoseBlockSize];
        CHECK(jacobians_ptr[i][kIdxPoseFrom] != NULL);
        jacobians_ptr[i][kIdxGyroBiasFrom] = new double[kErrorStateSize * kGyroBiasBlockSize];
        CHECK(jacobians_ptr[i][kIdxGyroBiasFrom] != NULL);
        jacobians_ptr[i][kIdxVelocityFrom] = new double[kErrorStateSize * kVelocityBlockSize];
        CHECK(jacobians_ptr[i][kIdxVelocityFrom] != NULL);
        jacobians_ptr[i][kIdxAccBiasFrom] = new double[kErrorStateSize * kAccelBiasBlockSize];
        CHECK(jacobians_ptr[i][kIdxAccBiasFrom] != NULL);
        jacobians_ptr[i][kIdxPoseTo] = new double[kErrorStateSize * kStatePoseBlockSize];
        CHECK(jacobians_ptr[i][kIdxPoseTo] != NULL);
        jacobians_ptr[i][kIdxGyroBiasTo] = new double[kErrorStateSize * kGyroBiasBlockSize];
        CHECK(jacobians_ptr[i][kIdxGyroBiasTo] != NULL);
        jacobians_ptr[i][kIdxVelocityTo] = new double[kErrorStateSize * kVelocityBlockSize];
        CHECK(jacobians_ptr[i][kIdxVelocityTo] != NULL);
        jacobians_ptr[i][kIdxAccBiasTo] = new double[kErrorStateSize * kAccelBiasBlockSize];
        CHECK(jacobians_ptr[i][kIdxAccBiasTo] != NULL);

        parameters_ptr[i][kIdxPoseFrom] = new double[kStatePoseBlockSize];
        CHECK(parameters_ptr[i][kIdxPoseFrom] != NULL);
        parameters_ptr[i][kIdxGyroBiasFrom] = new double[kGyroBiasBlockSize];
        CHECK(parameters_ptr[i][kIdxGyroBiasFrom] != NULL);
        parameters_ptr[i][kIdxVelocityFrom] = new double[kVelocityBlockSize];
        CHECK(parameters_ptr[i][kIdxVelocityFrom] != NULL);
        parameters_ptr[i][kIdxAccBiasFrom] = new double[kAccelBiasBlockSize];
        CHECK(parameters_ptr[i][kIdxAccBiasFrom] != NULL);
        parameters_ptr[i][kIdxPoseTo] = new double[kStatePoseBlockSize];
        CHECK(parameters_ptr[i][kIdxPoseTo] != NULL);
        parameters_ptr[i][kIdxGyroBiasTo] = new double[kGyroBiasBlockSize];
        CHECK(parameters_ptr[i][kIdxGyroBiasTo] != NULL);
        parameters_ptr[i][kIdxVelocityTo] = new double[kVelocityBlockSize];
        CHECK(parameters_ptr[i][kIdxVelocityTo] != NULL);
        parameters_ptr[i][kIdxAccBiasTo] = new double[kAccelBiasBlockSize];
    }
}

void deallocateMemoryForPointerArrays(std::vector<double**>& jacobians_ptr, std::vector<double*>& residuals_ptr, std::vector<double**>& parameters_ptr, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        delete[] jacobians_ptr[i][0];
        delete[] jacobians_ptr[i][1];
        delete[] jacobians_ptr[i][2];
        delete[] jacobians_ptr[i][3];
        delete[] jacobians_ptr[i][4];
        delete[] jacobians_ptr[i][5];
        delete[] jacobians_ptr[i][6];
        delete[] jacobians_ptr[i][7];
        delete[] parameters_ptr[i][0];
        delete[] parameters_ptr[i][1];
        delete[] parameters_ptr[i][2];
        delete[] parameters_ptr[i][3];
        delete[] parameters_ptr[i][4];
        delete[] parameters_ptr[i][5];
        delete[] parameters_ptr[i][6];
        delete[] parameters_ptr[i][7];
        delete[] jacobians_ptr[i];
        delete[] residuals_ptr[i];
        delete[] parameters_ptr[i];
    }
}


void evaluateInertialJacobian(
    ResidualBlockSet& residual_block_set, vi_map::VIMap* map, 
    OptimizationStateBuffer* buffer, 
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& jacobian_full,
    Eigen::Matrix<double, Eigen::Dynamic, 1>& residuals_full,
    Eigen::Matrix<double, Eigen::Dynamic, 1>& parameters_full) {
    
    // check that the size of the matrices is correct
    // extract the number of vertices from the map
    size_t num_vertices = map->numVertices();
    CHECK(num_vertices * kUpdateSize == jacobian_full.cols());
    CHECK(num_vertices * kFullResidualSize == jacobian_full.rows());
    CHECK(num_vertices * kFullResidualSize == residuals_full.rows());
    CHECK(num_vertices * kStateSize == parameters_full.rows());

    for (int i = 1; i < num_vertices; i++) {
        LOG(INFO) << "num_vert = " << num_vertices;
        LOG(INFO) << "i = " << i;
        Eigen::Matrix<double, kErrorStateSize, kUpdateSize> jacobian_from; // = jacobian_full.block<kErrorStateSize, kUpdateSize>(i * kFullResidualSize, (i - 1) * kUpdateSize);
        Eigen::Matrix<double, kErrorStateSize, kUpdateSize> jacobian_to; // = jacobian_full.block<kErrorStateSize, kUpdateSize>(i * kFullResidualSize, i * kUpdateSize);
        
        Eigen::Matrix<double, kErrorStateSize, 1> residuals; // = residuals_full.block<kErrorStateSize, 1>(i * kFullResidualSize, 0);
        Eigen::Matrix<double, 2 * kStateSize, 1> parameters; // = parameters_full.block<2* kStateSize, 1>((i - 1) * kStateSize, 0);
        //std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> parameters(8);
        // get residual block
        const ResidualBlock& residual_block = residual_block_set.getInertialResidualBlocks(i-1); // i - 1 since the iteration starts at 1 and the index starts at 0
        // get edge id
        pose_graph::EdgeId edge_id = residual_block.edge_id;
        // check that edge exists
        CHECK(map->hasEdge(edge_id));
        // get edge
        const vi_map::ViwlsEdge& inertial_edge = map->getEdgeAs<vi_map::ViwlsEdge>(edge_id);
        // get vertex from
        // check that vertex exists
        CHECK(map->hasVertex(inertial_edge.from()));
        vi_map::Vertex& vertex_from = map->getVertex(inertial_edge.from());
        // get vertex to
        CHECK(map->hasVertex(inertial_edge.to()));
        vi_map::Vertex& vertex_to = map->getVertex(inertial_edge.to());
        // vertex parameters from
        double* vertex_from_q_IM__M_p_MI = buffer->get_vertex_q_IM__M_p_MI_JPL(inertial_edge.from());
        double* vertex_from_b_g = vertex_from.getGyroBiasMutable();
        double* vertex_from_v_M = vertex_from.get_v_M_Mutable();
        double* vertex_from_b_a = vertex_from.getAccelBiasMutable();
        // vertex parameters to
        double* vertex_to_q_IM__M_p_MI = buffer->get_vertex_q_IM__M_p_MI_JPL(inertial_edge.to()); 
        double* vertex_to_b_g = vertex_to.getGyroBiasMutable();
        double* vertex_to_v_M = vertex_to.get_v_M_Mutable();
        double* vertex_to_b_a = vertex_to.getAccelBiasMutable();
        // create Eigen::Map
        Eigen::Map<Eigen::Matrix<double, 7, 1>> map_q_IM__M_p_MI_from(vertex_from_q_IM__M_p_MI);
        Eigen::Map<Eigen::Matrix<double, 3, 1>> map_b_g_from(vertex_from_b_g);
        Eigen::Map<Eigen::Matrix<double, 3, 1>> map_v_M_I_from(vertex_from_v_M);
        Eigen::Map<Eigen::Matrix<double, 3, 1>> map_b_a_from(vertex_from_b_a);

        Eigen::Map<Eigen::Matrix<double, 7, 1>> map_q_IM__M_p_MI_to(vertex_to_q_IM__M_p_MI);
        Eigen::Map<Eigen::Matrix<double, 3, 1>> map_b_g_to(vertex_to_b_g);
        Eigen::Map<Eigen::Matrix<double, 3, 1>> map_v_M_I_to(vertex_to_v_M);
        Eigen::Map<Eigen::Matrix<double, 3, 1>> map_b_a_to(vertex_to_b_a);
        // set parameters using eigen block
        parameters.block<kStateOrientationBlockSize, 1>(0, 0) = map_q_IM__M_p_MI_from.block<kStateOrientationBlockSize, 1>(0, 0);
        parameters.block<kGyroBiasBlockSize, 1>(kStateGyroBiasOffset, 0) = map_b_g_from;
        parameters.block<kVelocityBlockSize, 1>(kStateVelocityOffset, 0) = map_v_M_I_from;
        parameters.block<kAccelBiasBlockSize, 1>(kStateAccelBiasOffset, 0) = map_b_a_from;
        parameters.block<kPositionBlockSize, 1>(kStatePositionOffset, 0) = map_q_IM__M_p_MI_from.block<kPositionBlockSize, 1>(kStateOrientationBlockSize, 0);

        parameters.block<kStateOrientationBlockSize, 1>(kStateSize, 0) = map_q_IM__M_p_MI_to.block<kStateOrientationBlockSize, 1>(0, 0);
        parameters.block<kGyroBiasBlockSize, 1>(kStateGyroBiasOffset + kStateSize, 0) = map_b_g_to;
        parameters.block<kVelocityBlockSize, 1>(kStateVelocityOffset + kStateSize, 0) = map_v_M_I_to;
        parameters.block<kAccelBiasBlockSize, 1>(kStateAccelBiasOffset + kStateSize, 0) = map_b_a_to;
        parameters.block<kPositionBlockSize, 1>(kStatePositionOffset + kStateSize, 0) = map_q_IM__M_p_MI_to.block<kPositionBlockSize, 1>(kStateOrientationBlockSize, 0);

        CHECK(parameters.allFinite());
        // insert parameters into parameters_full
        parameters_full.block<2 * kStateSize, 1>((i - 1) * kStateSize, 0) = parameters;

        // evaluate jacobian
        residual_block.cost_function->Evaluate(parameters, residuals, jacobian_from, jacobian_to, true);
        CHECK(jacobian_from.allFinite());
        CHECK(jacobian_to.allFinite());
        CHECK(residuals.allFinite());
        //checkJacobian(jacobian);
        // copy jacobian to full jacobian
        jacobian_full.block<kErrorStateSize, kUpdateSize>(i * kFullResidualSize , (i - 1) * kUpdateSize) = jacobian_from;
        jacobian_full.block<kErrorStateSize, kUpdateSize>(i * kFullResidualSize , i * kUpdateSize) = jacobian_to;
        // copy residuals to full residuals
        residuals_full.block<kErrorStateSize, 1>(i * kErrorStateSize, 0) = residuals;
        // print residuals
        // LOG(INFO) << "Residuals: " << residuals;
        // copy parameters to full parameters
        // parameters_full.block<2 * kStateSize, 1>((i - 1) * kStateSize, 0) = parameters;
    }
}

void calculateResiduals(
    ResidualBlockSet& residual_block_set,
    vi_map::VIMap* map,
    Eigen::Matrix<double, Eigen::Dynamic, 1>& residuals_full, 
    Eigen::Matrix<double, Eigen::Dynamic, 1>& parameters_full) {
    
    LOG(INFO) << "CP 1";
    // check that the size of the matrices is correct
    // extract the number of vertices from the map
    size_t num_vertices = map->numVertices();
    CHECK(num_vertices * kFullResidualSize == residuals_full.rows());
    CHECK(num_vertices * kStateSize == parameters_full.rows());
    LOG(INFO) << "CP 2";
    // create dummy jacobians
    Eigen::Matrix<double, kErrorStateSize, kUpdateSize> jacobian_from;
    jacobian_from.setZero();
    Eigen::Matrix<double, kErrorStateSize, kUpdateSize> jacobian_to;
    jacobian_to.setZero();
    LOG(INFO) << "CP 3";
    for (int i = 1; i < num_vertices; i++) {
        LOG(INFO) << "num_vert = " << num_vertices;
        LOG(INFO) << "i = " << i;
        LOG(INFO) << "CP 4";
        Eigen::Matrix<double, kErrorStateSize, 1> residuals = residuals_full.block<kErrorStateSize, 1>(i * kFullResidualSize, 0);
        Eigen::Matrix<double, 2 * kStateSize, 1> parameters = parameters_full.block<2* kStateSize, 1>((i - 1) * kStateSize, 0);
        //std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> parameters(8);
        // get residual block
        const ResidualBlock& residual_block = residual_block_set.getInertialResidualBlocks(i-1); // i - 1 since the iteration starts at 1 and the index starts at 0
        LOG(INFO) << "CP 5";
        // compute residuals
        residual_block.cost_function->Evaluate(parameters, residuals, jacobian_from, jacobian_to, false);
        LOG(INFO) << "CP 17";
        residuals_full.block<kErrorStateSize, 1>(i * kErrorStateSize, 0) = residuals;
    }
} 

void checkBuffer(OptimizationStateBuffer* buffer, vi_map::VIMap* map) {
    //CHECK(buffer->vertex_q_IM__M_p_MI_JPL_.cols() > 0);
    //CHECK(buffer->vertex_q_IM__M_p_MI_JPL_.rows() == 7);
    //CHECK(buffer->vertex_q_IM__M_p_MI_JPL_.cols() == buffer->vertex_id_to_vertex_idx_.size());

    // check that every vertex in the buffer is in the map
    //for (const pose_graph::VertexId& vertex_id : buffer->vertex_id_to_vertex_idx_) {
    //    CHECK(map->hasVertex(vertex_id));
    //}
    // check that every vertex in the map is in the buffer
    pose_graph::VertexIdList all_vertices;
    map->getAllVertexIds(&all_vertices);
    for (const pose_graph::VertexId& vertex_id : all_vertices) {
        CHECK(buffer->get_vertex_q_IM__M_p_MI_JPL(vertex_id) != nullptr);
    }

} 
    

}  // namespace balm_error_terms
#endif  // BALM_INERTIAL_H_
