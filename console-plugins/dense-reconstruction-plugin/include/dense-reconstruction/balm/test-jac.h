#ifndef BALM_TEST_H_
#define BALM_TEST_H_

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

namespace balm_error_terms {
    void checkParameterizationQuaternionJPL(OptimizationStateBuffer& state_buffer, pose_graph::VertexIdList& ids, vi_map::VIMap* map) {
        for (const pose_graph::VertexId& id : ids) {
            double* q_IM__M_p_MI = state_buffer.get_vertex_q_IM__M_p_MI_JPL(id);
            Eigen::Map<Eigen::Vector4d> quat(q_IM__M_p_MI);
            ensurePositiveQuaternion(quat);
            //Eigen::Quaterniond quat_eigen(quat);
            //assertValidQuaternion(quat_eigen);

            Eigen::Matrix<double, 4, 3, Eigen::RowMajor> theta_local_begin;
            ceres_error_terms::JplQuaternionParameterization parameterization;
            parameterization.ComputeJacobian(quat.data(), theta_local_begin.data());

            LOG(INFO) << "Original quaternion from buffer: " << quat.transpose();
            //get the vertex
            vi_map::Vertex& vertex = map->getVertex(id);
            const aslam::Transformation& T_M_I = vertex.get_T_M_I();
            // get the rotation elements from T_G_M
            Eigen::Quaterniond quat_T_M_I = T_M_I.getRotation().toImplementation();
            LOG(INFO) << "Original quaternion from T_M_I: " << quat_T_M_I.coeffs().transpose();

            LOG(INFO) << "Theta local begin: " << theta_local_begin.transpose();

            Eigen::Vector3d mapped_vector = theta_local_begin.transpose() * quat;
            LOG(INFO) << "Mapped vector: " << mapped_vector.transpose();
            Eigen::Vector3d unlifted_vector = aslam::Quaternion::log(T_M_I.getRotation());
            LOG(INFO) << "Unlifted vector: " << unlifted_vector.transpose();
        }
    }



} // namespace balm_error_terms

#endif // BALM_TEST_H_