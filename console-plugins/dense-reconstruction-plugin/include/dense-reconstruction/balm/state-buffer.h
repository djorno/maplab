#ifndef BALM_OPTIMIZATION_STATE_BUFFER_H_
#define BALM_OPTIMIZATION_STATE_BUFFER_H_

#include <unordered_map>
#include <vector>

#include <aslam/common/memory.h>
#include <aslam/common/unique-id.h>
#include <glog/logging.h>
#include <vi-map/vi-map.h>

#include <Eigen/Core>
#include <aslam/common/unique-id.h>
#include <glog/logging.h>
#include <maplab-common/accessors.h>
#include <vi-map/sensor-manager.h>

namespace balm_error_terms {

inline void ensurePositiveQuaternion(Eigen::Ref<Eigen::Vector4d> quat_xyzw) {
  if (quat_xyzw(3) < 0.0) {
    quat_xyzw = -quat_xyzw;
  }
}

inline bool isValidQuaternion(const Eigen::Quaterniond& quat) {
  constexpr double kEpsilon = 1e-5;
  const double norm = quat.squaredNorm();
  return (quat.w() >= 0.0) && (norm < ((1.0 + kEpsilon) * (1.0 + kEpsilon))) &&
         (norm > ((1.0 - kEpsilon) * (1.0 - kEpsilon)));
}

inline void assertValidQuaternion(const Eigen::Quaterniond& quat) {
  CHECK(isValidQuaternion(quat))
      << "Quaternion: " << quat.coeffs().transpose()
      << " is not valid. (norm=" << quat.norm() << ", w=" << quat.w() << ")";
}

// Buffer for all states that can not be optimized directly on the map. For
// example the rotation of the keyframe pose is stored in a different
// convention than the optimization expects, hence, it is buffered here.
class OptimizationStateBuffer {
 public:
  double* get_vertex_q_IM__M_p_MI_JPL(
        const pose_graph::VertexId& id) {
    //LOG(INFO) << "id in func: " << id.hexString();
    const size_t index = common::getChecked(vertex_id_to_vertex_idx_, id);
    //LOG(INFO) << "index: " << index;
    CHECK_LT(index, static_cast<size_t>(vertex_q_IM__M_p_MI_.cols()));
    return vertex_q_IM__M_p_MI_.col(index).data();
    }

  double* get_baseframe_q_GM__G_p_GM_JPL(
        const vi_map::MissionBaseFrameId& id) {
    const size_t index = common::getChecked(baseframe_id_to_baseframe_idx_, id);
    CHECK_LT(index, static_cast<size_t>(baseframe_q_GM__G_p_GM_.cols()));
    return baseframe_q_GM__G_p_GM_.col(index).data();
    }

  void importKeyframePosesOfMissions(
        const vi_map::VIMap& map, const vi_map::MissionIdSet& mission_ids) {
    pose_graph::VertexIdList all_vertices;
    pose_graph::VertexIdList mission_vertices;
    for (const vi_map::MissionId& mission_id : mission_ids) {
        map.getAllVertexIdsInMissionAlongGraph(mission_id, &mission_vertices);
        all_vertices.insert(
            all_vertices.end(), mission_vertices.begin(), mission_vertices.end());
    }
    vertex_id_to_vertex_idx_.reserve(all_vertices.size());
    vertex_q_IM__M_p_MI_.resize(Eigen::NoChange, all_vertices.size());

    size_t vertex_idx = 0u;
    for (const pose_graph::VertexId& vertex_id : all_vertices) {
        const vi_map::Vertex& ba_vertex = map.getVertex(vertex_id);

        Eigen::Quaterniond q_M_I = ba_vertex.get_q_M_I();
        ensurePositiveQuaternion(q_M_I.coeffs());
        assertValidQuaternion(q_M_I);

        // Convert active Hamiltonian rotation (minkindr, Eigen) to passive JPL
        // which the error terms use. No inverse is required.
        vertex_q_IM__M_p_MI_.col(vertex_idx) << q_M_I.coeffs(),
            ba_vertex.get_p_M_I();
        CHECK(ba_vertex.id().isValid());
        CHECK(vertex_id_to_vertex_idx_.emplace(ba_vertex.id(), vertex_idx).second);
        ++vertex_idx;
    }
    CHECK_EQ(
        static_cast<size_t>(vertex_q_IM__M_p_MI_.cols()),
        vertex_id_to_vertex_idx_.size());
    CHECK_EQ(all_vertices.size(), vertex_id_to_vertex_idx_.size());
    }

  void importBaseframePoseOfMissions(
        const vi_map::VIMap& map, const vi_map::MissionIdSet& mission_ids) {
    baseframe_q_GM__G_p_GM_.resize(Eigen::NoChange, mission_ids.size());
    size_t base_frame_idx = 0u;
    for (const vi_map::MissionId& mission_id : mission_ids) {
        const vi_map::MissionBaseFrame& baseframe =
            map.getMissionBaseFrameForMission(mission_id);

        // Convert active Hamiltonian rotation (minkindr, Eigen) to passive JPL
        // which the error terms use. An inverse is required.
        Eigen::Quaterniond q_G_M_JPL(baseframe.get_q_G_M().inverse().coeffs());
        ensurePositiveQuaternion(q_G_M_JPL.coeffs());
        assertValidQuaternion(q_G_M_JPL);
        baseframe_q_GM__G_p_GM_.col(base_frame_idx) << q_G_M_JPL.coeffs(),
            baseframe.get_p_G_M();

        CHECK(baseframe.id().isValid());
        CHECK(baseframe_id_to_baseframe_idx_.emplace(baseframe.id(), base_frame_idx)
                .second);
        ++base_frame_idx;
    }

    CHECK_EQ(
        static_cast<size_t>(baseframe_q_GM__G_p_GM_.cols()),
        baseframe_id_to_baseframe_idx_.size());
    CHECK_EQ(mission_ids.size(), baseframe_id_to_baseframe_idx_.size());
    }
  void copyAllKeyframePosesBackToMap(
        vi_map::VIMap* map) const {
    CHECK_NOTNULL(map);
    CHECK_EQ(
        static_cast<size_t>(vertex_q_IM__M_p_MI_.cols()),
        vertex_id_to_vertex_idx_.size());
    typedef std::pair<const pose_graph::VertexId, size_t> value_type;
    for (const value_type& vertex_id_idx : vertex_id_to_vertex_idx_) {
        vi_map::Vertex& vertex = map->getVertex(vertex_id_idx.first);
        const size_t vertex_idx = vertex_id_idx.second;
        Eigen::Map<Eigen::Quaterniond> map_q_M_I(vertex.get_q_M_I_Mutable());
        Eigen::Map<Eigen::Vector3d> map_p_M_I(vertex.get_p_M_I_Mutable());

        CHECK_LT(vertex_idx, static_cast<size_t>(vertex_q_IM__M_p_MI_.cols()));

        // Change from JPL passive quaternion used by error terms to active Hamilton
        // quaternion.
        Eigen::Quaterniond q_I_M_JPL;
        q_I_M_JPL.coeffs() = vertex_q_IM__M_p_MI_.col(vertex_idx).head<4>();
        assertValidQuaternion(q_I_M_JPL);

        // I_q_G_JPL is in fact equal to active G_q_I - no inverse is needed.
        map_q_M_I = q_I_M_JPL;
        map_p_M_I = vertex_q_IM__M_p_MI_.col(vertex_idx).tail<3>();
    }
    }
  
  void copyAllBaseframePosesBackToMap(
        vi_map::VIMap* map) const {
    CHECK_NOTNULL(map);

    typedef std::pair<const vi_map::MissionBaseFrameId, size_t> value_type;
    for (const value_type& baseframe_id_idx : baseframe_id_to_baseframe_idx_) {
        vi_map::MissionBaseFrame& baseframe =
            map->getMissionBaseFrame(baseframe_id_idx.first);
        const size_t baseframe_idx = baseframe_id_idx.second;
        CHECK_LT(baseframe_idx, static_cast<size_t>(vertex_q_IM__M_p_MI_.cols()));

        // Change from JPL passive quaternion used by error terms to active
        // quaternion in the system. Inverse needed.
        Eigen::Quaterniond q_G_M_JPL(
            baseframe_q_GM__G_p_GM_.col(baseframe_idx).head<4>().data());
        assertValidQuaternion(q_G_M_JPL);
        baseframe.set_q_G_M(q_G_M_JPL.inverse());
        baseframe.set_p_G_M(baseframe_q_GM__G_p_GM_.col(baseframe_idx).tail<3>());
    }
    }
  
 private:
  // Keyframe poses as a 7d vector: [q_IM_xyzw, M_p_MI]  (passive JPL).
  std::unordered_map<pose_graph::VertexId, size_t> vertex_id_to_vertex_idx_;
  Eigen::Matrix<double, 7, Eigen::Dynamic> vertex_q_IM__M_p_MI_;

  // Mission baseframe poses as a 7d vector: [q_IM_xyzw, M_p_MI] (passive JPL).
  std::unordered_map<vi_map::MissionBaseFrameId, size_t>
      baseframe_id_to_baseframe_idx_;
  Eigen::Matrix<double, 7, Eigen::Dynamic> baseframe_q_GM__G_p_GM_;

  // Camera extrinsics as 7d vector: [q_IC_xyzw, I_p_IC] (passive JPL).
  std::unordered_map<aslam::CameraId, std::vector<aslam::NCameraId>>
      camera_id_to_ncamera_id_;
  std::unordered_map<aslam::CameraId, size_t> camera_id_to_camera_idx_;
  Eigen::Matrix<double, 7, Eigen::Dynamic> camera_q_CI__C_p_CI_;

  // Other sensor extrinsics: [q_BS_xyzw, B_p_BS] (passive_JPL)
  std::unordered_map<aslam::SensorId, size_t> other_sensor_id_to_sensor_idx_;
  Eigen::Matrix<double, 7, Eigen::Dynamic> other_sensor_q_SB__S_p_SB_;
};
}  // namespace balm_error_terms
#endif  // BALM_OPTIMIZATION_STATE_BUFFER_H_
