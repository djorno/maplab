#ifndef BALM_LI_MAP_H_
#define BALM_LI_MAP_H_

#include <Eigen/Core>
#include <aslam/common/pose-types.h>
#include <gflags/gflags.h>
#include <kindr/minimal/common.h>
#include <maplab-common/threading-helpers.h>
#include <resources-common/point-cloud.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <malloc.h>
#include <string>

#include <aslam/common/timer.h>
#include <console-common/console.h>
#include <dense-reconstruction/conversion-tools.h>
#include <dense-reconstruction/pmvs-file-utils.h>
#include <dense-reconstruction/pmvs-interface.h>
#include <dense-reconstruction/stereo-dense-reconstruction.h>
#include <depth-integration/depth-integration.h>
#include <gflags/gflags.h>
#include <landmark-triangulation/pose-interpolator.h>
#include <map-manager/map-manager.h>
#include <maplab-common/conversions.h>
#include <maplab-common/file-system-tools.h>
#include <vi-map/unique-id.h>
#include <vi-map/vi-map.h>

#include <algorithm>
#include <iterator>
#include <thread>
#include <vector>

// Path:
// console-plugins/dense-reconstruction-plugin/include/dense-reconstruction/balm/inertial-terms.h
namespace li_map {
void addLidarToMap(
    const vi_map::MissionIdList& mission_ids, vi_map::VIMap& vi_map,
    const std::vector<int64_t>& lidar_keyframe_timestamps) {
  // create pose interpolator and interpolate poses
  landmark_triangulation::PoseInterpolator pose_interpolator;
  const Eigen::Matrix<int64_t, 1, Eigen::Dynamic> pose_timestamps =
      Eigen::Map<const Eigen::Matrix<int64_t, 1, Eigen::Dynamic>>(
          lidar_keyframe_timestamps.data(), 1,
          lidar_keyframe_timestamps.size());
  aslam::TransformationVector poses_M_I;
  std::vector<Eigen::Vector3d> velocities_M_I;
  std::vector<Eigen::Vector3d> gyro_biases;
  std::vector<Eigen::Vector3d> accel_biases;
  pose_interpolator.getPosesAtTime(
      vi_map, mission_ids[0], pose_timestamps, &poses_M_I, &velocities_M_I,
      &gyro_biases, &accel_biases);
  // get the root vertex of the map

  const vi_map::MissionId& mission_id = mission_ids[0];
  const pose_graph::Edge::EdgeType edge_type =
      vi_map.getGraphTraversalEdgeType(mission_id);
  const vi_map::VIMission& mission = vi_map.getMission(mission_id);
  const pose_graph::VertexId root_vertex_id = mission.getRootVertexId();
  pose_graph::VertexId current_vertex_id = root_vertex_id;
  pose_graph::VertexId previous_vertex_id = root_vertex_id;
  pose_graph::VertexId lidar_vertex_id;
  pose_graph::EdgeId edge_id;
  vi_map.getNextVertex(current_vertex_id, &current_vertex_id);
  std::unordered_set<pose_graph::EdgeId> connecting_edge_ids;
  // build iterator through the lidar_keyframe_timestamps
  std::vector<int64_t>::const_iterator lidar_timestamp_it =
      lidar_keyframe_timestamps.begin();
  // get the timestamp of the current vertex
  vi_map::Vertex& current_vertex = vi_map.getAnyVertex(current_vertex_id);
  vi_map::Vertex& previous_vertex = vi_map.getAnyVertex(previous_vertex_id);

  int64_t current_vertex_timestamp =
      vi_map.getVertexTimestampNanoseconds(current_vertex_id);
  int64_t previous_vertex_timestamp =
      vi_map.getVertexTimestampNanoseconds(previous_vertex_id);
  // check that the root vertex has a smaller timestamp than the first lidar
  // keyframe
  CHECK(previous_vertex_timestamp < *lidar_timestamp_it);

  aslam::Transformation T_M_I;
  Eigen::Vector3d v_M;

  bool all_lidar_vertices_inserted = false;
  do {
    // get the timestamp of the current vertex
    vi_map::Vertex& current_vertex = vi_map.getAnyVertex(current_vertex_id);
    current_vertex_timestamp =
        vi_map.getVertexTimestampNanoseconds(current_vertex_id);
    vi_map::Vertex& previous_vertex = vi_map.getAnyVertex(previous_vertex_id);
    previous_vertex_timestamp =
        vi_map.getVertexTimestampNanoseconds(previous_vertex_id);
    // check if the lidar timestamp is smaller than the current vertex timestamp
    if (current_vertex_timestamp < *lidar_timestamp_it) {
      // get the next vertex id
      previous_vertex_id = current_vertex_id;
      CHECK(vi_map.getNextVertex(current_vertex_id, &current_vertex_id));
      continue;
    }
    // check if the lidar timestamp is equal to the current vertex timestamp
    if (current_vertex_timestamp == *lidar_timestamp_it) {
      // unlikely case not treated
      // TODO Implement this case
      LOG(ERROR) << "Timestamps are equal, not implemented!";
    } else {
      // case that the lidar timestamp is smaller than the current vertex
      // timestamp
      CHECK(current_vertex_timestamp > *lidar_timestamp_it);
      CHECK(previous_vertex_timestamp < *lidar_timestamp_it);

      // get the edge to split
      previous_vertex.getAnyOutgoingEdges(&connecting_edge_ids);

      CHECK(!connecting_edge_ids.empty())
          << "No outgoing edges from vertex " << previous_vertex_id;
      bool edge_found = false;
      for (const pose_graph::EdgeId& edge_id_lp : connecting_edge_ids) {
        const pose_graph::Edge& edge =
            vi_map.getAnyEdgeAs<pose_graph::Edge>(edge_id_lp);
        if (edge.getType() == edge_type) {
          CHECK(!edge_found)
              << "There is more than one outgoing edge of type '"
              << pose_graph::Edge::edgeTypeToString(edge_type)
              << "' from vertex " << previous_vertex_id
              << "! The map is either inconsistent or this edge type cannot be "
                 "used to traverse the pose graph in a unique way.";
          edge_id = edge_id_lp;
        }
      }
      // get the edge
      vi_map::ViwlsEdge& edge = vi_map.getAnyEdgeAs<vi_map::ViwlsEdge>(edge_id);

      // get the IMU data and timestamps
      Eigen::Matrix<int64_t, 1, Eigen::Dynamic> imu_timestamps =
          edge.getImuTimestamps();
      CHECK(imu_timestamps.cols() > 1) << "No IMU data in edge!";
      // check the extrema of the IMU timestamps
      CHECK(imu_timestamps(0) <= current_vertex_timestamp);
      CHECK(
          imu_timestamps(imu_timestamps.cols() - 1) >=
          current_vertex_timestamp);
      CHECK(imu_timestamps(0) >= previous_vertex_timestamp);
      // check that the IMU timestamps are sorted
      CHECK(std::is_sorted(
          imu_timestamps.data(),
          imu_timestamps.data() + imu_timestamps.cols()));
      Eigen::Matrix<double, 6, Eigen::Dynamic> imu_data = edge.getImuData();
      CHECK(imu_data.cols() > 0) << "No IMU data in edge!";

      // split IMU data at the lidar timestamp
      size_t imu_split_index =
          std::upper_bound(
              imu_timestamps.data(),
              imu_timestamps.data() + imu_timestamps.cols(),
              *lidar_timestamp_it) -
          imu_timestamps.data();

      CHECK(imu_split_index < imu_timestamps.cols())
          << "Timestamp not found in IMU data!";
      CHECK(imu_split_index > 0);
      size_t new_vertex_index =
          std::distance(lidar_keyframe_timestamps.begin(), lidar_timestamp_it);

      // create lidar vertex
      Eigen::Matrix<double, 6, 1> imu_ba_bw;
      imu_ba_bw.block(0, 0, 3, 1) = accel_biases[new_vertex_index];
      imu_ba_bw.block(3, 0, 3, 1) = gyro_biases[new_vertex_index];
      T_M_I = poses_M_I[new_vertex_index];
      v_M = velocities_M_I[new_vertex_index];

      CHECK(imu_ba_bw.allFinite());
      CHECK(accel_biases[new_vertex_index].isApprox(
          previous_vertex.getAccelBias()))
          << "Accel bias not equal! "
          << accel_biases[new_vertex_index].transpose() << " vs. "
          << previous_vertex.getAccelBias().transpose();
      CHECK(
          gyro_biases[new_vertex_index].isApprox(previous_vertex.getGyroBias()))
          << "Gyro bias not equal! "
          << gyro_biases[new_vertex_index].transpose() << " vs. "
          << previous_vertex.getGyroBias().transpose();

      lidar_vertex_id = aslam::createRandomId<pose_graph::VertexId>();
      vi_map::LidarVertex::UniquePtr lidar_vertex_ptr =
          aligned_unique<vi_map::LidarVertex>(
              lidar_vertex_id, imu_ba_bw, mission_id);
      vi_map::LidarVertex& lidar_vertex = *lidar_vertex_ptr;
      lidar_vertex.set_T_M_I(T_M_I);
      lidar_vertex.set_v_M(v_M);
      // make vertex unique pointer
      vi_map.addLidarVertex(std::move(lidar_vertex_ptr));

      // create incoming edge to the new vertex
      Eigen::Matrix<double, 6, Eigen::Dynamic> imu_data_incoming;
      imu_data_incoming.resize(6, imu_split_index + 1);
      imu_data_incoming.setConstant(Eigen::NumTraits<double>::quiet_NaN());
      imu_data_incoming.block(0, 0, 6, imu_split_index) =
          imu_data.block(0, 0, 6, imu_split_index);
      imu_data_incoming.col(imu_split_index) =
          imu_data.col(imu_split_index - 1);

      Eigen::Matrix<int64_t, 1, Eigen::Dynamic> imu_timestamps_incoming;
      imu_timestamps_incoming.resize(1, imu_split_index + 1);
      imu_timestamps_incoming.setConstant(
          Eigen::NumTraits<int64_t>::quiet_NaN());
      imu_timestamps_incoming.block(0, 0, 1, imu_split_index) =
          imu_timestamps.block(0, 0, 1, imu_split_index);
      imu_timestamps_incoming.col(imu_split_index) =
          Eigen::Matrix<int64_t, 1, 1>(*lidar_timestamp_it);

      CHECK(imu_data_incoming.allFinite());
      CHECK(imu_data_incoming.col(imu_split_index)
                .isApprox(imu_data.col(imu_split_index - 1)));
      CHECK(imu_data_incoming.cols() > 1);
      CHECK(imu_timestamps_incoming.cols() > 1);
      CHECK(imu_data_incoming.allFinite());
      CHECK(imu_timestamps_incoming.allFinite());
      CHECK(imu_data_incoming.cols() > 1);
      CHECK(std::is_sorted(
          imu_timestamps_incoming.data(),
          imu_timestamps_incoming.data() + imu_timestamps_incoming.cols()));
      CHECK(imu_timestamps_incoming(0) == previous_vertex_timestamp)
          << "Prev: " << previous_vertex_timestamp << " vs. "
          << imu_timestamps_incoming(0);
      CHECK(
          imu_timestamps_incoming(imu_timestamps_incoming.cols() - 1) ==
          *lidar_timestamp_it)
          << "Lidar: " << *lidar_timestamp_it << " vs. "
          << imu_timestamps_incoming(imu_timestamps_incoming.cols() - 1)
          << " vs "
          << imu_timestamps_incoming(imu_timestamps_incoming.cols() - 2);

      // create incoming edge to the new vertex
      pose_graph::EdgeId incoming_edge_id =
          aslam::createRandomId<pose_graph::EdgeId>();
      vi_map::ViwlsEdge::UniquePtr incoming_edge =
          aligned_unique<vi_map::ViwlsEdge>(
              incoming_edge_id, previous_vertex_id, lidar_vertex_id,
              imu_timestamps_incoming, imu_data_incoming);

      vi_map.addLidarEdge(std::move(incoming_edge));

      // create outgoing edge from the new vertex
      // interpolate the imu measurements at the lidar timestamp
      Eigen::Matrix<double, 6, 1> imu_data_interpolated =
          imu_data.block(0, imu_split_index, 6, 1);
      // create imu_data matrix with the interpolated imu data
      Eigen::Matrix<double, 6, Eigen::Dynamic> imu_data_outgoing;
      imu_data_outgoing.resize(6, imu_data.cols() - imu_split_index + 1);
      imu_data_outgoing.setConstant(Eigen::NumTraits<double>::quiet_NaN());
      imu_data_outgoing.block(0, 0, 6, 1) = imu_data_interpolated;
      imu_data_outgoing.block(0, 1, 6, imu_data.cols() - imu_split_index) =
          imu_data.block(
              0, imu_split_index, 6, imu_data.cols() - imu_split_index);

      // same with timestamps
      Eigen::Matrix<int64_t, 1, Eigen::Dynamic> imu_timestamps_outgoing;
      imu_timestamps_outgoing.resize(
          1, imu_timestamps.cols() - imu_split_index + 1);
      imu_timestamps_outgoing.setConstant(
          Eigen::NumTraits<int64_t>::quiet_NaN());
      imu_timestamps_outgoing.block(0, 0, 1, 1) =
          Eigen::Matrix<int64_t, 1, 1>(*lidar_timestamp_it);
      imu_timestamps_outgoing.block(
          0, 1, 1, imu_timestamps.cols() - imu_split_index) =
          imu_timestamps.block(
              0, imu_split_index, 1, imu_timestamps.cols() - imu_split_index);

      CHECK(imu_timestamps_outgoing.cols() > 1);
      CHECK(imu_timestamps_outgoing.cols() == imu_data_outgoing.cols());
      CHECK(imu_data_outgoing.cols() > 1);
      CHECK(imu_data_outgoing.allFinite());
      CHECK(imu_timestamps_outgoing.allFinite());
      CHECK(imu_timestamps_outgoing(0) == *lidar_timestamp_it);
      CHECK(
          imu_timestamps_outgoing(imu_timestamps_outgoing.cols() - 1) ==
          current_vertex_timestamp);
      CHECK(std::is_sorted(
          imu_timestamps_outgoing.data(),
          imu_timestamps_outgoing.data() + imu_timestamps_outgoing.cols()));
      CHECK(
          imu_timestamps_outgoing.cols() + imu_timestamps_incoming.cols() - 2 ==
          imu_timestamps.cols());
      CHECK(imu_timestamps_outgoing(0) == *lidar_timestamp_it)
          << "Lidar: " << *lidar_timestamp_it << " vs. "
          << imu_timestamps_outgoing(0);
      CHECK(
          imu_timestamps_outgoing(imu_timestamps_outgoing.cols() - 1) ==
          current_vertex_timestamp);

      pose_graph::EdgeId outgoing_edge_id =
          aslam::createRandomId<pose_graph::EdgeId>();
      vi_map::ViwlsEdge::UniquePtr outgoing_edge =
          aligned_unique<vi_map::ViwlsEdge>(
              outgoing_edge_id, lidar_vertex_id, current_vertex_id,
              imu_timestamps_outgoing, imu_data_outgoing);
      // add edge to map
      vi_map.addLidarEdge(std::move(outgoing_edge));

      // link edges to vertices
      vi_map::Vertex& previous_vertex = vi_map.getAnyVertex(previous_vertex_id);
      // check that the previous vertex does not have any outgoing lidar edges
      if (previous_vertex.hasOutgoingLidarEdges()) {
        pose_graph::EdgeIdSet previous_vertex_outgoing_lidar_edges;
        previous_vertex.getOutgoingLidarEdges(
            &previous_vertex_outgoing_lidar_edges);
        CHECK_EQ(previous_vertex_outgoing_lidar_edges.size(), 1)
            << "There should be exactly one outgoing lidar edge from the "
               "previous vertex!";
        previous_vertex.removeOutgoingLidarEdge(
            *previous_vertex_outgoing_lidar_edges.begin());
      }

      CHECK(!previous_vertex.hasOutgoingLidarEdges());

      // set the outgoing lidar edge
      previous_vertex.addOutgoingLidarEdge(incoming_edge_id);
      // set the incoming lidar edge
      lidar_vertex.addIncomingLidarEdge(incoming_edge_id);
      // get the current vertex
      vi_map::Vertex& current_vertex = vi_map.getAnyVertex(current_vertex_id);
      // check that the current vertex does not have any incoming lidar edges
      if (current_vertex.hasIncomingLidarEdges()) {
        pose_graph::EdgeIdSet current_vertex_incoming_lidar_edges;
        current_vertex.getIncomingLidarEdges(
            &current_vertex_incoming_lidar_edges);
        CHECK_EQ(current_vertex_incoming_lidar_edges.size(), 1)
            << "There should be exactly one incoming lidar edge from the "
               "current vertex!";
        current_vertex.removeIncomingLidarEdge(
            *current_vertex_incoming_lidar_edges.begin());
      }
      CHECK(!current_vertex.hasIncomingLidarEdges());
      // set the incoming lidar edge
      current_vertex.addIncomingLidarEdge(outgoing_edge_id);
      // set the outgoing lidar edge
      lidar_vertex.addOutgoingLidarEdge(outgoing_edge_id);
      // update the previous id to the lidar_vertex_id
      previous_vertex_id = lidar_vertex_id;
      // step forward in the lidar keyframe timestamps
      std::advance(lidar_timestamp_it, 1);
      // check if all lidar vertices have been inserted
      if (lidar_timestamp_it == lidar_keyframe_timestamps.end()) {
        all_lidar_vertices_inserted = true;
      }
    }
  } while (!all_lidar_vertices_inserted);
}

}  // namespace li_map

#endif  // BALM_LI_MAP_H_