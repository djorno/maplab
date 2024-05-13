#ifndef BALM_LI_MAP_H_
#define BALM_LI_MAP_H_

#include "dense-reconstruction/balm/inertial-error-term.h"
#include "dense-reconstruction/balm/inertial-terms.h"

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

#include <thread>
#include <vector>
#include <algorithm>
#include <iterator>


// Path: console-plugins/dense-reconstruction-plugin/include/dense-reconstruction/balm/inertial-terms.h

    void addLidarToMap(const vi_map::MissionIdList& mission_ids, 
            vi_map::VIMapManager::MapWriteAccess& vi_map, 
            const aslam::TransformationVector& poses_M_I, 
            const std::vector<Eigen::Vector3d> keyframe_velocities, 
            const std::vector<Eigen::Vector3d> keyframe_gyro_bias, 
            const std::vector<Eigen::Vector3d> keyframe_acc_bias, 
            const std::vector<int64_t>& lidar_keyframe_timestamps){
        // get the root vertex of the map
        const vi_map::VIMission& mission = getMission(mission_id);
        const pose_graph::VertexId root_vertex_id = mission.getRootVertexId();
        pose_graph::VertexId current_vertex_id;
        pose_graph::VertexId previous_vertex_id = root_vertex_id;
        std::unordered_set<pose_graph::EdgeId> = connecting_edge_ids;
        // build iterator through the lidar_keyframe_timestamps
        std::vector<int64_t>::iterator lidar_timestamp_it = lidar_keyframe_timestamps.begin();
        // get the timestamp of the current vertex
        vi_map::Vertex& current_vertex = vi_map.getVertex(current_vertex_id);
        int64_t current_vertex_timestamp = current_vertex.getMinTimestampNanoseconds();
        // check that the root vertex has a smaller timestamp than the first lidar keyframe
        CHECK(current_vertex_timestamp < *lidar_timestamp_it);

        
        bool all_lidar_vertices_inserted = false;
        do {
            // get the timestamp of the current vertex
            vi_map::Vertex& current_vertex = vi_map.getVertex(current_vertex_id);
            int64_t current_vertex_timestamp = current_vertex.getMinTimestampNanoseconds();
            // check if the lidar timestamp is smaller than the current vertex timestamp
            if (current_vertex_timestamp < *lidar_timestamp_it) {
                // get the next vertex id
                previous_vertex_id = current_vertex_id;
                CHECK(getNextVertex(current_vertex_id, &current_vertex_id));
                continue;
            }
            // check if the lidar timestamp is equal to the current vertex timestamp
            if (current_vertex_timestamp == *lidar_timestamp_it) {
                // unlikely case not treated
                // TODO Implement this case
                LOG(ERROR) << "Timestamps are equal, not implemented!";
            }
            else {
                // case that the lidar timestamp is smaller than the current vertex timestamp
                CHECK(current_vertex_timestamp > *lidar_timestamp_it);
                CHECK(previous_vertex_timestamp < *lidar_timestamp_it);
                
                // get the edge to split
                previous_vertex_id.getOutgoingEdges(&connecting_edge_ids);
                CHECK_EQ(connecting_edge_ids.size(), 1) << "There should be exactly one outgoing edge from the previous vertex!";
                pose_graph::EdgeId edge_id = *connecting_edge_ids.begin();
                // get the edge
                vi_map::ViwlsEdge& edge = vi_map.getEdgeAs<vi_map::ViwlsEdge>(edge_id);

                // get the IMU data and timestamps
                Eigen::Matrix<int64_t, 1, Eigen::Dynamic> imu_timestamps = edge.getImuTimestamps();
                Eigen::Matrix<double, 6, Eigen::Dynamic> imu_data = edge.getImuData();

                // split IMU data at the lidar timestamp
                size_t split_index = 0;
                auto it = std::lower_bound(imu_timestamps.begin(), imu_timestamps.end(), *lidar_timestamp_it);
                CHECK(it != imu_timestamps.end()) << "Timestamp not found in IMU data!";

                int imu_split_index = std::distance(imu_timestamps.begin(), it);
                int new_vertex_index = std::distance(lidar_keyframe_timestamps.begin(), lidar_timestamp_it);

                // create lidar vertex
                Eigen::Matrix<double, 6, 1> imu_ba_bw << keyframe_acc_bias[new_vertex_index], keyframe_gyro_bias[new_vertex_index];
                aslam::Transformation T_G_M = poses_M_I[new_vertex_index];
                Eigen::Vector3d v_M = keyframe_velocities[new_vertex_index];
                pose_graph::VertexId lidar_vertex_id = aslam::createRandomId<pose_graph::VertexId>();
                vi_map::LidarVertex::UniquePtr lidar_vertex = aligned_unique<vi_map::LidarVertex>(vertex_id, imu_ba_bw, mission_id);
                vertex->set_T_G_M(T_G_M);
                vertex->setVelocity(v_M);
                vi_map->addLidarVertex(std::move(vertex));

                // create incoming edge to the new vertex
                pose_graph::EdgeId incoming_edge_id = aslam::createRandomId<pose_graph::EdgeId>();
                vi_map::ViwlsEdge::UniquePtr incoming_edge = aligned_unique<vi_map::ViwlsEdge>(incoming_edge_id, previous_vertex_id, lidar_vertex_id, imu_timestamps.block(0, 0, 1, imu_split_index), imu_data.block(0, 0, 6, imu_split_index));
                vi_map->addEdge(std::move(incoming_edge));

                // create outgoing edge from the new vertex
                // interpolate the imu measurements at the lidar timestamp
                Eigen::Matrix<double, 6, 1> imu_data_interpolated = imu_data.block(0, imu_split_index, 6, 1);
                // create imu_data matrix with the interpolated imu data
                Eigen::Matrix<double, 6, Eigen::Dynamic> imu_data_new;
                imu_data_outgoing.resize(6, imu_data.cols() - imu_split_index);
                imu_data_outgoing.block(0, 0, 6, 1) = imu_data_interpolated;
                imu_data_outgoing.block(0, 1, 6, imu_data.cols() - imu_split_index - 1) = imu_data.block(0, imu_split_index, 6, imu_data.cols() - imu_split_index - 1);
                // same with timestamps
                Eigen::Matrix<int64_t, 1, Eigen::Dynamic> imu_timestamps_new;
                imu_timestamps_outgoing.resize(1, imu_timestamps.cols() - imu_split_index);
                imu_timestamps_outgoing.block(0, 0, 1, 1) = Eigen::Matrix<int64_t, 1, 1>(*lidar_timestamp_it);
                imu_timestamps_outgoing.block(0, 1, 1, imu_timestamps.cols() - imu_split_index - 1) = imu_timestamps.block(0, imu_split_index, 1, imu_timestamps.cols() - imu_split_index - 1);
                pose_graph::EdgeId outgoing_edge_id = aslam::createRandomId<pose_graph::EdgeId>();
                vi_map::ViwlsEdge::UniquePtr outgoing_edge = aligned_unique<vi_map::ViwlsEdge>(outgoing_edge_id, lidar_vertex_id, current_vertex_id, imu_timestamps_outgoing, imu_data_outgoing);

                // add edge to map
                vi_map->addEdge(std::move(outgoing_edge));

                // link edges to vertices
                // get the previous vertex
                vi_map::Vertex& previous_vertex = vi_map.getVertex(previous_vertex_id);
                // check that the previous vertex does not have any outgoing lidar edges
                if (previous_vertex.hasOutgoingLidarEdges()) {
                    pose_graph::EdgeIdSet previous_vertex_outgoing_lidar_edges;
                    previous_vertex.getOutgoingLidarEdges(&previous_vertex_outgoing_lidar_edges);
                    CHECK_EQ(previous_vertex_outgoing_lidar_edges.size(), 1) << "There should be exactly one outgoing lidar edge from the previous vertex!";
                    previous_vertex.removeOutgoingLidarEdge(*previous_vertex_outgoing_lidar_edges.begin());
                }

                CHECK(!previous_vertex.hasOutgoingLidarEdged();)
                
                // set the outgoing lidar edge
                previous_vertex.addOutgoingLidarEdge(incoming_edge_id);
                // set the incoming lidar edge
                lidar_vertex.addIncomingLidarEdge(incoming_edge_id);

                // get the current vertex
                vi_map::Vertex& current_vertex = vi_map.getVertex(current_vertex_id);
                // check that the current vertex does not have any incoming lidar edges
                if (current_vertex.hasIncomingLidarEdges()) {
                    pose_graph::EdgeIdSet current_vertex_incoming_lidar_edges;
                    current_vertex.getIncomingLidarEdges(&current_vertex_incoming_lidar_edges);
                    CHECK_EQ(current_vertex_incoming_lidar_edges.size(), 1) << "There should be exactly one incoming lidar edge from the current vertex!";
                    current_vertex.removeIncomingLidarEdge(*current_vertex_incoming_lidar_edges.begin());
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

#endif  // BALM_LI_MAP_H_