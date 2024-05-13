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
namespace li_map {
    void findFirstLargerTimestampIndices(const std::vector<int64_t>& target_timestamps,
                                            const std::vector<int64_t>& source_timestamps, std::vector<size_t>& closest_indices) {
        // This function takes two vectors of timestamps and finds the first timestamp in the target_timestamps vector which is larger than a corresponding timestamp in the source_vector.
        // Returns a vector of indices of the first larger timestamps in the target_timestamps vector for each timestamp in the source_vector.

        // check if closest indices is the correct size
        CHECK_EQ(source_timestamps.size(), closest_indices.size());
        size_t current_vi_index = 0;

        for (size_t i = 0; i < source_timestamps.size(); ++i) {
            auto it = std::lower_bound(target_timestamps.begin() + current_vi_index, target_timestamps.end(), source_timestamps[i]);

            // Check if 'it' points to the end (no timestamp in target_timestamps is greater or equal)
            if (it == target_timestamps.end()) {
                // Handle the case where source_timestamps[i] is larger than all target_timestamps
                closest_indices[i] = target_timestamps.size(); // Index past the last element !! gives an error if inserted directly
            } else { 
                // Consider both the previous and current timestamp
                closest_indices[i] = it - target_timestamps.begin();
            }
            current_vi_index = it - target_timestamps.begin();
        }
    }

    void buildVerticesAndEdges(const Eigen::Matrix<double, 6, Eigen::Dynamic>& bias_data, 
                            const Eigen::Matrix<double, 6, Eigen::Dynamic>& imu_data,
                            const std::vector<int64_t>& vi_vertex_timestamps, 
                            const std::vector<int64_t>& lidar_vertex_timestamps,
                            const std::vector<int64_t>& imu_timestamps,
                            const Eigen::Matrix<int64_t, 1, Eigen::Dynamic>& imu_timestamps_mat,
                            const aslam::TransformationVector& poses_M_I, 
                            const std::vector<Eigen::Vector3d>& keyframe_velocities,
                            const std::vector<Eigen::Vector3d>& keyframe_gyro_bias,
                            const std::vector<Eigen::Vector3d>& keyframe_acc_bias,
                            const vi_map::MissionId& mission_id, 
                            vi_map::VIMap& li_map, 
                            vi_map::VIMapManager::MapWriteAccess& vi_map) {
        // This function interpolates the bias estimates to the LiDAR keyframe timestamps
        // The bias estimates are interpolated using linear interpolation
        // The bias estimates are stored in a 6xM matrix where M is the total number of VI vertices
        // The timestamps of the VI vertices are stored in a 1xM vector
        // The timestamps of the LiDAR keyframes are stored in a 1xN vector
        // The interpolated bias estimates are stored in a 6xN matrix where N is the total number of LiDAR keyframes
        const unsigned int num_lidar_vertices = lidar_vertex_timestamps.size();

        // check if the dimensions of the bias data and timestamps are correct
        CHECK_EQ(bias_data.cols(), vi_vertex_timestamps.size());
        // check if the bias data and timestamps are not empty
        CHECK_GT(bias_data.cols(), 0);
        CHECK_GT(vi_vertex_timestamps.size(), 0);
        CHECK_GT(num_lidar_vertices, 0);

        // extract a reference aslam::VisualNFrame::Ptr visual_n_frame from the first vertex
        pose_graph::VertexIdList vi_vertex_ids;
        vi_map->getAllVertexIdsInMissionAlongGraph(mission_id, &vi_vertex_ids);

        // get the closest indices of the VI vertices to the LiDAR keyframe timestamps using findFirstLargerTimestampIndices
        std::vector<size_t> closest_indices_vert(num_lidar_vertices);
        findFirstLargerTimestampIndices(vi_vertex_timestamps, lidar_vertex_timestamps, closest_indices_vert);
        // get the closest indices of the IMU data points to the LiDAR keyframe timestamps using findFirstLargerTimestampIndices
        std::vector<size_t> closest_indices_imu(num_lidar_vertices);
        findFirstLargerTimestampIndices(imu_timestamps, lidar_vertex_timestamps, closest_indices_imu);

        // print the num_lidar_vertices
        std::vector<pose_graph::VertexId>* vertex_ids = new std::vector<pose_graph::VertexId>;
        vertex_ids->resize(num_lidar_vertices);

        Eigen::Matrix<double, 6, 1> bias_data_interpolated;
        Eigen::Vector3d velocity_data_interpolated;
        Eigen::Matrix<double, 6, 1> imu_data_interpolated;
        Eigen::Matrix<double, 6, 1> imu_data_interpolated_prev;

        // iterate over all LiDAR keyframe timestamps
        for (size_t i = 0; i < num_lidar_vertices; ++i) {
            // get the closest indices of the VI vertices to the LiDAR keyframe timestamps
            size_t closest_index_vert = closest_indices_vert[i];
            size_t closest_index_imu = closest_indices_imu[i];
            // get the LiDAR keyframe timestamp
            int64_t timestamp_lidar = lidar_vertex_timestamps[i];
            // check edge cases
            CHECK_GE(closest_index_vert, 0);
            CHECK_GE(closest_index_imu, 0);
            CHECK_LT(closest_index_vert, vi_vertex_timestamps.size());
            CHECK_LT(closest_index_imu, imu_timestamps.size());

            // Interpolate the IMU data using linear interpolation
            int64_t timestamp_prev_imu = imu_timestamps[closest_index_imu - 1];
            int64_t timestamp_next_imu = imu_timestamps[closest_index_imu];
            // get the previous and current IMU data
            Eigen::Matrix<double, 6, 1> imu_data_prev = imu_data.block(0, closest_index_imu - 1, 6, 1);
            Eigen::Matrix<double, 6, 1> imu_data_next = imu_data.block(0, closest_index_imu, 6, 1);

            CHECK_GE(timestamp_lidar, timestamp_prev_imu);
            CHECK_LE(timestamp_lidar, timestamp_next_imu);
            CHECK(!(timestamp_lidar == timestamp_prev_imu)) << "Timestamps are equal, should not be possible!";
            
            if (timestamp_lidar == timestamp_next_imu) {
                imu_data_interpolated = imu_data_next; // duplicates are not critical
            }
            else {
                double timestamp_lidar_double = static_cast<double>(timestamp_lidar);
                double timestamp_prev_imu_double = static_cast<double>(timestamp_prev_imu);
                double timestamp_next_imu_double = static_cast<double>(timestamp_next_imu);
                double alpha = (timestamp_lidar_double - timestamp_prev_imu_double) / (timestamp_next_imu_double - timestamp_prev_imu_double);
                CHECK(alpha > 0.0);
                CHECK(alpha < 1.0);
                //imu_data_interpolated = imu_data_prev + (imu_data_next - imu_data_prev) * alpha;
                imu_data_interpolated = imu_data_next;
                //imu_data_interpolated = imu_data_prev + (imu_data_next - imu_data_prev) * (timestamp_lidar_double - timestamp_prev_imu_double) / (timestamp_next_imu_double - timestamp_prev_imu_double);
            }

            // Interpolate the bias estimates using linear interpolation
            // get the previous and next bias estimates
            Eigen::Matrix<double, 6, 1> bias_data_prev = bias_data.block(0, closest_index_vert - 1, 6, 1);
            Eigen::Matrix<double, 6, 1> bias_data_next = bias_data.block(0, closest_index_vert, 6, 1);
            // get the previous and next timestamps
            int64_t timestamp_prev = vi_vertex_timestamps[closest_index_vert - 1];
            int64_t timestamp_next = vi_vertex_timestamps[closest_index_vert];
            // interpolate the bias estimates using linear interpolation
            CHECK_GE(timestamp_lidar, timestamp_prev);
            CHECK_LE(timestamp_lidar, timestamp_next);
            CHECK(!(timestamp_lidar == timestamp_prev)) << "Timestamps are equal, should not be possible!";

            if (timestamp_lidar == timestamp_next) {
                bias_data_interpolated = bias_data_next;
                //bias_data_interpolated << keyframe_acc_bias[i], keyframe_gyro_bias[i];
                velocity_data_interpolated = keyframe_velocities[i];
                
            }
            else {
                // caste timestamps to double
                double timestamp_lidar_double = static_cast<double>(timestamp_lidar);
                double timestamp_prev_double = static_cast<double>(timestamp_prev);
                double timestamp_next_double = static_cast<double>(timestamp_next);
                bias_data_interpolated = bias_data_prev + (bias_data_next - bias_data_prev) * (timestamp_lidar_double - timestamp_prev_double) / (timestamp_next_double - timestamp_prev_double);
                //bias_data_interpolated << keyframe_acc_bias[i], keyframe_gyro_bias[i];
                //LOG(INFO) << "Bias data prev: " << bias_data_prev;
                //LOG(INFO) << "Bias data next: " << bias_data_next;
                //LOG(INFO) << "Bias data interpolated: " << bias_data_interpolated;
                //bias_data_interpolated << keyframe_gyro_bias[i], keyframe_acc_bias[i];
                //LOG(INFO) << "Bias data from int: " << bias_data_interpolated;
                // Interpolate the velocity estimates using linear interpolation
                velocity_data_interpolated = keyframe_velocities[i];
            }
            if (bias_data_interpolated.norm() > 1) {
                LOG(WARNING) << "Bias data issue at i: " << i << ", bias = " << bias_data_interpolated.transpose() << " at timestamp: " << timestamp_lidar;
            }
            if (velocity_data_interpolated.norm() > 1e1) {
                LOG(WARNING) << "Velocity data issue at i: " << i << ", v = " << velocity_data_interpolated.transpose() << " at timestamp: " << timestamp_lidar;
            }
            // create a new vertex
            pose_graph::VertexId vertex_id = aslam::createRandomId<pose_graph::VertexId>();
            vi_map::Vertex::UniquePtr vertex = aligned_unique<vi_map::Vertex>(vertex_id, bias_data_interpolated, velocity_data_interpolated, mission_id, poses_M_I[i]);
            // add the vertex to the LiMap
            li_map.addVertex(std::move(vertex));
            // save the vertex id
            (*vertex_ids)[i] = vertex_id;
            // set the root vertex id if this is the first vertex
            if (i == 0) {
                li_map.getMission(mission_id).setRootVertexId(vertex_id);
            }
            else {
                // add an edge between the current vertex and the previous vertex
                pose_graph::EdgeId edge_id;
                aslam::generateId(&edge_id);
                // extract relevant IMU data and timestamps
                Eigen::Matrix<double, 6, Eigen::Dynamic> edge_imu_data;
                Eigen::Matrix<int64_t, 1, Eigen::Dynamic> edge_imu_timestamps;
                int range = closest_indices_imu[i] - closest_indices_imu[i - 1] + 1;
                edge_imu_data.resize(6, range);
                edge_imu_timestamps.resize(1, range);
                edge_imu_data.block(0, 0, 6, 1) = imu_data_interpolated_prev;
                edge_imu_data.block(0, 1, 6, range - 1) = imu_data.block(0, closest_indices_imu[i - 1], 6, range - 1);
                edge_imu_timestamps.block(0, 0, 1, 1) = Eigen::Matrix<int64_t, 1, 1>(lidar_vertex_timestamps[i - 1]); 
                edge_imu_timestamps.block(0, 1, 1, range - 1) = imu_timestamps_mat.block(0, closest_indices_imu[i - 1], 1, range - 1);

                // create a new edge
                vi_map::Edge* edge(new vi_map::ViwlsEdge(
                edge_id, (*vertex_ids)[i - 1], (*vertex_ids)[i], edge_imu_timestamps,
                edge_imu_data));
                li_map.addEdge(vi_map::Edge::UniquePtr(edge));
            }
            imu_data_interpolated_prev = imu_data_interpolated;
        }
    }

    void buildLIMap(const vi_map::MissionIdList& mission_ids, 
            vi_map::VIMapManager::MapWriteAccess& vi_map, 
            const aslam::TransformationVector& poses_M_I, 
            const std::vector<Eigen::Vector3d> keyframe_velocities, 
            const std::vector<Eigen::Vector3d> keyframe_gyro_bias, 
            const std::vector<Eigen::Vector3d> keyframe_acc_bias, 
            const std::vector<int64_t>& lidar_keyframe_timestamps, 
            vi_map::VIMap& li_map){
        // IMU data extraction and preparation for BALM
        pose_graph::EdgeIdList vi_edges;
        vi_map->getAllEdgeIdsInMissionAlongGraph(
            mission_ids[0], pose_graph::Edge::EdgeType::kViwls, &vi_edges);

        const vi_map::Imu& imu_sensor = li_map.getMissionImu(mission_ids[0]);
        const vi_map::ImuSigmas& imu_sigmas = imu_sensor.getImuSigmas();

        // count the total amount of imu data points 
        int num_imu_data = 0;
        int num_vertices = 1; // start at 1 to count the last vertex as well
        for (const pose_graph::EdgeId edge_id : vi_edges) {
          const vi_map::ViwlsEdge& inertial_edge =
              vi_map->getEdgeAs<vi_map::ViwlsEdge>(edge_id);

          num_imu_data += inertial_edge.getImuData().cols();
          num_vertices++;
        }
        // print total number of imu data points
        LOG(INFO) << "Total number of IMU data points: " << num_imu_data;

        // allocate memory for imu data and timestamps
        Eigen::Matrix<double, 6, Eigen::Dynamic> imu_data;
        Eigen::Matrix<int64_t, 1, Eigen::Dynamic> imu_timestamps_mat;
        // std::vector<int64_t> imu_timestamps;

        // allocate memory for gyro and acc bias estimates
        Eigen::Matrix<double, 6, Eigen::Dynamic> bias_data;
        std::vector<int64_t> vi_vertex_timestamps;

        // set dimensions of imu data, timestamps, gyro and acc bias estimates
        imu_data.resize(6, num_imu_data);
        bias_data.resize(6, num_vertices); // 3 for acc and 3 for gyro bias

        imu_timestamps_mat.resize(1, num_imu_data);


        // iterate over all edges and extract imu data and timestamps
        num_imu_data = 0;
        num_vertices = 0;
        for (const pose_graph::EdgeId edge_id : vi_edges) {
            const vi_map::ViwlsEdge& inertial_edge =
              vi_map->getEdgeAs<vi_map::ViwlsEdge>(edge_id);
            // Extract IMU data and concatenate it into a matrix using Eigen block
            int num_cols = inertial_edge.getImuData().cols();
            CHECK_GT(num_cols, 0);
            imu_data.block(0, num_imu_data, 6, num_cols) = inertial_edge.getImuData();
            // Extract IMU timestamps and concatenate it into a matrix using Eigen block
            imu_timestamps_mat.block(0, num_imu_data, 1, num_cols) = inertial_edge.getImuTimestamps();
            num_imu_data += num_cols;
            // if we are in the first iteration, we need to extract the initial gyro and acc bias estimates
            if (!num_vertices) {
                // Extract gyro and acc bias estimates and concatenate it into a matrix using Eigen block
                vi_map::Vertex& vertex = vi_map->getVertex(inertial_edge.from());
                bias_data.block(0, num_vertices, 3, 1) = vertex.getAccelBias();
                bias_data.block(3, num_vertices, 3, 1) = vertex.getGyroBias();
                // Extract vertex timestamps and concatenate it using emplace_back
                vi_vertex_timestamps.emplace_back(vertex.getMinTimestampNanoseconds());
                num_vertices++;
            }
            // Extract gyro and acc bias estimates and concatenate it into a matrix using Eigen block
            vi_map::Vertex& vertex = vi_map->getVertex(inertial_edge.to());
            bias_data.block(0, num_vertices, 3, 1) = vertex.getAccelBias();
            bias_data.block(3, num_vertices, 3, 1) = vertex.getGyroBias();
            // Extract vertex timestamps and concatenate it using emplace_back
            vi_vertex_timestamps.emplace_back(vertex.getMinTimestampNanoseconds());
            num_vertices++;
        }
        // 
        // Create the std::vector
        std::vector<int64_t> imu_timestamps(imu_timestamps_mat.data(), imu_timestamps_mat.data() + imu_timestamps_mat.rows() * imu_timestamps_mat.cols()); 
        /* Data is now in the formats: 
        (N is the total number of IMU data points, M is the total number of VI vertices)
        imu_data: 6xN: accx accy accz gyrox gyroy gyroz
        imu_timestamps: 1xN: nanosecond timestamp
        bias_data: 6xM: accx accy accz gyrox gyroy gyroz
        vi_vertex_timestamps: 1xM: nanosecond timestamp
        */ 

       // sync IMU data to LiDAR data
        buildVerticesAndEdges(bias_data, 
                imu_data, 
                vi_vertex_timestamps, 
                lidar_keyframe_timestamps, 
                imu_timestamps, imu_timestamps_mat, 
                poses_M_I, keyframe_velocities, 
                keyframe_gyro_bias, keyframe_acc_bias, 
                mission_ids[0], li_map, vi_map);
    }
    void addLidarToMap(const vi_map::MissionIdList& mission_ids, 
            vi_map::VIMap& vi_map, 
            const aslam::TransformationVector& poses_M_I, 
            const std::vector<Eigen::Vector3d> keyframe_velocities, 
            const std::vector<Eigen::Vector3d> keyframe_gyro_bias, 
            const std::vector<Eigen::Vector3d> keyframe_acc_bias, 
            const std::vector<int64_t>& lidar_keyframe_timestamps){
        // get the root vertex of the map
        
        const vi_map::MissionId& mission_id = mission_ids[0];
        const pose_graph::Edge::EdgeType edge_type = vi_map.getGraphTraversalEdgeType(mission_id);
        const vi_map::VIMission& mission = vi_map.getMission(mission_id);
        const pose_graph::VertexId root_vertex_id = mission.getRootVertexId();
        pose_graph::VertexId current_vertex_id = root_vertex_id;
        pose_graph::VertexId previous_vertex_id = root_vertex_id;
        pose_graph::VertexId lidar_vertex_id;
        pose_graph::EdgeId edge_id;
        vi_map.getNextVertex(current_vertex_id, &current_vertex_id);
        std::unordered_set<pose_graph::EdgeId> connecting_edge_ids;
        // build iterator through the lidar_keyframe_timestamps
        std::vector<int64_t>::const_iterator lidar_timestamp_it = lidar_keyframe_timestamps.begin();
        // get the timestamp of the current vertex
        vi_map::Vertex& current_vertex = vi_map.getVertex(current_vertex_id);
        vi_map::Vertex& previous_vertex = vi_map.getVertex(previous_vertex_id);

        int64_t current_vertex_timestamp = current_vertex.getMinTimestampNanoseconds();
        int64_t previous_vertex_timestamp = previous_vertex.getMinTimestampNanoseconds();
        // check that the root vertex has a smaller timestamp than the first lidar keyframe
        CHECK(previous_vertex_timestamp < *lidar_timestamp_it);

        aslam::Transformation T_M_I;
        Eigen::Vector3d v_M;

        
        bool all_lidar_vertices_inserted = false;
        do {
            // get the timestamp of the current vertex
            vi_map::Vertex& current_vertex = vi_map.getAnyVertex(current_vertex_id); 
            current_vertex_timestamp = vi_map.getVertexTimestampNanoseconds(current_vertex_id);
            vi_map::Vertex& previous_vertex = vi_map.getAnyVertex(previous_vertex_id);
            previous_vertex_timestamp = vi_map.getVertexTimestampNanoseconds(previous_vertex_id);
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
            }
            else {
                // case that the lidar timestamp is smaller than the current vertex timestamp
                CHECK(current_vertex_timestamp > *lidar_timestamp_it);
                CHECK(previous_vertex_timestamp < *lidar_timestamp_it);
                
                // get the edge to split
                previous_vertex.getOutgoingEdges(&connecting_edge_ids);
                CHECK(!connecting_edge_ids.empty()) << "No outgoing edges from vertex " << previous_vertex_id;
                bool edge_found = false;
                for (const pose_graph::EdgeId& edge_id_lp : connecting_edge_ids) {
                    const pose_graph::Edge& edge = vi_map.getEdgeAs<pose_graph::Edge>(edge_id_lp);
                    if (edge.getType() == edge_type) {
                        CHECK(!edge_found)
                            << "There is more than one outgoing edge of type '"
                            << pose_graph::Edge::edgeTypeToString(edge_type) << "' from vertex "
                            << previous_vertex_id
                            << "! The map is either inconsistent or this edge type cannot be "
                                "used to traverse the pose graph in a unique way.";
                        edge_id = edge_id_lp;
                    }
                }
                // get the edge
                vi_map::ViwlsEdge& edge = vi_map.getEdgeAs<vi_map::ViwlsEdge>(edge_id);

                // get the IMU data and timestamps
                Eigen::Matrix<int64_t, 1, Eigen::Dynamic> imu_timestamps = edge.getImuTimestamps();
                Eigen::Matrix<double, 6, Eigen::Dynamic> imu_data = edge.getImuData();

                // split IMU data at the lidar timestamp
                size_t imu_split_index = std::upper_bound(imu_timestamps.data(), 
                                      imu_timestamps.data() + imu_timestamps.cols(),
                                      *lidar_timestamp_it) - imu_timestamps.data();
                CHECK(imu_split_index < imu_timestamps.cols()) << "Timestamp not found in IMU data!";
                size_t new_vertex_index = std::distance(lidar_keyframe_timestamps.begin(), lidar_timestamp_it);
                // create lidar vertex
                Eigen::Matrix<double, 6, 1> imu_ba_bw;
                imu_ba_bw.block(0, 0, 3, 1) = keyframe_acc_bias[new_vertex_index];
                imu_ba_bw.block(2, 0, 3, 1) = keyframe_gyro_bias[new_vertex_index];
                T_M_I = poses_M_I[new_vertex_index];
                v_M = keyframe_velocities[new_vertex_index];
                lidar_vertex_id = aslam::createRandomId<pose_graph::VertexId>();
                vi_map::LidarVertex::UniquePtr lidar_vertex_ptr = aligned_unique<vi_map::LidarVertex>(lidar_vertex_id, imu_ba_bw, mission_id);
                vi_map::LidarVertex& lidar_vertex = *lidar_vertex_ptr;
                lidar_vertex.set_T_M_I(T_M_I);
                lidar_vertex.set_v_M(v_M);
                // make vertex unique pointer
                vi_map.addLidarVertex(std::move(lidar_vertex_ptr));

                // create incoming edge to the new vertex
                pose_graph::EdgeId incoming_edge_id = aslam::createRandomId<pose_graph::EdgeId>();
                vi_map::ViwlsEdge::UniquePtr incoming_edge = aligned_unique<vi_map::ViwlsEdge>(incoming_edge_id, previous_vertex_id, lidar_vertex_id, imu_timestamps.block(0, 0, 1, imu_split_index), imu_data.block(0, 0, 6, imu_split_index));
                vi_map.addLidarEdge(std::move(incoming_edge));

                // create outgoing edge from the new vertex
                // interpolate the imu measurements at the lidar timestamp
                Eigen::Matrix<double, 6, 1> imu_data_interpolated = imu_data.block(0, imu_split_index, 6, 1);
                // create imu_data matrix with the interpolated imu data
                Eigen::Matrix<double, 6, Eigen::Dynamic> imu_data_outgoing;
                imu_data_outgoing.resize(6, imu_data.cols() - imu_split_index);
                imu_data_outgoing.block(0, 0, 6, 1) = imu_data_interpolated;
                imu_data_outgoing.block(0, 1, 6, imu_data.cols() - imu_split_index - 1) = imu_data.block(0, imu_split_index, 6, imu_data.cols() - imu_split_index - 1);
                // same with timestamps
                Eigen::Matrix<int64_t, 1, Eigen::Dynamic> imu_timestamps_outgoing;
                imu_timestamps_outgoing.resize(1, imu_timestamps.cols() - imu_split_index);
                imu_timestamps_outgoing.block(0, 0, 1, 1) = Eigen::Matrix<int64_t, 1, 1>(*lidar_timestamp_it);
                imu_timestamps_outgoing.block(0, 1, 1, imu_timestamps.cols() - imu_split_index - 1) = imu_timestamps.block(0, imu_split_index, 1, imu_timestamps.cols() - imu_split_index - 1);
                pose_graph::EdgeId outgoing_edge_id = aslam::createRandomId<pose_graph::EdgeId>();
                vi_map::ViwlsEdge::UniquePtr outgoing_edge = aligned_unique<vi_map::ViwlsEdge>(outgoing_edge_id, lidar_vertex_id, current_vertex_id, imu_timestamps_outgoing, imu_data_outgoing);
                // add edge to map
                vi_map.addLidarEdge(std::move(outgoing_edge));

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

                CHECK(!previous_vertex.hasOutgoingLidarEdges());
                
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

}  // namespace inertial

#endif  // BALM_LI_MAP_H_