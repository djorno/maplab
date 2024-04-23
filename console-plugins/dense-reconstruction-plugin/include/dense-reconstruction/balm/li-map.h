#ifndef BALM_LI_MAP_H_
#define BALM_LI_MAP_H_

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
                            const Eigen::Matrix<double, 3, Eigen::Dynamic>& velocity_data,
                            const std::vector<int64_t>& vi_vertex_timestamps, 
                            const std::vector<int64_t>& lidar_vertex_timestamps,
                            const std::vector<int64_t>& imu_timestamps,
                            const Eigen::Matrix<int64_t, 1, Eigen::Dynamic>& imu_timestamps_mat,
                            const aslam::TransformationVector& poses_G_S, const vi_map::MissionId& mission_id, 
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
        CHECK_GT(velocity_data.cols(), 0);
        CHECK_GT(vi_vertex_timestamps.size(), 0);
        CHECK_GT(num_lidar_vertices, 0);

        // extract a reference aslam::VisualNFrame::Ptr visual_n_frame from the first vertex
        pose_graph::VertexIdList vi_vertex_ids;
        vi_map->getAllVertexIdsInMissionAlongGraph(mission_id, &vi_vertex_ids);
        aslam::NCamera::ConstPtr n_cameras = vi_map->getVertex(*vi_vertex_ids.begin()).getNCameras();
        // create a visual n frame
        // Create a non-const shared_ptr 
        //std::shared_ptr<aslam::NCamera> n_cameras_shared(n_cameras); 

        // Create the VisualNFrame object
        //aslam::VisualNFrame::Ptr visual_n_frame(new aslam::VisualNFrame(n_cameras_shared));
        // aslam::VisualNFrame::Ptr visual_n_frame(new aslam::VisualNFrame(n_cameras));
        // CHECK(visual_n_frame != nullptr);


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

            // check edge cases
            CHECK_GT(closest_index_vert, 0);
            CHECK_GT(closest_index_imu, 0);
            CHECK_LT(closest_index_vert, vi_vertex_timestamps.size());
            CHECK_LT(closest_index_imu, imu_timestamps.size());

            // Interpolate the bias estimates using linear interpolation
            // get the previous and current bias estimates
            Eigen::Matrix<double, 6, 1> bias_data_prev = bias_data.block(0, closest_index_vert - 1, 6, 1);
            Eigen::Matrix<double, 6, 1> bias_data_next = bias_data.block(0, closest_index_vert, 6, 1);
            // get previous and current velocity estimates
            Eigen::Matrix<double, 3, 1> velocity_data_prev = velocity_data.block(0, closest_index_vert - 1, 3, 1);
            Eigen::Matrix<double, 3, 1> velocity_data_next = velocity_data.block(0, closest_index_vert, 3, 1);
            // get the previous and current timestamps
            int64_t timestamp_prev = vi_vertex_timestamps[closest_index_vert - 1];
            int64_t timestamp_next = vi_vertex_timestamps[closest_index_vert];
            // get the LiDAR keyframe timestamp
            int64_t timestamp_lidar = lidar_vertex_timestamps[i];
            // interpolate the bias estimates using linear interpolation
            CHECK_GE(timestamp_lidar, timestamp_prev);
            CHECK_LE(timestamp_lidar, timestamp_next);
            if (timestamp_lidar == timestamp_prev) {
                bias_data_interpolated = bias_data_prev;
                velocity_data_interpolated = velocity_data_prev;
            }
            else if (timestamp_lidar == timestamp_next) {
                bias_data_interpolated = bias_data_next;
                velocity_data_interpolated = velocity_data_next;
            }
            else {
                // caste timestamps to double
                double timestamp_lidar_double = static_cast<double>(timestamp_lidar);
                double timestamp_prev_double = static_cast<double>(timestamp_prev);
                double timestamp_next_double = static_cast<double>(timestamp_next);
                bias_data_interpolated = bias_data_prev + (bias_data_next - bias_data_prev) * (timestamp_lidar_double - timestamp_prev_double) / (timestamp_next_double - timestamp_prev_double);
                // Interpolate the velocity estimates using linear interpolation
                velocity_data_interpolated = velocity_data_prev + (velocity_data_next - velocity_data_prev) * (timestamp_lidar_double - timestamp_prev_double) / (timestamp_next_double - timestamp_prev_double);
            }
            // Interpolate the IMU data using linear interpolation
            int64_t timestamp_prev_imu = imu_timestamps[closest_index_imu - 1];
            int64_t timestamp_next_imu = imu_timestamps[closest_index_imu];
            // get the previous and current IMU data
            Eigen::Matrix<double, 6, 1> imu_data_prev = imu_data.block(0, closest_index_imu - 1, 6, 1);
            Eigen::Matrix<double, 6, 1> imu_data_next = imu_data.block(0, closest_index_imu, 6, 1);
            
            if (timestamp_lidar == timestamp_prev_imu) {
                imu_data_interpolated = imu_data_prev;
            }
            else if (timestamp_lidar == timestamp_next_imu) {
                imu_data_interpolated = imu_data_next;
            }
            else {
                double timestamp_lidar_double = static_cast<double>(timestamp_lidar);
                double timestamp_prev_imu_double = static_cast<double>(timestamp_prev_imu);
                double timestamp_next_imu_double = static_cast<double>(timestamp_next_imu);
                imu_data_interpolated = imu_data_prev + (imu_data_next - imu_data_prev) * (timestamp_lidar_double - timestamp_prev_imu_double) / (timestamp_next_imu_double - timestamp_prev_imu_double);
            }
            // create a new vertex
            pose_graph::VertexId vertex_id = aslam::createRandomId<pose_graph::VertexId>();
            vi_map::Vertex::UniquePtr vertex = aligned_unique<vi_map::Vertex>(vertex_id, bias_data_interpolated, velocity_data_interpolated, mission_id, poses_G_S[i]);//, n_cameras);
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
                vi_map::Edge* edge(new vi_map::ViwlsEdge(
                edge_id, (*vertex_ids)[i - 1], (*vertex_ids)[i], edge_imu_timestamps,
                edge_imu_data));
                li_map.addEdge(vi_map::Edge::UniquePtr(edge));

                //vi_map::ViwlsEdge::UniquePtr edge = aligned_unique<vi_map::ViwlsEdge>(edge_id, (*vertex_ids)[i - 1], (*vertex_ids)[i], edge_imu_timestamp, edge_imu_data);
                //li_map.addEdge(std::move(edge));
            }
            imu_data_interpolated_prev = imu_data_interpolated;
        }
    }

    void buildLIMap(const vi_map::MissionIdList& mission_ids, vi_map::VIMapManager::MapWriteAccess& vi_map, const aslam::TransformationVector& poses_G_S, const std::vector<int64_t>& lidar_keyframe_timestamps, vi_map::VIMap& li_map){
        // IMU data extraction and preparation for BALM
        pose_graph::EdgeIdList vi_edges;
        vi_map->getAllEdgeIdsInMissionAlongGraph(
            mission_ids[0], pose_graph::Edge::EdgeType::kViwls, &vi_edges);

        const vi_map::Imu& imu_sensor = li_map.getMissionImu(mission_ids[0]);
        const vi_map::ImuSigmas& imu_sigmas = imu_sensor.getImuSigmas();        

        // const double gravity_magnitude = 9.81;

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
        Eigen::Matrix<double, 3, Eigen::Dynamic> velocity_data;
        std::vector<int64_t> vi_vertex_timestamps;

        // set dimensions of imu data, timestamps, gyro and acc bias estimates
        imu_data.resize(6, num_imu_data);
        bias_data.resize(6, num_vertices); // 3 for acc and 3 for gyro bias
        velocity_data.resize(3, num_vertices); // 3 for velocity

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

        //   (inertial_edge.getImuData(), // 6xN matrix: accx accy accz gyrox gyroy gyroz
        //           inertial_edge.getImuTimestamps(), // Nx vector nanosecond timestamp
        //           imu_sigmas.gyro_noise_density,
        //           imu_sigmas.gyro_bias_random_walk_noise_density,
        //           imu_sigmas.acc_noise_density,
        //           imu_sigmas.acc_bias_random_walk_noise_density,
        //           gravity_magnitude);
            // if we are in the first iteration, we need to extract the initial gyro and acc bias estimates
            if (!num_vertices) {
                // Extract gyro and acc bias estimates and concatenate it into a matrix using Eigen block
                vi_map::Vertex& vertex_from = vi_map->getVertex(inertial_edge.from());
                bias_data.block(0, num_vertices, 3, 1) = vertex_from.getAccelBias();
                bias_data.block(3, num_vertices, 3, 1) = vertex_from.getGyroBias();
                velocity_data.block(0, num_vertices, 3, 1) = vertex_from.get_v_M();
                // Extract vertex timestamps and concatenate it using emplace_back
                vi_vertex_timestamps.emplace_back(vertex_from.getMinTimestampNanoseconds());
                num_vertices++;
            }
            // Extract gyro and acc bias estimates and concatenate it into a matrix using Eigen block
            vi_map::Vertex& vertex_to = vi_map->getVertex(inertial_edge.to());
            bias_data.block(0, num_vertices, 3, 1) = vertex_to.getAccelBias();
            bias_data.block(3, num_vertices, 3, 1) = vertex_to.getGyroBias();
            velocity_data.block(0, num_vertices, 3, 1) = vertex_to.get_v_M();
            // Extract vertex timestamps and concatenate it using emplace_back
            vi_vertex_timestamps.emplace_back(vertex_to.getMinTimestampNanoseconds());
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
       /*
       Steps:
       1. Find the VI vertex that is closest in time to the LiDAR keyframe
       2. Interpolate the Bias estimates to the LiDAR keyframe
       3. Find the IMU data points that are closest in time to the LiDAR keyframe
       4. Interpolate the IMU data to the LiDAR keyframe
       */
        // 1. Find the VI vertex that is closest in time to the LiDAR keyframe
        std::vector<size_t> closest_indices(lidar_keyframe_timestamps.size());
        findFirstLargerTimestampIndices(vi_vertex_timestamps, lidar_keyframe_timestamps, closest_indices);
        // 2. Interpolate the Bias estimates to the LiDAR keyframe
        Eigen::Matrix<double, 6, Eigen::Dynamic> interpolated_bias;
        //Eigen::Matrix<double, 3, Eigen::Dynamic> interpolated_acc_bias;
        interpolated_bias.resize(6, lidar_keyframe_timestamps.size());
        //interpolated_acc_bias.resize(3, lidar_keyframe_timestamps.size());
        buildVerticesAndEdges(bias_data, imu_data, velocity_data, vi_vertex_timestamps, lidar_keyframe_timestamps, imu_timestamps, imu_timestamps_mat, poses_G_S, mission_ids[0], li_map, vi_map);
    }

}  // namespace inertial

#endif  // BALM_LI_MAP_H_