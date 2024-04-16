#include <memory>
#include <Eigen/Core>

#include "dense-reconstruction/balm/bavoxel.h"
#include "dense-reconstruction/dense-reconstruction-plugin.h"
#include "dense-reconstruction/voxblox-params.h"
//#include "dense-reconstruction/balm/inertial-terms.h"

#include <chrono>
#include <cstring>
#include <malloc.h>
#include <string>
#include <vector>

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
#include <visualization/common-rviz-visualization.h>
#include <visualization/rviz-visualization-sink.h>
#include <visualization/viwls-graph-plotter.h>
#include <voxblox/alignment/icp.h>
#include <voxblox/core/tsdf_map.h>
#include <voxblox/integrator/esdf_integrator.h>
#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox/io/mesh_ply.h>
#include <voxblox/mesh/mesh_integrator.h>
#include <voxblox_ros/esdf_server.h>
#include <voxblox_ros/mesh_vis.h>

#include <posegraph/vertex.h>

#include <string>
#include <unordered_set>
#include <vector>

#include <gtest/gtest_prod.h>

#include <aslam/cameras/ncamera.h>
#include <aslam/common/memory.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/frames/visual-nframe.h>
#include <map-resources/resource-common.h>
#include <maplab-common/macros.h>
#include <maplab-common/pose_types.h>
#include <maplab-common/proto-helpers.h>
#include <maplab-common/traits.h>
#include <sensors/absolute-6dof-pose.h>

#include "vi-map/landmark-store.h"
#include "vi-map/landmark.h"
#include "vi-map/mission.h"
#include "vi-map/unique-id.h"
#include "vi-map/vi_map.pb.h"

namespace li_map {

// Forward declaration for Edge class, since Vertex and Edge will reference each other 
class Edge; 

class Vertex {
public:
    MAPLAB_POINTER_TYPEDEFS(Vertex);
    Vertex(const pose_graph::VertexId& id,
            const Eigen::Matrix<double, 6, 1>& imu_biases,
            const aslam::Transformation& pose,
            const int64_t timestamp_ns) :
            id_(id),
            imu_biases_(imu_biases),
            pose_(pose),
            timestamp_ns_(timestamp_ns) {

            }
    const pose_graph::VertexId& getId() const {
        return id_;
    }
    Eigen::Matrix<double, 6, 1>& getImuBiases() const {
        return imu_biases_;
    }
    aslam::Transformation& getPose() const{
        return pose_;
    }
    int64_t getTimestamp() const {
        return timestamp_ns_;
    }

    bool Vertex::addIncomingEdge(const pose_graph::EdgeId& edge) {
        incoming_edges_.insert(edge);
        return true;
    }

    bool Vertex::addOutgoingEdge(const pose_graph::EdgeId& edge) {
        outgoing_edges_.insert(edge);
        return true;
    }
    void Vertex::getOutgoingEdges(pose_graph::EdgeIdSet* edges) const {
        CHECK_NOTNULL(edges);
        *edges = outgoing_edges_;
    }

    void Vertex::getIncomingEdges(pose_graph::EdgeIdSet* edges) const {
        CHECK_NOTNULL(edges);
        *edges = incoming_edges_;
    }
    // method for overwriting the pose of the vertex
    void setPose(const aslam::Transformation& pose){
        pose_ = pose;
    }
    // method for overwriting the biases of the vertex
    void setImuBiases(const Eigen::Matrix<double, 6, 1>& imu_biases){
        imu_biases_ = imu_biases;
    }
    // ... other methods related to vertex management

private:
    pose_graph::VertexId id_;
    Eigen::Matrix<double, 6, 1> imu_biases_;
    aslam::Transformation pose_;
    int64_t timestamp_ns_;
    pose_graph::EdgeIdSet incoming_edges_;
    pose_graph::EdgeIdSet outgoing_edges_;
};

class Edge {
public:
    MAPLAB_POINTER_TYPEDEFS(Edge);
    Edge(const pose_graph::EdgeId& id, 
        const pose_graph::VertexId& from,
        const pose_graph::VertexId& to,
        const Eigen::Matrix<double, 6, Eigen::Dynamic>>& imu_measurements,
        const Eigen::Matrix<double, 1, Eigen::Dynamic>& imu_timestamps) :
        id_(id),
        from_(from),
        to_(to),
        imu_measurements_(imu_measurements),
        imu_timestamps_(imu_timestamps) {
        };

    const pose_graph::EdgeId& id() const {
    return id_;
    }
    const pose_graph::VertexId& from() const {
        return from_;
    }
    const pose_graph::VertexId& to() const {
        return to_;
    }
    const std::vector<Eigen::Matrix<double, 6, 1>>& getImuMeasurements() const {
        return imu_measurements_;
    }
    const std::vector<int64_t>& getImuTimestamps() const {
        return imu_timestamps_;
    }

private:
    const pose_graph::VertexId from_;
    const pose_graph::VertexId to_;
    Eigen::Matrix<double, 1, Eigen::Dynamic> imu_measurements_;
    Eigen::Matrix<int64_t, 1, Eigen::Dynamic> imu_timestamps_;
    pose_graph::EdgeId id_;
};

class LiMap {
public:
    Vertex::UniquePtr getVertex(const pose_graph::VertexId& id) {
        auto it = vertex_id_map_.find(id);
        if (it != vertex_id_map_.end()) {
            return std::move(it->second); // Found!
        } else {
            return nullptr;  // Not found
        }
    }
    Edge::UniquePtr getEdge(const pose_graph::EdgeId& id) {
        auto it = edge_id_map_.find(id);
        if (it != edge_id_map_.end()) {
            return std::move(it->second); // Found!
        } else {
            return nullptr;  // Not found
        }
    }
    void addVertex(Vertex::UniquePtr vertex) {
        const pose_graph::VertexId& id = vertex->getId();
        CHECK(vertex_id_map_.emplace(id, std::move(vertex)).second) 
             << "Vertex with ID " << id << " already exists!";
    } 
    void addEdge(Edge::UniquePtr edge) {
        const pose_graph::EdgeId& id = edge->getId();
        CHECK(edge_id_map_.emplace(id, std::move(edge)).second) 
             << "Edge with ID " << id << " already exists!";
    } 
    // ... other methods for querying and managing the graph
    
private:
    std::vector<Vertex::UniquePtr> vertices_;
    std::vector<Edge::UniquePtr> edges_;
    std::unordered_map<pose_graph::VertexId, Vertex::UniquePtr> vertex_id_map_;
    std::unordered_map<pose_graph::EdgeId, Edge::UniquePtr> edge_id_map_;
};

} // namespace li_map 