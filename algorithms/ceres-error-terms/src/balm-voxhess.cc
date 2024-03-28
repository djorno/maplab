#include "ceres-error-terms/balm-voxhess.h"

#include <chrono>
#include <cstring>
#include <malloc.h>
#include <string>
#include <vector>

#include <aslam/common/timer.h>
#include <dense-reconstruction/stereo-dense-reconstruction.h>
#include <depth-integration/depth-integration.h>
#include <gflags/gflags.h>
#include <map-manager/map-manager.h>
#include <maplab-common/conversions.h>
#include <maplab-common/file-system-tools.h>
#include <vi-map/unique-id.h>
#include <vi-map/vi-map.h>
#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Geometry/Quaternion.h"
#include "maplab-common/quaternion-math.h"
#include "posegraph/unique-id.h"

DEFINE_bool(
    ba_dense_depth_map_reprojection_use_undistorted_camera, false,
    "If enabled, the depth map reprojection assumes that the map has "
    "been created using the undistorted camera model. Therefore, the no "
    "distortion is used during reprojection.");

DEFINE_string(
    ba_dense_result_mesh_output_file, "",
    "Path to the PLY mesh file that is generated from the "
    "reconstruction command.");

DEFINE_string(
    ba_dense_image_export_path, "",
    "Export folder for image export function. console command: "
    "export_timestamped_images");

DEFINE_int32(
    ba_dense_depth_resource_output_type, 17,
    "Output resource type of the dense reconstruction algorithms."
    "Supported commands: "
    "stereo_dense_reconstruction "
    "Supported types: "
    "PointCloudXYZRGBN = 17, RawDepthMap = 8");

DEFINE_int32(
    ba_dense_depth_resource_input_type, 21,
    "Input resource type of the dense reconstruction algorithms."
    "Supported commands: "
    "create_tsdf_from_depth_resource "
    "Supported types: "
    "RawDepthMap = 8, OptimizedDepthMap = 9, PointCloudXYZ = 16, "
    "PointCloudXYZRGBN = 17, kPointCloudXYZI = 21");

DEFINE_double(
    ba_balm_kf_distance_threshold_m, 0.1,
    "BALM distance threshold to add a new keyframe [m].");
DEFINE_double(
    ba_balm_kf_rotation_threshold_deg, 1,
    "BALM rotation threshold to add a new keyframe [deg].");
DEFINE_double(
    ba_balm_kf_time_threshold_s, 0.1,
    "BALM force a keyframe at fixed time intervals [s].");

DEFINE_double(
    ba_balm_voxel_size, 1.0, "BALM voxel size to use to look for planes in.");
DEFINE_uint32(
    ba_balm_max_layers, 3,
    "BALM maximum number of subdividing of a voxel when looking for a plane.");
DEFINE_uint32(
    ba_balm_min_plane_points, 15,
    "BALM minimum number of points needed when looking for a plane.");
DEFINE_double(
    ba_balm_max_eigen_value, 0.05,
    "BALM maximum least significant eigen value, when looking for a plane in a "
    "voxel. Smaller values will result in flatter planes, but will need better "
    "initial poses.");

namespace ceres_error_terms {

Eigen::Matrix3d hat(const Eigen::Vector3d& v) {
  Eigen::Matrix3d Omega;
  Omega << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
  return Omega;
}

VoxHessFeature::VoxHessFeature(
    std::vector<size_t>&& idx, PointCluster&& fixed_point,
    std::vector<PointCluster>&& clusters_S, double c)
    : index(std::move(idx)),
      original_feature_cluster_G(fixed_point),
      observation_clusters_S(std::move(clusters_S)),
      coeff(c) {
  planes_S.reserve(observation_clusters_S.size());
}

void VoxHessFeature::generateOriginalPlanes() {
  for (size_t i = 0; i < observation_clusters_S.size(); ++i) {
    const PointCluster& cluster = observation_clusters_S[i];
    CHECK_GT(cluster.N, 0);

    Eigen::Vector3d v_bar = cluster.v / cluster.N;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cluster.cov());
    Eigen::Vector3d n = saes.eigenvectors().col(0);

    BALMPlane plane;
    plane.n = (n.dot(v_bar) <= 0) ? n : -n;
    plane.p = v_bar;

    bool is_good_plane =
        // Heuristic for plane quality
        (cluster.N >= 10 &&
         saes.eigenvalues()[0] / saes.eigenvalues()[1] <= 0.25);
    plane.sigmainv = is_good_plane ? 0.1 : 0.0;

    planes_S.push_back(plane);
  }
}

double VoxHessFeature::evaluateResidual(
    const std::vector<double*>& poses_M_I, PointCluster& full_feature_cluster_G,
    Eigen::Vector3d& lmbd, Eigen::Matrix3d& U,
    const aslam::Transformation& T_I_S,
    const aslam::Transformation& T_G_M) const {
  full_feature_cluster_G = original_feature_cluster_G;
  for (size_t i = 0; i < observation_clusters_S.size(); ++i) {
    const size_t pose_idx = index[i];
    CHECK_NOTNULL(poses_M_I[pose_idx]);

    const Eigen::Map<const Eigen::Matrix<double, 7, 1>> T_q_p(
        poses_M_I[pose_idx]);
    const Eigen::Quaterniond q_I_M(T_q_p.head<4>());
    const Eigen::Vector3d p_M_I = T_q_p.tail<3>();
    Eigen::Matrix3d R_I_M;
    common::toRotationMatrixJPL(q_I_M.coeffs(), &R_I_M);
    const aslam::Quaternion q_I_M_HML(R_I_M);
    const aslam::Transformation T_M_I(p_M_I, q_I_M_HML.inverse());

    const aslam::Transformation T_G_S = T_G_M * T_M_I * T_I_S;
    full_feature_cluster_G += observation_clusters_S[i].transform(T_G_S);
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(
      full_feature_cluster_G.cov());
  lmbd = saes.eigenvalues();
  U = saes.eigenvectors();

  return coeff * lmbd[0];
}

BALMPlane VoxHessFeature::evaluatePlane(
    const std::vector<double*>& poses_M_I, const aslam::Transformation& T_I_S,
    const aslam::Transformation& T_G_M) const {
  PointCluster full_feature_cluster_G = original_feature_cluster_G;
  for (size_t i = 0; i < observation_clusters_S.size(); ++i) {
    const size_t pose_idx = index[i];
    CHECK_NOTNULL(poses_M_I[pose_idx]);

    const Eigen::Map<const Eigen::Matrix<double, 7, 1>> T_q_p(
        poses_M_I[pose_idx]);
    const Eigen::Quaterniond q_I_M(T_q_p.head<4>());
    const Eigen::Vector3d p_M_I = T_q_p.tail<3>();
    Eigen::Matrix3d R_I_M;
    common::toRotationMatrixJPL(q_I_M.coeffs(), &R_I_M);
    const aslam::Quaternion q_I_M_HML(R_I_M);
    const aslam::Transformation T_M_I(p_M_I, q_I_M_HML.inverse());

    const aslam::Transformation T_M_S = T_M_I * T_I_S;
    full_feature_cluster_G += observation_clusters_S[i].transform(T_M_S);
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(
      full_feature_cluster_G.cov());
  BALMPlane plane;
  plane.n = saes.eigenvectors().col(0);
  plane.p = full_feature_cluster_G.v / full_feature_cluster_G.N;

  // Heuristic for plane quality
  bool is_good_plane =
      (full_feature_cluster_G.N >= 4 &&
       saes.eigenvalues()[0] / saes.eigenvalues()[1] <= 0.05);
  plane.sigmainv = is_good_plane ? 0.1 : 0.0;
  return plane;
}

VoxHess::VoxHess() = default;

VoxHess::VoxHess(vi_map::VIMap* map) {
  evaluateVoxHess(map);
}

void VoxHess::addFeature(
    std::vector<size_t>&& index, PointCluster&& fix_point,
    std::vector<PointCluster>&& sig_orig) {
  double total_points = 0.0;
  for (const auto& p : sig_orig) {
    total_points += p.N;
  }
  features_.emplace_back(
      std::move(index), std::move(fix_point), std::move(sig_orig),
      total_points);
}

void VoxHess::evaluateOnlyResidual(
    const aslam::TransformationVector& poses_M_I, double& total_residual) {
  total_residual = 0.0;
  constexpr int lambda_idx = 0;  // Smallest eigenvalue

  for (const auto& feature : features_) {
    PointCluster combined_sig = feature.original_feature_cluster_G;
    for (size_t i = 0; i < feature.observation_clusters_S.size(); ++i) {
      const size_t pose_idx = feature.index[i];
      combined_sig +=
          feature.observation_clusters_S[i].transform(poses_M_I[pose_idx]);
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(
        combined_sig.cov(), Eigen::EigenvaluesOnly);
    total_residual += feature.coeff * saes.eigenvalues()[lambda_idx];
  }
}

pose_graph::VertexIdList VoxHess::getVertexIds(
    const vi_map::MissionId& mission_id) const {
  auto it = vertex_ids_.find(mission_id);
  if (it != vertex_ids_.end()) {
    return it->second;
  }
  return {};
}

// Implementations for evaluateVoxHess and cutVoxel are included here.
// These are complex and dependent on the calling environment, but their
// internal logic is preserved from the original file.

void VoxHess::evaluateVoxHess(const vi_map::VIMap* map) {
  vi_map::MissionIdList mission_ids;
  map->getAllMissionIdsSortedByTimestamp(&mission_ids);

  aslam::TransformationVector poses_G_S;
  std::vector<resources::PointCloud> pointclouds;
  std::unordered_map<int64_t, pose_graph::VertexId> timestamp_to_vertex;

  pose_graph::VertexIdList vertices;
  map->getAllVertexIdsAlongGraphsSortedByTimestamp(&vertices);
  for (const auto& vertex_id : vertices) {
    const auto& vertex = map->getVertex(vertex_id);
    timestamp_to_vertex.emplace(vertex.getMinTimestampNanoseconds(), vertex_id);
  }

  int64_t time_last_kf = -1;
  vi_map::MissionId last_mission_id;
  last_mission_id.setInvalid();

  depth_integration::IntegrationFunctionPointCloudMaplabWithExtras
      integration_function =
          [this, &poses_G_S, &time_last_kf, &last_mission_id, &pointclouds,
           &timestamp_to_vertex](
              const aslam::Transformation& T_G_S, const int64_t timestamp_ns,
              const vi_map::MissionId& mission_id, const size_t,
              const resources::PointCloud& points_S) {
            poses_G_S.emplace_back(T_G_S);
            time_last_kf = timestamp_ns;
            last_mission_id = mission_id;
            pointclouds.emplace_back(points_S);
            this->vertex_ids_[mission_id].push_back(
                timestamp_to_vertex.at(timestamp_ns));
          };

  const int64_t time_threshold_ns = FLAGS_ba_balm_kf_time_threshold_s * 1e9;

  depth_integration::ResourceSelectionFunction selection_function =
      [&](const aslam::Transformation& T_G_S, const int64_t timestamp_ns,
          const vi_map::MissionId& mission_id, const size_t) {
        if (!last_mission_id.isValid() || mission_id != last_mission_id) {
          return true;
        }
        if (timestamp_ns - time_last_kf >= time_threshold_ns) {
          return true;
        }
        const aslam::Transformation T_Skf_S =
            poses_G_S.back().inverse() * T_G_S;
        if (T_Skf_S.getPosition().norm() >
                FLAGS_ba_balm_kf_distance_threshold_m ||
            aslam::AngleAxis(T_Skf_S.getRotation()).angle() >
                FLAGS_ba_balm_kf_rotation_threshold_deg * kDegToRad) {
          return true;
        }
        return false;
      };

  const backend::ResourceType input_resource_type =
      static_cast<backend::ResourceType>(
          FLAGS_ba_dense_depth_resource_input_type);

  depth_integration::integrateAllDepthResourcesOfType(
      mission_ids, input_resource_type,
      FLAGS_ba_dense_depth_map_reprojection_use_undistorted_camera, *map,
      integration_function, selection_function);

  SurfaceMap surface_map;
  for (size_t i = 0; i < poses_G_S.size(); ++i) {
    cutVoxel(surface_map, pointclouds[i], poses_G_S[i], i, poses_G_S.size());
  }

  // Find planes and prune the tree.
  for (auto it = surface_map.begin(); it != surface_map.end();) {
    if (it->second && it->second->findPlanes(this, nullptr)) {
      ++it;
    } else {
      it = surface_map.erase(it);
    }
  }
  VLOG(3) << "VoxHess Generated";
}

void VoxHess::cutVoxel(
    SurfaceMap& surface_map, const resources::PointCloud& points_S,
    const aslam::Transformation& T_G_S, size_t index, size_t /*num_scans*/) {
  resources::PointCloud points_G;
  points_G.appendTransformed(points_S, T_G_S);

  const Eigen::Map<const Eigen::Matrix3Xd> xyz_S(
      points_S.xyz.data(), 3, points_S.size());
  const Eigen::Map<const Eigen::Matrix3Xd> xyz_G(
      points_G.xyz.data(), 3, points_G.size());

  for (size_t i = 0; i < points_S.size(); ++i) {
    const Eigen::Vector3d pvec_tran = xyz_G.col(i);

    resources::VoxelPosition position(
        pvec_tran.x(), pvec_tran.y(), pvec_tran.z(), FLAGS_balm_voxel_size);

    std::unique_ptr<OctoTreeNode>& node_ptr = surface_map[position];
    if (!node_ptr) {
      const float half_length = FLAGS_balm_voxel_size / 2.0;
      Eigen::Vector3d center = {
          (position.x + 0.5f) * FLAGS_balm_voxel_size,
          (position.y + 0.5f) * FLAGS_balm_voxel_size,
          (position.z + 0.5f) * FLAGS_balm_voxel_size};
      // Assign the new node to the (previously null) pointer in the map.
      node_ptr = std::make_unique<OctoTreeNode>(center, half_length, 0);
    }

    // Now we can safely add the point to the node.
    node_ptr->addPoint(index, xyz_S.col(i), pvec_tran);
  }
}

OctoTreeNode::ScanContribution::ScanContribution(size_t idx)
    : scan_index(idx) {}

void OctoTreeNode::ScanContribution::addPoint(
    const Eigen::Vector3d& p_orig, const Eigen::Vector3d& p_tran) {
  points_S.push_back(p_orig);
  points_G.push_back(p_tran);
  cluster_S.push(p_orig);
  cluster_G.push(p_tran);
}

void OctoTreeNode::ScanContribution::clearPoints() {
  points_S.clear();
  points_S.shrink_to_fit();
  points_G.clear();
  points_G.shrink_to_fit();
  cluster_S = PointCluster();
  cluster_G = PointCluster();
}

OctoTreeNode::OctoTreeNode(
    const Eigen::Vector3d& center, float half_length, uint32_t layer)
    : layer_(layer), center_(center), half_length_(half_length) {}

void OctoTreeNode::addPoint(
    size_t scan_idx, const Eigen::Vector3d& point_S,
    const Eigen::Vector3d& point_G) {
  // Find the contribution from this scan, or create a new one.
  auto it = std::find_if(
      contributions_.begin(), contributions_.end(),
      [scan_idx](const ScanContribution& contrib) {
        return contrib.scan_index == scan_idx;
      });

  if (it == contributions_.end()) {
    contributions_.emplace_back(scan_idx);
    it = std::prev(contributions_.end());
  }
  it->addPoint(point_S, point_G);
}

bool OctoTreeNode::hasEnoughPoints() const {
  size_t total_points = 0;
  for (const auto& contrib : contributions_) {
    total_points += contrib.cluster_G.N;
  }
  return contributions_.size() >= 2 &&
         total_points > FLAGS_balm_min_plane_points;
}

bool OctoTreeNode::isPlanar() const {
  PointCluster total_stats;
  for (const auto& contrib : contributions_) {
    total_stats += contrib.cluster_G;
  }

  if (total_stats.N < 3) {
    return false;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(total_stats.cov());
  const Eigen::Vector3d eigenvalues = saes.eigenvalues();
  if (eigenvalues[1] < 1e-9) {  // Avoid division by zero
    return false;
  }

  return (eigenvalues[0] / eigenvalues[1]) < FLAGS_balm_max_eigen_value;
}

void OctoTreeNode::subdivide() {
  const float quarter_length = half_length_ / 2.0f;

  for (auto& contrib : contributions_) {
    for (size_t i = 0; i < contrib.points_G.size(); ++i) {
      const Eigen::Vector3d& point_G = contrib.points_G[i];
      const bool x = point_G.x() > center_.x();
      const bool y = point_G.y() > center_.y();
      const bool z = point_G.z() > center_.z();
      const int child_idx = (x << 2) | (y << 1) | z;

      if (children_[child_idx] == nullptr) {
        Eigen::Vector3d child_center = {
            center_.x() + (x ? 1 : -1) * quarter_length,
            center_.y() + (y ? 1 : -1) * quarter_length,
            center_.z() + (z ? 1 : -1) * quarter_length};
        children_[child_idx] = std::make_unique<OctoTreeNode>(
            child_center, quarter_length, layer_ + 1);
      }
      children_[child_idx]->addPoint(
          contrib.scan_index, contrib.points_S[i], point_G);
    }
    // Free memory in the parent node after distributing points to children
    contrib.clearPoints();
  }
}

bool OctoTreeNode::findPlanes(
    VoxHess* vox_hess, resources::PointCloud* debug_points_G) {
  if (!hasEnoughPoints()) {
    return false;
  }

  if (isPlanar()) {
    if (debug_points_G != nullptr) {
      const float intensity = 255.0 * rand() / (RAND_MAX + 1.0f);
      for (const auto& contrib : contributions_) {
        for (const auto& point : contrib.points_G) {
          debug_points_G->xyz.emplace_back(point.x());
          debug_points_G->xyz.emplace_back(point.y());
          debug_points_G->xyz.emplace_back(point.z());
          debug_points_G->scalars.emplace_back(intensity);
        }
      }
    }

    // Extract data to create a plane feature.
    std::vector<size_t> indices;
    std::vector<PointCluster> sig_origins;
    indices.reserve(contributions_.size());
    sig_origins.reserve(contributions_.size());

    for (auto&& contrib : contributions_) {
      indices.push_back(contrib.scan_index);
      sig_origins.push_back(std::move(contrib.cluster_S));
    }

    vox_hess->addFeature(
        std::move(indices), PointCluster(), std::move(sig_origins));
    return true;

  } else if (layer_ == FLAGS_balm_max_layers) {
    return false;
  }

  subdivide();

  bool keep_node = false;
  for (auto& child : children_) {
    if (child != nullptr) {
      if (child->findPlanes(vox_hess, debug_points_G)) {
        keep_node = true;
      } else {
        // Prune child if it and its descendants yield no planes
        child.reset();
      }
    }
  }

  return keep_node;
}

}  // namespace ceres_error_terms