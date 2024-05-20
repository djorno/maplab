#include "dense-reconstruction/balm/li-map.h"
#include "dense-reconstruction/voxblox-params.h"

#include <chrono>
#include <cstring>
#include <malloc.h>
#include <string>
#include <vector>

#include <aslam/common/timer.h>
#include <ceres-error-terms/balm-voxhess.h>
#include <dense-reconstruction/stereo-dense-reconstruction.h>
#include <depth-integration/depth-integration.h>
#include <gflags/gflags.h>
#include <map-manager/map-manager.h>
#include <maplab-common/conversions.h>
#include <maplab-common/file-system-tools.h>
#include <vi-map/unique-id.h>
#include <vi-map/vi-map.h>

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
    ba_balm_kf_distance_threshold_m, 0.5,
    "BALM distance threshold to add a new keyframe [m].");
DEFINE_double(
    ba_balm_kf_rotation_threshold_deg, 10.0,
    "BALM rotation threshold to add a new keyframe [deg].");
DEFINE_double(
    ba_balm_kf_time_threshold_s, 1.0,
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

PointCluster::PointCluster()
    : N(0), P(Eigen::Matrix3d::Zero()), v(Eigen::Vector3d::Zero()) {}

void PointCluster::push(const Eigen::Vector3d& vec) {
  N++;
  P += vec * vec.transpose();
  v += vec;
}

Eigen::Matrix3d PointCluster::cov() {
  Eigen::Vector3d center = v / N;
  return P / N - center * center.transpose();
}

PointCluster& PointCluster::operator+=(const PointCluster& sigv) {
  this->P += sigv.P;
  this->v += sigv.v;
  this->N += sigv.N;

  return *this;
}

PointCluster PointCluster::transform(const aslam::Transformation& T) const {
  const Eigen::Matrix3d R = T.getRotationMatrix();
  const Eigen::Vector3d p = T.getPosition();
  const Eigen::Matrix3d rp = R * v * p.transpose();

  PointCluster sigv;
  sigv.N = N;
  sigv.v = R * v + N * p;
  sigv.P = R * P * R.transpose() + rp + rp.transpose() + N * p * p.transpose();
  return sigv;
}

Eigen::Matrix3d hat(const Eigen::Vector3d& v) {
  Eigen::Matrix3d Omega;
  Omega << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
  return Omega;
}

void VoxHess::push_voxel(
    const std::vector<size_t>* index, const std::vector<PointCluster>* sig_orig,
    const PointCluster* fix) {
  double coe = 0;
  for (const PointCluster& p : *sig_orig) {
    coe += p.N;
  }

  indices.emplace_back(index);
  sig_vecs.emplace_back(fix);
  plvec_voxels.emplace_back(sig_orig);
  coeffs.emplace_back(coe);
}

void VoxHess::evaluate_only_residual(
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

OctoTreeNode::OctoTreeNode() {
  layer = 0;

  vec_orig.resize(1);
  vec_tran.resize(1);
  sig_orig.resize(1);
  sig_tran.resize(1);

  for (size_t i = 0; i < 8; ++i) {
    leaves[i] = nullptr;
  }
}

bool OctoTreeNode::judge_eigen() {
  PointCluster covMat;
  for (const PointCluster& p : sig_tran) {
    covMat += p;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat.cov());
  double decision = saes.eigenvalues()[0] / saes.eigenvalues()[1];

  return (decision < FLAGS_ba_balm_max_eigen_value);
}

void OctoTreeNode::cut_func() {
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

bool OctoTreeNode::recut(VoxHess* vox_opt, resources::PointCloud* points_G) {
  size_t point_size = 0;
  for (const PointCluster& p : sig_tran) {
    point_size += p.N;
  }

  const size_t num_pointclouds = index.size();
  if (num_pointclouds < 2 || point_size <= FLAGS_ba_balm_min_plane_points) {
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
  } else if (layer == FLAGS_ba_balm_max_layers) {
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

OctoTreeNode::~OctoTreeNode() {
  for (size_t i = 0; i < 8; ++i) {
    if (leaves[i] != nullptr) {
      delete leaves[i];
    }
  }
}

void VoxHess::cut_voxel(
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

    resources::VoxelPosition position(pvec_tran, FLAGS_ba_balm_voxel_size);
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

      node->voxel_center[0] = (0.5 + position.x) * FLAGS_ba_balm_voxel_size;
      node->voxel_center[1] = (0.5 + position.y) * FLAGS_ba_balm_voxel_size;
      node->voxel_center[2] = (0.5 + position.z) * FLAGS_ba_balm_voxel_size;
      node->quater_length = FLAGS_ba_balm_voxel_size / 4.0;
      node->layer = 0;

      surface_map[position] = node;
    }
  }
}

VoxHess::VoxHess(vi_map::VIMap* map) {
  vi_map::MissionIdList mission_ids;
  map->getAllMissionIdsSortedByTimestamp(&mission_ids);

  // Setting up BALM variables
  aslam::TransformationVector poses_G_S;
  aslam::TransformationVector poses_M_B;
  std::vector<resources::PointCloud> pointclouds;

  // Keyframe the point clouds, otherwise the memory blows up.
  // Base logic on motion and at fixed time intervals otherwise.
  int64_t time_last_kf;
  vi_map::MissionId last_mission_id;
  last_mission_id.setInvalid();
  std::vector<int64_t> keyframe_timestamps;
  std::vector<Eigen::Vector3d> keyframe_velocities;
  std::vector<Eigen::Vector3d> keyframe_gyro_bias;
  std::vector<Eigen::Vector3d> keyframe_accel_bias;

  // Accumulate point cloud into BALM format. Play with vi_map cache size
  // to facilitate multiple iterations over the same resources as we do
  // that for the undistortion.

  const size_t original_cache_size = map->getMaxCacheSize();
  map->setMaxCacheSize(1);
  depth_integration::IntegrationFunctionPointCloudMaplabWithExtrasAndImu
      integration_function =
          [&map, &poses_G_S, &time_last_kf, &last_mission_id, &pointclouds,
           &keyframe_timestamps, &keyframe_velocities, &keyframe_gyro_bias,
           &keyframe_accel_bias, &poses_M_B](
              const aslam::Transformation& T_G_S,
              const aslam::Transformation& T_M_B, const Eigen::Vector3d& v_M_B,
              const Eigen::Vector3d& gb_B, const Eigen::Vector3d& ab_B,
              const int64_t timestamp_ns, const vi_map::MissionId& mission_id,
              const size_t /*counter*/, const resources::PointCloud& points_S) {
            poses_G_S.emplace_back(T_G_S);
            time_last_kf = timestamp_ns;
            last_mission_id = mission_id;
            pointclouds.emplace_back(points_S);
            keyframe_timestamps.emplace_back(timestamp_ns);
            keyframe_velocities.emplace_back(v_M_B);
            keyframe_gyro_bias.emplace_back(gb_B);
            keyframe_accel_bias.emplace_back(ab_B);
            poses_M_B.emplace_back(T_M_B);

            // Increase cache size by one
            map->setMaxCacheSize(map->getMaxCacheSize() + 1u);
          };

  const int64_t time_threshold_ns =
      FLAGS_ba_balm_kf_time_threshold_s * kSecondsToNanoSeconds;
  const double distance_threshold = FLAGS_ba_balm_kf_distance_threshold_m;
  const double rotation_threshold =
      FLAGS_ba_balm_kf_rotation_threshold_deg * kDegToRad;

  depth_integration::ResourceSelectionFunction selection_function =
      [&poses_G_S, &time_last_kf, &last_mission_id, &time_threshold_ns,
       &distance_threshold, &rotation_threshold](
          const aslam::Transformation& T_G_S, const int64_t timestamp_ns,
          const vi_map::MissionId& mission_id, const size_t /*counter*/) {
        // We started another mission, so insert a keyframe and
        // re-initialize the other keyframe metrics we keep track of.
        if (!last_mission_id.isValid() || mission_id != last_mission_id) {
          return true;
        }

        // Keyframe if enough time has elapsed since the last keyframe.
        if (timestamp_ns - time_last_kf >= time_threshold_ns) {
          return true;
        }

        // Finally keyframe based on travelled distance or rotation.
        const aslam::Transformation& T_G_Skf = poses_G_S.back();
        const aslam::Transformation T_Skf_S = T_G_Skf.inverse() * T_G_S;
        const double distance_to_last_kf_m = T_Skf_S.getPosition().norm();
        const double rotation_to_last_kf_rad =
            aslam::AngleAxis(T_Skf_S.getRotation()).angle();
        if (distance_to_last_kf_m > distance_threshold ||
            rotation_to_last_kf_rad > rotation_threshold) {
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

  // get vi_map::VIMap
  vi_map::VIMap& vi_map = *map;
  LOG(INFO) << "Number of visual vertices: " << vi_map.numVertices();
  LOG(INFO) << "Number of lidar before: " << vi_map.numLidarVertices();
  li_map::addLidarToMap(mission_ids, vi_map, keyframe_timestamps);

  int win_size = poses_G_S.size();
  LOG(INFO) << "Selected a total of " << poses_G_S.size()
            << " LiDAR scans as keyframes.";
  LOG(INFO) << "Number of visual keyframes: " << vi_map.numVertices();
  LOG(INFO) << "Number of lidar after: " << vi_map.numLidarVertices();

  SurfaceMap surface_map;
  for (size_t i = 0; i < win_size; ++i) {
    cut_voxel(surface_map, pointclouds[i], poses_G_S[i], i, win_size);
  }

  resources::PointCloud planes_G;
  for (auto iter = surface_map.begin(); iter != surface_map.end();) {
    if (iter->second->recut(this, &planes_G)) {
      ++iter;
    } else {
      delete iter->second;
      iter = surface_map.erase(iter);
    }
  }
}

}  // namespace ceres_error_terms