#ifndef CERES_ERROR_TERMS_BALM_VOXHESS_H_
#define CERES_ERROR_TERMS_BALM_VOXHESS_H_

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
#include "posegraph/unique-id.h"

DECLARE_bool(dense_depth_map_reprojection_use_undistorted_camera);

DECLARE_string(dense_result_mesh_output_file);

DECLARE_string(dense_image_export_path);

DECLARE_int32(dense_depth_resource_output_type);

DECLARE_int32(dense_depth_resource_input_type);

DECLARE_double(balm_kf_distance_threshold_m);
DECLARE_double(balm_kf_rotation_threshold_deg);
DECLARE_double(balm_kf_time_threshold_s);

DECLARE_double(balm_voxel_size);
DECLARE_uint32(balm_max_layers);
DECLARE_uint32(balm_min_plane_points);
DECLARE_double(balm_max_eigen_value);

DECLARE_double(balm_vis_voxel_size);

namespace ceres_error_terms {
#define PLM(a)                     \
  std::vector<                     \
      Eigen::Matrix<double, a, a>, \
      Eigen::aligned_allocator<Eigen::Matrix<double, a, a>>>
#define PLV(a)                     \
  std::vector<                     \
      Eigen::Matrix<double, a, 1>, \
      Eigen::aligned_allocator<Eigen::Matrix<double, a, 1>>>
class OctoTreeNode;
typedef std::unordered_map<
    resources::VoxelPosition, std::unique_ptr<OctoTreeNode>>
    SurfaceMap;

struct BALMPlane {
  Eigen::Vector3d n;  // normal
  Eigen::Vector3d p;  // point on plane
  double sigmainv;    // 1 / sigma
};

struct PointCluster {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Matrix3d P = Eigen::Matrix3d::Zero();
  Eigen::Vector3d v = Eigen::Vector3d::Zero();
  size_t N = 0;

  PointCluster() = default;

  void push(const Eigen::Vector3d& vec) {
    N++;
    P += vec * vec.transpose();
    v += vec;
  }

  Eigen::Matrix3d cov() const {
    if (N == 0)
      return Eigen::Matrix3d::Zero();
    Eigen::Vector3d center = v / N;
    return P / N - center * center.transpose();
  }

  PointCluster& operator+=(const PointCluster& other) {
    this->P += other.P;
    this->v += other.v;
    this->N += other.N;
    return *this;
  }

  PointCluster transform(const aslam::Transformation& T) const {
    const Eigen::Matrix3d R = T.getRotationMatrix();
    const Eigen::Vector3d p = T.getPosition();
    const Eigen::Matrix3d rp = R * v * p.transpose();

    PointCluster transformed_sig;
    transformed_sig.N = N;
    transformed_sig.v = R * v + N * p;
    transformed_sig.P =
        R * P * R.transpose() + rp + rp.transpose() + N * p * p.transpose();
    return transformed_sig;
  }
};

Eigen::Matrix3d hat(const Eigen::Vector3d& v);

// Container for a single collection of BALM planes expressing a feature
struct VoxHessFeature {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // a vector of indices, each entry idx corresponds to the contribution of the
  // pose i to the PC
  std::vector<size_t> index;
  const PointCluster original_feature_cluster_G;
  std::vector<PointCluster> observation_clusters_S;
  double coeff;
  std::vector<BALMPlane> planes_S;

  VoxHessFeature(
      std::vector<size_t>&& idx, PointCluster&& fixed_point,
      std::vector<PointCluster>&& clusters_S, double c);

  void generateOriginalPlanes();

  double evaluateResidual(
      const std::vector<double*>& poses_M_I,
      PointCluster& full_feature_cluster_G, Eigen::Vector3d& lmbd,
      Eigen::Matrix3d& U, const aslam::Transformation& T_I_S,
      const aslam::Transformation& T_G_M) const;

  BALMPlane evaluatePlane(
      const std::vector<double*>& poses_M_I, const aslam::Transformation& T_I_S,
      const aslam::Transformation& T_G_M) const;
};

class VoxHess {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VoxHess();
  explicit VoxHess(vi_map::VIMap* map);

  void addFeature(
      std::vector<size_t>&& index, PointCluster&& fix_point,
      std::vector<PointCluster>&& sig_orig);

  void evaluateOnlyResidual(
      const aslam::TransformationVector& poses_M_I, double& total_residual);

  void evaluateVoxHess(const vi_map::VIMap* map);

  void cutVoxel(
      SurfaceMap& surface_map, const resources::PointCloud& points_S,
      const aslam::Transformation& T_G_S, size_t index, size_t num_scans);

  pose_graph::VertexIdList getVertexIds(
      const vi_map::MissionId& mission_id) const;

  int getNumFeatures() const {
    return static_cast<int>(features_.size());
  }

  VoxHessFeature& getFeatureMutable(size_t idx) {
    CHECK_LT(idx, features_.size());
    return features_[idx];
  }

  const VoxHessFeature& getFeature(size_t idx) const {
    CHECK_LT(idx, features_.size());
    return features_[idx];
  }

  std::vector<VoxHessFeature>& getFeatureVecMutable() {
    return features_;
  }

  const std::vector<VoxHessFeature>& getFeatureVec() const {
    return features_;
  }

  auto begin() {
    return features_.begin();
  }

  auto end() {
    return features_.end();
  }

 private:
  std::vector<VoxHessFeature> features_;
  vi_map::MissionVertexIdList vertex_ids_;
};

class OctoTreeNode {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct ScanContribution {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // index of the scan that observes this feature
    size_t scan_index;
    // PC in sensor frame
    PointCluster cluster_S;
    // PC in global (common) frame
    PointCluster cluster_G;
    // Full point cloud positions in sensor frame
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        points_S;
    // Full point cloud positions in global frame
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        points_G;

    explicit ScanContribution(size_t idx);
    void addPoint(const Eigen::Vector3d& p_orig, const Eigen::Vector3d& p_tran);
    void clearPoints();
  };

  OctoTreeNode(
      const Eigen::Vector3d& center, float half_length, uint32_t layer);

  void addPoint(
      size_t scan_idx, const Eigen::Vector3d& p_orig,
      const Eigen::Vector3d& p_tran);

  // The main recursive function to find and extract planes.
  // Returns true if this node or any of its children should be kept.
  bool findPlanes(VoxHess* vox_hess, resources::PointCloud* debug_points_G);

 private:
  uint32_t layer_;
  Eigen::Vector3d center_;
  float half_length_;

  std::vector<ScanContribution> contributions_;
  std::array<std::unique_ptr<OctoTreeNode>, 8> children_;

  bool hasEnoughPoints() const;
  bool isPlanar() const;
  void subdivide();
};

}  // namespace ceres_error_terms
#endif  // CERES_ERROR_TERMS_BALM_VOXHESS_H_