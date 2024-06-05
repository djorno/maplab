#ifndef CERES_ERROR_TERMS_BALM_VOXHESS_H_
#define CERES_ERROR_TERMS_BALM_VOXHESS_H_

#include "dense-reconstruction/balm/li-map.h"
#include "dense-reconstruction/dense-reconstruction-plugin.h"
#include "dense-reconstruction/voxblox-params.h"

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
typedef std::unordered_map<resources::VoxelPosition, OctoTreeNode*> SurfaceMap;

struct BALMPlane {
  Eigen::Vector3d n;  // normal
  Eigen::Vector3d p;  // point on plane
};

class PointCluster {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Matrix3d P;
  Eigen::Vector3d v;
  size_t N;

  PointCluster();

  void push(const Eigen::Vector3d& vec);

  Eigen::Matrix3d cov();

  PointCluster& operator+=(const PointCluster& sigv);

  PointCluster transform(const aslam::Transformation& T) const;
};

Eigen::Matrix3d hat(const Eigen::Vector3d& v);

class VoxHess {
 public:
  std::vector<const std::vector<size_t>*> indices;
  std::vector<const PointCluster*> sig_vecs;
  std::vector<const std::vector<PointCluster>*> plvec_voxels;
  std::vector<double> coeffs;

  VoxHess(vi_map::VIMap* map);

  VoxHess();

  void evaluate_voxhess(vi_map::VIMap* map);

  void push_voxel(
      const std::vector<size_t>* index,
      const std::vector<PointCluster>* sig_orig, const PointCluster* fix);

  void evaluate_only_residual(
      const aslam::TransformationVector& xs, double& residual);
  void cut_voxel(
      SurfaceMap& surface_map, const resources::PointCloud& points_S,
      const aslam::Transformation& T_G_S, size_t index, size_t num_scans);

 private:
  bool update_map_ = true;
};

class VoxHessAtom {
 public:
  const std::vector<size_t>& index;
  const PointCluster& sig;
  const std::vector<PointCluster>& sig_origin;
  double coeff;

  VoxHessAtom(const VoxHess& voxhess, size_t& feature_index);

  double evaluate_residual(
      const std::vector<double*>& xs, PointCluster& sig_mutable,
      Eigen::Vector3d& lmbd, Eigen::Matrix3d& U,
      const aslam::Transformation& T_I_S, const aslam::Transformation& T_G_M);

  void evaluate_plane(
      const std::vector<double*>& xs, BALMPlane& plane,
      const aslam::Transformation& T_I_S, const aslam::Transformation& T_G_M);

  void evaluate_plane_per_pose(
      const double* x_i_ptr, BALMPlane& plane, const size_t sig_i,
      const aslam::Transformation& T_I_S, const aslam::Transformation& T_G_M);
};

class OctoTreeNode {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  size_t layer;
  std::vector<size_t> index;
  std::vector<PLV(3)> vec_orig, vec_tran;
  std::vector<PointCluster> sig_orig, sig_tran;
  PointCluster fix_point;

  OctoTreeNode* leaves[8];
  float voxel_center[3];
  float quater_length;

  OctoTreeNode();

  bool judge_eigen();

  void cut_func();

  bool recut(VoxHess* vox_opt, resources::PointCloud* points_G);

  ~OctoTreeNode();
};

}  // namespace ceres_error_terms
#endif  // CERES_ERROR_TERMS_BALM_VOXHESS_H_