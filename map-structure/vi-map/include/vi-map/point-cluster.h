#ifndef VI_MAP_POINT_CLUSTER_H_
#define VI_MAP_POINT_CLUSTER_H_

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
#include "vi-map/vertex.h"

namespace vi_map {
class PointCluster {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Matrix3d P;
  Eigen::Vector3d v;
  size_t N;

  PointCluster() {
    P.setZero();
    v.setZero();
    N = 0;
  }

  void push(const Eigen::Vector3d& vec) {
    N++;
    P += vec * vec.transpose();
    v += vec;
  }

  Eigen::Matrix3d cov() {
    Eigen::Vector3d center = v / N;
    return P / N - center * center.transpose();
  }

  PointCluster& operator+=(const PointCluster& sigv) {
    this->P += sigv.P;
    this->v += sigv.v;
    this->N += sigv.N;

    return *this;
  }

  PointCluster transform(const aslam::Transformation& T) const {
    const Eigen::Matrix3d R = T.getRotationMatrix();
    const Eigen::Vector3d p = T.getPosition();
    const Eigen::Matrix3d rp = R * v * p.transpose();

    PointCluster sigv;
    sigv.N = N;
    sigv.v = R * v + N * p;
    sigv.P =
        R * P * R.transpose() + rp + rp.transpose() + N * p * p.transpose();
    return sigv;
  }
};
}  // namespace vi_map

#endif  // VI_MAP_POINT_CLUSTER_H_