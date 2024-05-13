#ifndef POSEGRAPH_LIDAR_POSE_GRAPH_H_
#define POSEGRAPH_LIDAR_POSE_GRAPH_H_

#include <memory>
#include <unordered_map>
#include <vector>

#include "posegraph/edge.h"
#include "posegraph/unique-id.h"
#include "posegraph/vertex.h"
#include "posegraph/pose-graph.h"

namespace pose_graph {

class LidarPoseGraph: public PoseGraph {
 public:
  MAPLAB_POINTER_TYPEDEFS(LidarPoseGraph);
};

}  // namespace pose_graph

#endif  // POSEGRAPH_LIDAR_POSE_GRAPH_H_
