#ifndef BALM_LI_MAP_H_
#define BALM_LI_MAP_H_

#include <Eigen/Core>
#include <cstdint>
#include <vector>
#include <vi-map/vi-map.h>

namespace li_map {

// Declaration of the function to be implemented in the .cc file
void addLidarToMap(
    const vi_map::MissionIdList& mission_ids, vi_map::VIMap& vi_map,
    const std::vector<int64_t>& lidar_keyframe_timestamps);

}  // namespace li_map

#endif  // BALM_LI_MAP_H_