#ifndef MAP_RESOURCES_RESOURCE_LOADER_INL_H_
#define MAP_RESOURCES_RESOURCE_LOADER_INL_H_

#include <string>

#include <glog/logging.h>
#include <maplab-common/file-system-tools.h>

#include "map-resources/resource-common.h"

namespace backend {

template <typename DataType>
void ResourceLoader::addResource(
    const ResourceId& id, const ResourceType& type, const std::string& folder,
    const DataType& resource) {
  CHECK(!folder.empty());

  // Check if the newest resource should be added to the map.
  if (always_cache_newest_resource_) {
    putCacheResource<DataType>(id, type, resource);
  }

  std::string file_path;
  getResourceFilePath(id, type, folder, &file_path);
  saveResourceToFile(file_path, type, resource);
}

template <typename DataType>
void ResourceLoader::getResource(
    const ResourceId& id, const ResourceType& type, const std::string& folder,
    DataType* resource) const {
  CHECK(!folder.empty());
  CHECK_NOTNULL(resource);
  if (getCacheResource<DataType>(id, type, resource)) {
    return;
  } else {
    std::string file_path;
    getResourceFilePath(id, type, folder, &file_path);
    CHECK(loadResourceFromFile(file_path, type, resource))
        << "Failed to load " << ResourceTypeNames[static_cast<size_t>(type)]
        << " resource with id " << id.hexString()
        << " from file: " << file_path;
    putCacheResource<DataType>(id, type, *resource);
  }
}

template <typename DataType>
bool ResourceLoader::checkResourceFile(
    const ResourceId& id, const ResourceType& type,
    const std::string& folder) const {
  CHECK(!folder.empty());
  DataType resource;
  std::string file_path;
  getResourceFilePath(id, type, folder, &file_path);
  return loadResourceFromFile(file_path, type, &resource);
}

template <typename DataType>
void ResourceLoader::replaceResource(
    const ResourceId& id, const ResourceType& type, const std::string& folder,
    const DataType& resource) {
  CHECK(!folder.empty());
  deleteCacheResource<DataType>(id, type);
  deleteResourceFile(id, type, folder);
  addResource<DataType>(id, type, folder, resource);
}

template <typename DataType>
void ResourceLoader::saveResourceToFile(
    const std::string& file, const ResourceType& type,
    const DataType& /*resource*/) const {
  LOG(FATAL) << "ResourceLoader::saveResourceToFile() is not implemented for "
             << "this DataType! Cannot write resource of type "
             << ResourceTypeNames[static_cast<size_t>(type)]
             << " to file: " << file;
}

template <typename DataType>
bool ResourceLoader::loadResourceFromFile(
    const std::string& file, const ResourceType& type,
    DataType* /*resource*/) const {
  LOG(FATAL) << "ResourceLoader::loadResourceFromFile() is not implemented for "
             << "this DataType! Cannot load resource of type "
             << ResourceTypeNames[static_cast<size_t>(type)]
             << " from file: " << file;
  return false;
}

template <typename DataType>
void ResourceLoader::deleteResource(
    const ResourceId& id, const ResourceType& type, const std::string& folder) {
  CHECK(!folder.empty());
  // This is more expensive than the templated deleteResource function, because
  // it needs to check all caches, but at least we don't need to template this
  // function.
  deleteCacheResource<DataType>(id, type);
  deleteResourceFile(id, type, folder);
}

}  // namespace backend

#endif  // MAP_RESOURCES_RESOURCE_LOADER_INL_H_
