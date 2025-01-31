#ifndef MAP_RESOURCES_RESOURCE_CACHE_INL_H_
#define MAP_RESOURCES_RESOURCE_CACHE_INL_H_

#include "map-resources/resource-common.h"

namespace backend {

template <typename DataType>
bool ResourceCache::getCacheResource(
    const ResourceId& id, const ResourceType& type, DataType* resource) const {
  CHECK_NOTNULL(resource);
  typename Cache<DataType>::ResourceDeque* cache = getCache<DataType>(type);

  bool found = false;
  if (cache != nullptr) {
    typename Cache<DataType>::ConstIterator it = std::find_if(
        cache->begin(), cache->end(),
        [id](const typename Cache<DataType>::Element& element) {
          return element.first == id;
        });
    if (it != cache->end()) {
      *resource = it->second;
      found = true;
    }
  }

  if (found) {
    ++(statistic_.hit[static_cast<size_t>(type)]);
  } else {
    ++(statistic_.miss[static_cast<size_t>(type)]);
  }
  return found;
}

template <typename DataType>
void ResourceCache::putCacheResource(
    const ResourceId& id, const ResourceType& type,
    const DataType& resource) const {
  typename Cache<DataType>::ResourceDeque* cache = getCache<DataType>(type);
  if (cache == nullptr) {
    cache = initCache<DataType>(type);
  }

  // Check if it is already in the cache.
  typename Cache<DataType>::ConstIterator it = std::find_if(
      cache->begin(), cache->end(),
      [id](const typename Cache<DataType>::Element& element) {
        return element.first == id;
      });
  CHECK(it == cache->end())
      << "Cannot put same resource in the cache twice! Id: " << id.hexString();

  cache->emplace_back(id, resource);
  while (cache->size() > max_cache_size_) {
    cache->pop_front();
  }

  updateCacheSizeStatistic<DataType>(type, *cache, &statistic_);
}

template <typename DataType>
bool ResourceCache::deleteCacheResource(
    const ResourceId& id, const ResourceType& type) const {
  typename Cache<DataType>::ResourceDeque* cache = getCache<DataType>(type);
  if (cache != nullptr) {
    typename Cache<DataType>::Iterator it = std::find_if(
        cache->begin(), cache->end(),
        [id](const typename Cache<DataType>::Element& element) {
          return element.first == id;
        });
    if (it != cache->end()) {
      cache->erase(it);

      updateCacheSizeStatistic<DataType>(type, *cache, &statistic_);
      return true;
    }
  }
  return false;
}

template <typename DataType>
typename ResourceCache::Cache<DataType>::ResourceDequePtr&
ResourceCache::getCachePtr(const ResourceType& /*type*/) const {
  LOG(FATAL) << "Implement ResourceCache::getCachePtr for your DataType!";
}

template <typename DataType>
typename ResourceCache::Cache<DataType>::ResourceDeque* ResourceCache::getCache(
    const ResourceType& type) const {
  return getCachePtr<DataType>(type).get();
}

template <typename DataType>
typename ResourceCache::Cache<DataType>::ResourceDeque*
ResourceCache::initCache(const ResourceType& type) const {
  typename ResourceCache::Cache<DataType>::ResourceDequePtr& cache_ptr =
      getCachePtr<DataType>(type);
  cache_ptr.reset(new typename ResourceCache::Cache<DataType>::ResourceDeque);
  return CHECK_NOTNULL(cache_ptr.get());
}

template <typename DataType>
void updateCacheSizeStatistic(
    const ResourceType& type,
    const typename ResourceCache::Cache<DataType>::ResourceDeque& cache,
    CacheStatistic* statistic) {
  CHECK_NOTNULL(statistic);
  const size_t type_idx = static_cast<size_t>(type);
  CHECK_LT(type_idx, statistic->cache_size.size());
  statistic->cache_size[type_idx] = cache.size();
}

}  // namespace backend

#endif  // MAP_RESOURCES_RESOURCE_CACHE_INL_H_
