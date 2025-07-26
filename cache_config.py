# cache_config.py - Advanced Caching Configuration for F1 Dashboard

import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import streamlit as st

class CacheConfig:
    """Configuration settings for the F1 dashboard caching system."""
    
    # Cache TTL (Time To Live) in seconds
    DATA_CACHE_TTL = 7200  # 2 hours for F1 data
    FEATURE_CACHE_TTL = 3600  # 1 hour for feature engineering
    MODEL_CACHE_TTL = 1800  # 30 minutes for trained models
    CIRCUIT_CACHE_TTL = 3600  # 1 hour for circuit clustering
    PREDICTION_CACHE_TTL = 900  # 15 minutes for predictions
    
    # Memory management
    MAX_CACHE_SIZE_MB = 500  # Maximum total cache size in MB
    MAX_CACHED_MODELS = 10  # Maximum number of cached models
    MAX_CACHED_FEATURES = 20  # Maximum number of cached feature sets
    
    # Cache invalidation settings
    AUTO_CLEANUP_INTERVAL = 1800  # 30 minutes
    MAX_CACHE_AGE_HOURS = 24  # Maximum age before forced cleanup
    
    # Performance settings
    ENABLE_COMPRESSION = True  # Enable cache compression for large objects
    ENABLE_PERSISTENT_CACHE = True  # Save cache to disk between sessions
    CACHE_HIT_TRACKING = True  # Track cache hit/miss statistics
    
    # Session state keys
    SESSION_KEYS = {
        'data_cache': 'cached_f1_data',
        'models_cache': 'models_cache',
        'features_cache': 'features_cache',
        'cache_keys': 'cache_keys',
        'cache_stats': 'cache_statistics',
        'last_cleanup': 'last_cache_cleanup'
    }

class CacheManager:
    """Advanced cache manager for the F1 prediction dashboard."""
    
    def __init__(self):
        self.config = CacheConfig()
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state for caching."""
        for key, default_value in {
            self.config.SESSION_KEYS['cache_keys']: {},
            self.config.SESSION_KEYS['models_cache']: {},
            self.config.SESSION_KEYS['features_cache']: {},
            self.config.SESSION_KEYS['cache_stats']: {
                'hits': 0, 'misses': 0, 'total_requests': 0,
                'data_loads': 0, 'model_trains': 0, 'feature_engineers': 0
            },
            self.config.SESSION_KEYS['last_cleanup']: None,
            'data_loaded': False,
            'last_data_load': None
        }.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def should_refresh_cache(self, cache_key: str, max_age_seconds: Optional[int] = None) -> bool:
        """Check if cache should be refreshed based on age."""
        if max_age_seconds is None:
            max_age_seconds = self.config.DATA_CACHE_TTL
        
        cache_keys = st.session_state[self.config.SESSION_KEYS['cache_keys']]
        
        if cache_key not in cache_keys:
            return True
        
        cache_time = cache_keys[cache_key]
        return (time.time() - cache_time) > max_age_seconds
    
    def update_cache_timestamp(self, cache_key: str):
        """Update cache timestamp for a given key."""
        st.session_state[self.config.SESSION_KEYS['cache_keys']][cache_key] = time.time()
    
    def get_cache_age(self, cache_key: str) -> Optional[timedelta]:
        """Get the age of a cached item."""
        cache_keys = st.session_state[self.config.SESSION_KEYS['cache_keys']]
        
        if cache_key not in cache_keys:
            return None
        
        cache_time = cache_keys[cache_key]
        return timedelta(seconds=time.time() - cache_time)
    
    def record_cache_hit(self, cache_type: str = 'general'):
        """Record a cache hit for statistics."""
        if self.config.CACHE_HIT_TRACKING:
            stats = st.session_state[self.config.SESSION_KEYS['cache_stats']]
            stats['hits'] += 1
            stats['total_requests'] += 1
    
    def record_cache_miss(self, cache_type: str = 'general'):
        """Record a cache miss for statistics."""
        if self.config.CACHE_HIT_TRACKING:
            stats = st.session_state[self.config.SESSION_KEYS['cache_stats']]
            stats['misses'] += 1
            stats['total_requests'] += 1
            
            # Track specific miss types
            if cache_type == 'data':
                stats['data_loads'] += 1
            elif cache_type == 'model':
                stats['model_trains'] += 1
            elif cache_type == 'feature':
                stats['feature_engineers'] += 1
    
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        stats = st.session_state[self.config.SESSION_KEYS['cache_stats']]
        total = stats['total_requests']
        
        if total == 0:
            return 0.0
        
        return (stats['hits'] / total) * 100
    
    def get_total_cache_size(self) -> float:
        """Calculate total cache size in MB."""
        import sys
        
        total_size = 0
        
        # Data cache size
        if st.session_state.get('data_loaded', False):
            data = st.session_state.get(self.config.SESSION_KEYS['data_cache'], {})
            total_size += sys.getsizeof(data)
        
        # Models cache size
        models_cache = st.session_state[self.config.SESSION_KEYS['models_cache']]
        for model_data, _ in models_cache.values():
            total_size += sys.getsizeof(model_data)
        
        # Features cache size
        features_cache = st.session_state[self.config.SESSION_KEYS['features_cache']]
        for features_data, _ in features_cache.values():
            total_size += sys.getsizeof(features_data)
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def cleanup_old_cache(self, force: bool = False):
        """Clean up old cache entries based on age and size."""
        current_time = time.time()
        
        # Check if cleanup is needed
        last_cleanup = st.session_state[self.config.SESSION_KEYS['last_cleanup']]
        if not force and last_cleanup:
            if (current_time - last_cleanup) < self.config.AUTO_CLEANUP_INTERVAL:
                return
        
        # Clean up old cache keys
        cache_keys = st.session_state[self.config.SESSION_KEYS['cache_keys']]
        max_age = self.config.MAX_CACHE_AGE_HOURS * 3600
        
        old_keys = [
            key for key, timestamp in cache_keys.items()
            if (current_time - timestamp) > max_age
        ]
        
        for key in old_keys:
            del cache_keys[key]
        
        # Clean up models cache if too many cached
        models_cache = st.session_state[self.config.SESSION_KEYS['models_cache']]
        if len(models_cache) > self.config.MAX_CACHED_MODELS:
            # Remove oldest models
            sorted_models = sorted(models_cache.items(), key=lambda x: x[1][1])  # Sort by timestamp
            excess_count = len(models_cache) - self.config.MAX_CACHED_MODELS
            
            for i in range(excess_count):
                del models_cache[sorted_models[i][0]]
        
        # Clean up features cache if too many cached
        features_cache = st.session_state[self.config.SESSION_KEYS['features_cache']]
        if len(features_cache) > self.config.MAX_CACHED_FEATURES:
            # Remove oldest features
            sorted_features = sorted(features_cache.items(), key=lambda x: x[1][1])  # Sort by timestamp
            excess_count = len(features_cache) - self.config.MAX_CACHED_FEATURES
            
            for i in range(excess_count):
                del features_cache[sorted_features[i][0]]
        
        # Check total cache size and clean if necessary
        total_size = self.get_total_cache_size()
        if total_size > self.config.MAX_CACHE_SIZE_MB:
            # Clear features cache first (easiest to regenerate)
            st.session_state[self.config.SESSION_KEYS['features_cache']] = {}
            
            # If still too large, clear some models
            if self.get_total_cache_size() > self.config.MAX_CACHE_SIZE_MB:
                models_cache = st.session_state[self.config.SESSION_KEYS['models_cache']]
                # Keep only the 3 most recent models
                sorted_models = sorted(models_cache.items(), key=lambda x: x[1][1], reverse=True)
                st.session_state[self.config.SESSION_KEYS['models_cache']] = dict(sorted_models[:3])
        
        # Update last cleanup time
        st.session_state[self.config.SESSION_KEYS['last_cleanup']] = current_time
    
    def clear_all_cache(self):
        """Clear all cached data."""
        for key in [
            self.config.SESSION_KEYS['models_cache'],
            self.config.SESSION_KEYS['features_cache'],
            self.config.SESSION_KEYS['cache_keys']
        ]:
            st.session_state[key] = {}
        
        st.session_state['data_loaded'] = False
        st.session_state['last_data_load'] = None
        
        # Reset statistics
        st.session_state[self.config.SESSION_KEYS['cache_stats']] = {
            'hits': 0, 'misses': 0, 'total_requests': 0,
            'data_loads': 0, 'model_trains': 0, 'feature_engineers': 0
        }
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = st.session_state[self.config.SESSION_KEYS['cache_stats']].copy()
        
        stats.update({
            'total_cache_size_mb': self.get_total_cache_size(),
            'hit_rate_percentage': self.get_cache_hit_rate(),
            'cached_models_count': len(st.session_state[self.config.SESSION_KEYS['models_cache']]),
            'cached_features_count': len(st.session_state[self.config.SESSION_KEYS['features_cache']]),
            'data_cache_loaded': st.session_state.get('data_loaded', False),
            'last_data_load': st.session_state.get('last_data_load'),
            'last_cleanup': st.session_state[self.config.SESSION_KEYS['last_cleanup']]
        })
        
        return stats
    
    def optimize_cache(self) -> Dict[str, str]:
        """Optimize cache performance and return optimization results."""
        results = {}
        
        # Check and clean up old cache
        old_size = self.get_total_cache_size()
        self.cleanup_old_cache(force=True)
        new_size = self.get_total_cache_size()
        
        size_reduction = old_size - new_size
        if size_reduction > 0:
            results['size_optimization'] = f"Reduced cache size by {size_reduction:.1f} MB"
        else:
            results['size_optimization'] = "Cache size already optimal"
        
        # Check cache hit rate
        hit_rate = self.get_cache_hit_rate()
        if hit_rate < 50:
            results['hit_rate_warning'] = f"Low hit rate ({hit_rate:.1f}%) - consider adjusting TTL settings"
        else:
            results['hit_rate_status'] = f"Good hit rate ({hit_rate:.1f}%)"
        
        # Check memory usage
        total_size = self.get_total_cache_size()
        if total_size > self.config.MAX_CACHE_SIZE_MB * 0.8:
            results['memory_warning'] = f"High memory usage ({total_size:.1f} MB)"
        else:
            results['memory_status'] = f"Memory usage optimal ({total_size:.1f} MB)"
        
        return results

# Global cache manager instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

# Convenience functions for use in the dashboard
def should_refresh_cache(cache_key: str, max_age_seconds: Optional[int] = None) -> bool:
    """Check if cache should be refreshed."""
    return get_cache_manager().should_refresh_cache(cache_key, max_age_seconds)

def update_cache_timestamp(cache_key: str):
    """Update cache timestamp."""
    get_cache_manager().update_cache_timestamp(cache_key)

def record_cache_hit(cache_type: str = 'general'):
    """Record cache hit."""
    get_cache_manager().record_cache_hit(cache_type)

def record_cache_miss(cache_type: str = 'general'):
    """Record cache miss."""
    get_cache_manager().record_cache_miss(cache_type)

def cleanup_cache(force: bool = False):
    """Clean up old cache entries."""
    get_cache_manager().cleanup_old_cache(force)

def get_cache_statistics() -> Dict[str, Any]:
    """Get cache statistics."""
    return get_cache_manager().get_cache_statistics()

def optimize_cache() -> Dict[str, str]:
    """Optimize cache performance."""
    return get_cache_manager().optimize_cache() 