"""
Redis cache management for StellarVault AI Engine
"""
import json
import pickle
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import redis.asyncio as redis
from loguru import logger

from .config import settings


class RedisCache:
    """
    Redis cache manager for high-performance data caching
    """
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.is_connected = False
        
    async def initialize(self):
        """
        Initialize Redis connection
        """
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            await self.redis_client.ping()
            self.is_connected = True
            logger.info("✅ Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            self.is_connected = False
            raise
    
    async def close(self):
        """
        Close Redis connection
        """
        if self.redis_client:
            await self.redis_client.close()
            self.is_connected = False
            logger.info("Redis connection closed")
    
    async def ping(self) -> bool:
        """
        Test Redis connection
        """
        try:
            if self.redis_client:
                await self.redis_client.ping()
                return True
            return False
        except Exception:
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        """
        if not self.is_connected:
            return None
            
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache
        """
        if not self.is_connected:
            return False
            
        try:
            ttl = ttl or settings.REDIS_TTL
            serialized_value = json.dumps(value, default=str)
            await self.redis_client.setex(key, ttl, serialized_value)
            return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache
        """
        if not self.is_connected:
            return False
            
        try:
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values from cache
        """
        if not self.is_connected or not keys:
            return {}
            
        try:
            values = await self.redis_client.mget(keys)
            result = {}
            for key, value in zip(keys, values):
                if value:
                    result[key] = json.loads(value)
            return result
        except Exception as e:
            logger.error(f"Cache get_many error: {e}")
            return {}
    
    async def set_many(self, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Set multiple values in cache
        """
        if not self.is_connected or not data:
            return False
            
        try:
            ttl = ttl or settings.REDIS_TTL
            pipe = self.redis_client.pipeline()
            
            for key, value in data.items():
                serialized_value = json.dumps(value, default=str)
                pipe.setex(key, ttl, serialized_value)
            
            await pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Cache set_many error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache
        """
        if not self.is_connected:
            return False
            
        try:
            result = await self.redis_client.exists(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment a counter in cache
        """
        if not self.is_connected:
            return None
            
        try:
            result = await self.redis_client.incrby(key, amount)
            return result
        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {e}")
            return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        """
        if not self.is_connected:
            return {"connected": False}
            
        try:
            info = await self.redis_client.info()
            return {
                "connected": True,
                "used_memory": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0)
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"connected": False, "error": str(e)}


# Global cache instance
redis_client = RedisCache()


# Cache decorators and utilities
def cache_key(prefix: str, *args) -> str:
    """
    Generate cache key from prefix and arguments
    """
    key_parts = [str(arg) for arg in args if arg is not None]
    return f"{prefix}:{':'.join(key_parts)}"


async def cached_market_data(symbol: str, source: str, data_type: str) -> Optional[Dict[str, Any]]:
    """
    Get cached market data
    """
    key = cache_key("market_data", symbol, source, data_type)
    return await redis_client.get(key)


async def cache_market_data(symbol: str, source: str, data_type: str, data: Dict[str, Any], ttl: int = 300) -> bool:
    """
    Cache market data (5 minutes default TTL)
    """
    key = cache_key("market_data", symbol, source, data_type)
    return await redis_client.set(key, data, ttl)


async def cached_valuation(asset_id: str, model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get cached valuation
    """
    key = cache_key("valuation", asset_id, model_name)
    return await redis_client.get(key)


async def cache_valuation(asset_id: str, model_name: str, valuation: Dict[str, Any], ttl: int = 3600) -> bool:
    """
    Cache valuation (1 hour default TTL)
    """
    key = cache_key("valuation", asset_id, model_name)
    return await redis_client.set(key, valuation, ttl)


async def cached_risk_assessment(asset_id: str) -> Optional[Dict[str, Any]]:
    """
    Get cached risk assessment
    """
    key = cache_key("risk_assessment", asset_id)
    return await redis_client.get(key)


async def cache_risk_assessment(asset_id: str, assessment: Dict[str, Any], ttl: int = 1800) -> bool:
    """
    Cache risk assessment (30 minutes default TTL)
    """
    key = cache_key("risk_assessment", asset_id)
    return await redis_client.set(key, assessment, ttl)


async def cached_portfolio_analysis(user_id: str, portfolio_id: str) -> Optional[Dict[str, Any]]:
    """
    Get cached portfolio analysis
    """
    key = cache_key("portfolio_analysis", user_id, portfolio_id)
    return await redis_client.get(key)


async def cache_portfolio_analysis(user_id: str, portfolio_id: str, analysis: Dict[str, Any], ttl: int = 600) -> bool:
    """
    Cache portfolio analysis (10 minutes default TTL)
    """
    key = cache_key("portfolio_analysis", user_id, portfolio_id)
    return await redis_client.set(key, analysis, ttl)


async def invalidate_cache_pattern(pattern: str) -> int:
    """
    Invalidate cache keys matching a pattern
    """
    if not redis_client.is_connected:
        return 0
        
    try:
        keys = await redis_client.redis_client.keys(pattern)
        if keys:
            await redis_client.redis_client.delete(*keys)
            return len(keys)
        return 0
    except Exception as e:
        logger.error(f"Cache invalidation error for pattern {pattern}: {e}")
        return 0


async def get_cache_usage() -> Dict[str, int]:
    """
    Get cache usage statistics by prefix
    """
    if not redis_client.is_connected:
        return {}
        
    try:
        all_keys = await redis_client.redis_client.keys("*")
        usage = {}
        
        for key in all_keys:
            prefix = key.split(":")[0]
            usage[prefix] = usage.get(prefix, 0) + 1
            
        return usage
    except Exception as e:
        logger.error(f"Cache usage error: {e}")
        return {}


# Cache warming utilities
async def warm_cache_for_asset(asset_id: str):
    """
    Warm cache for a specific asset
    """
    logger.info(f"Warming cache for asset {asset_id}")
    # This would typically trigger valuation and risk assessment
    # caching for commonly requested assets
    pass


async def warm_cache_for_user(user_id: str):
    """
    Warm cache for a specific user
    """
    logger.info(f"Warming cache for user {user_id}")
    # This would typically cache user's portfolio data
    # and frequently accessed market data
    pass 


class CacheManager:
    """
    Cache management class
    """
    
    def __init__(self):
        self.redis_client = None
    
    async def initialize(self):
        """
        Initialize cache connections
        """
        try:
            import aioredis
            self.redis_client = aioredis.from_url(settings.REDIS_URL)
            await self.redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_client = None
    
    async def close(self):
        """
        Close cache connections
        """
        if self.redis_client:
            await self.redis_client.close()
    
    async def health_check(self) -> bool:
        """
        Check cache health
        """
        if not self.redis_client:
            return False
        try:
            await self.redis_client.ping()
            return True
        except Exception:
            return False


# Global cache manager instance
cache_manager = CacheManager() 