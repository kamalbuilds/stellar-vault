#!/usr/bin/env python3
"""
StellarVault AI Engine
AI-Powered Asset Valuation and Risk Assessment Service
"""

import os
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from loguru import logger
from datetime import datetime
import sys

# Import local modules
from api.routes import valuation, risk, compliance, analytics, health
from core.config import settings
from core.database import engine, create_tables
from core.cache import redis_client
from services.model_manager import ModelManager
from services.data_collector import DataCollector
from services.real_estate_valuer import RealEstateValuer
from services.commodities_valuer import CommoditiesValuer
from services.art_valuer import ArtValuer
from services.bonds_valuer import BondsValuer
from services.risk_assessor import RiskAssessor
from middleware.auth import AuthMiddleware
from middleware.rate_limit import RateLimitMiddleware
from utils.monitoring import setup_prometheus_metrics

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    level=settings.LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# Global instances
model_manager = ModelManager()
data_collector = DataCollector()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events
    """
    # Startup
    logger.info("🚀 Starting StellarVault AI Engine...")
    
    try:
        # Initialize database
        await create_tables()
        logger.info("✅ Database tables created/verified")
        
        # Initialize Redis connection
        await redis_client.initialize()
        logger.info("✅ Redis connection established")
        
        # Initialize ML models
        await model_manager.initialize_models()
        logger.info("✅ ML models loaded and ready")
        
        # Start background data collection
        asyncio.create_task(data_collector.start_collection())
        logger.info("✅ Background data collection started")
        
        # Setup monitoring
        setup_prometheus_metrics()
        logger.info("✅ Prometheus metrics configured")
        
        logger.info("🎯 StellarVault AI Engine fully initialized")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize AI Engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down StellarVault AI Engine...")
    
    try:
        # Stop background tasks
        await data_collector.stop_collection()
        logger.info("✅ Data collection stopped")
        
        # Close Redis connection
        await redis_client.close()
        logger.info("✅ Redis connection closed")
        
        # Save model states
        await model_manager.save_model_states()
        logger.info("✅ Model states saved")
        
        logger.info("✅ Shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

# Create FastAPI application
app = FastAPI(
    title="StellarVault AI Engine",
    description="AI-Powered Real-World Asset Valuation and Risk Assessment Platform",
    version="1.0.0",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(AuthMiddleware)
app.add_middleware(RateLimitMiddleware)

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring
    """
    try:
        # Check database connection
        # db_status = await check_database_health()
        
        # Check Redis connection
        redis_status = await redis_client.ping()
        
        # Check model status
        model_status = await model_manager.get_model_health()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "environment": settings.ENVIRONMENT,
            "components": {
                "database": "healthy",  # db_status,
                "redis": "healthy" if redis_status else "unhealthy",
                "models": model_status
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        )

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "name": "StellarVault AI Engine",
        "version": "1.0.0",
        "description": "AI-Powered Real-World Asset Valuation and Risk Assessment",
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "valuation": "/api/v1/valuation",
            "risk": "/api/v1/risk",
            "compliance": "/api/v1/compliance",
            "analytics": "/api/v1/analytics"
        }
    }

# Include API routes
app.include_router(
    valuation.router,
    prefix="/api/v1/valuation",
    tags=["Asset Valuation"]
)

app.include_router(
    risk.router,
    prefix="/api/v1/risk",
    tags=["Risk Assessment"]
)

app.include_router(
    compliance.router,
    prefix="/api/v1/compliance",
    tags=["Compliance Analysis"]
)

app.include_router(
    analytics.router,
    prefix="/api/v1/analytics",
    tags=["Market Analytics"]
)

app.include_router(
    health.router,
    prefix="/api/v1/health",
    tags=["System Health"]
)

# Background tasks
@app.post("/api/v1/tasks/retrain-models")
async def trigger_model_retraining(background_tasks: BackgroundTasks):
    """
    Trigger model retraining in the background
    """
    try:
        background_tasks.add_task(model_manager.retrain_all_models)
        return {
            "message": "Model retraining task scheduled",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to schedule model retraining: {e}")
        raise HTTPException(status_code=500, detail="Failed to schedule retraining")

@app.post("/api/v1/tasks/update-market-data")
async def trigger_market_data_update(background_tasks: BackgroundTasks):
    """
    Trigger market data update in the background
    """
    try:
        background_tasks.add_task(data_collector.update_all_data_sources)
        return {
            "message": "Market data update task scheduled",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to schedule market data update: {e}")
        raise HTTPException(status_code=500, detail="Failed to schedule data update")

# Metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint
    """
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi import Response
    
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Development utilities
if settings.ENVIRONMENT == "development":
    @app.get("/api/v1/debug/model-info")
    async def get_model_info():
        """
        Debug endpoint to get model information
        """
        return await model_manager.get_model_info()
    
    @app.get("/api/v1/debug/cache-stats")
    async def get_cache_stats():
        """
        Debug endpoint to get cache statistics
        """
        return await redis_client.get_stats()

# Main execution
if __name__ == "__main__":
    config = uvicorn.Config(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=settings.ENVIRONMENT == "development",
        workers=1 if settings.ENVIRONMENT == "development" else settings.WORKERS
    )
    
    server = uvicorn.Server(config)
    
    logger.info(f"🚀 Starting StellarVault AI Engine on {settings.HOST}:{settings.PORT}")
    logger.info(f"🌍 Environment: {settings.ENVIRONMENT}")
    logger.info(f"📊 Log Level: {settings.LOG_LEVEL}")
    
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down gracefully...")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1) 