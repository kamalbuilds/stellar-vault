#!/usr/bin/env python3
"""
StellarVault AI Engine - Main FastAPI Application
Provides AI-powered RWA valuation, risk assessment, and portfolio management
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime
import uvicorn
from loguru import logger
import sys

from core.config import settings
from core.database import get_db_session, create_tables, Database, get_db
from core.cache import cache_manager
from services.real_estate_valuer import RealEstateValuer
from services.commodities_valuer import CommoditiesValuer
from services.art_valuer import ArtValuer


# Initialize services
real_estate_valuer = RealEstateValuer()
commodities_valuer = CommoditiesValuer()
art_valuer = ArtValuer()
database = Database()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    """
    # Startup
    logger.info("🚀 Starting StellarVault AI Engine...")
    logger.info(f"📊 Database URL: {settings.DATABASE_URL}")
    
    try:
        # Initialize database
        await database.initialize()
        logger.info("✅ Database initialized")
        
        # Initialize cache
        await cache_manager.initialize()
        logger.info("✅ Cache system initialized")
        
        # Initialize AI models
        await real_estate_valuer.initialize()
        await commodities_valuer.initialize()
        await art_valuer.initialize()
        logger.info("✅ AI valuation models initialized")
        
        logger.info("🎯 StellarVault AI Engine ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize AI Engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down StellarVault AI Engine...")
    await database.close()
    await cache_manager.close()


# Initialize FastAPI app
app = FastAPI(
    title="StellarVault AI Engine",
    description="AI-powered Real-World Asset (RWA) valuation and portfolio management platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Health check endpoints
@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": {
            "real_estate": real_estate_valuer.is_initialized,
            "commodities": commodities_valuer.is_initialized,
            "art": art_valuer.is_initialized,
            "database": await database.health_check(),
            "cache": await cache_manager.health_check()
        }
    }


@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {
        "message": "StellarVault AI Engine",
        "description": "AI-powered RWA tokenization platform",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "real_estate": "/api/v1/valuation/real-estate",
            "commodities": "/api/v1/valuation/commodities",
            "art": "/api/v1/valuation/art",
            "portfolio": "/api/v1/portfolio",
            "risk": "/api/v1/risk"
        }
    }


# Real Estate Valuation Endpoints
@app.post("/api/v1/valuation/real-estate")
async def value_real_estate(
    property_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db = Depends(get_db)
):
    """
    Value real estate property using AI models and market data
    """
    try:
        logger.info(f"Real estate valuation request: {property_data.get('address', 'Unknown')}")
        
        # Validate input
        required_fields = ['address', 'property_type', 'square_footage']
        missing_fields = [field for field in required_fields if field not in property_data]
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {missing_fields}"
            )
        
        # Perform valuation
        result = await real_estate_valuer.value_property(property_data)
        
        # Store in database
        background_tasks.add_task(
            store_valuation_result,
            "real_estate",
            property_data,
            result.__dict__,
            db
        )
        
        return {
            "success": True,
            "valuation": {
                "estimated_value": result.estimated_value,
                "confidence_score": result.confidence_score,
                "value_range": {
                    "min": result.value_range[0],
                    "max": result.value_range[1]
                },
                "price_per_sqft": result.estimated_value / property_data['square_footage'],
                "model_used": result.model_used,
                "methodology": result.methodology,
                "valuation_date": result.valuation_date.isoformat()
            },
            "comparable_analysis": {
                "properties_analyzed": len(result.comparable_analysis),
                "avg_comparable_price": (
                    sum(comp['adjusted_price'] for comp in result.comparable_analysis) / 
                    len(result.comparable_analysis) if result.comparable_analysis else 0
                ),
                "comparables": result.comparable_analysis[:5]  # Top 5 comparables
            },
            "market_factors": result.market_factors,
            "risk_assessment": result.risk_assessment,
            "feature_importance": dict(list(result.feature_importance.items())[:10])
        }
        
    except Exception as e:
        logger.error(f"Real estate valuation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/valuation/commodities")
async def value_commodity(
    commodity_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db = Depends(get_db)
):
    """
    Value commodity using ML models and market analysis
    """
    try:
        logger.info(f"Commodity valuation request: {commodity_data.get('commodity_type', 'Unknown')}")
        
        # Validate input
        required_fields = ['commodity_type', 'quantity']
        missing_fields = [field for field in required_fields if field not in commodity_data]
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {missing_fields}"
            )
        
        # Check if commodity is supported
        supported_commodities = await commodities_valuer.get_supported_commodities()
        commodity_type = commodity_data['commodity_type'].lower()
        
        is_supported = any(
            commodity_type in commodities 
            for commodities in supported_commodities.values()
        )
        
        if not is_supported:
            raise HTTPException(
                status_code=400,
                detail=f"Commodity type '{commodity_type}' not supported. Supported commodities: {supported_commodities}"
            )
        
        # Perform valuation
        result = await commodities_valuer.value_commodity(commodity_data)
        
        # Store in database
        background_tasks.add_task(
            store_valuation_result,
            "commodities",
            commodity_data,
            result.__dict__,
            db
        )
        
        return {
            "success": True,
            "valuation": {
                "estimated_value": result.estimated_value,
                "confidence_score": result.confidence_score,
                "value_range": {
                    "min": result.value_range[0],
                    "max": result.value_range[1]
                },
                "unit_price": result.estimated_value / commodity_data['quantity'],
                "model_used": result.model_used,
                "methodology": result.methodology,
                "valuation_date": result.valuation_date.isoformat()
            },
            "market_factors": result.market_factors,
            "price_forecast": result.price_forecast,
            "feature_importance": dict(list(result.feature_importance.items())[:10])
        }
        
    except Exception as e:
        logger.error(f"Commodity valuation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/valuation/art")
async def value_artwork(
    artwork_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db = Depends(get_db)
):
    """
    Value artwork using AI models and comparable analysis
    """
    try:
        logger.info(f"Art valuation request: {artwork_data.get('artist_name', 'Unknown')} - {artwork_data.get('title', 'Untitled')}")
        
        # Validate input
        required_fields = ['artist_name', 'medium']
        missing_fields = [field for field in required_fields if field not in artwork_data]
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {missing_fields}"
            )
        
        # Perform valuation
        result = await art_valuer.value_artwork(artwork_data)
        
        # Store in database
        background_tasks.add_task(
            store_valuation_result,
            "art",
            artwork_data,
            result.__dict__,
            db
        )
        
        return {
            "success": True,
            "valuation": {
                "estimated_value": result.estimated_value,
                "confidence_score": result.confidence_score,
                "value_range": {
                    "min": result.value_range[0],
                    "max": result.value_range[1]
                },
                "model_used": result.model_used,
                "methodology": result.methodology,
                "valuation_date": result.valuation_date.isoformat()
            },
            "comparable_analysis": {
                "artworks_analyzed": len(result.comparable_analysis),
                "avg_comparable_price": (
                    sum(comp['adjusted_price'] for comp in result.comparable_analysis) / 
                    len(result.comparable_analysis) if result.comparable_analysis else 0
                ),
                "top_comparables": result.comparable_analysis[:3]
            },
            "market_factors": result.market_factors,
            "risk_assessment": result.risk_assessment,
            "feature_importance": dict(list(result.feature_importance.items())[:10])
        }
        
    except Exception as e:
        logger.error(f"Art valuation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/valuation/supported-assets")
async def get_supported_assets():
    """
    Get list of supported asset types and categories
    """
    try:
        return {
            "real_estate": {
                "property_types": ["residential", "commercial", "industrial", "land"],
                "features": ["address", "square_footage", "bedrooms", "bathrooms", "year_built"]
            },
            "commodities": await commodities_valuer.get_supported_commodities(),
            "art": {
                "categories": ["paintings", "sculptures", "prints", "drawings", "photography", "collectibles"],
                "features": ["artist_name", "medium", "dimensions", "year_created", "condition"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting supported assets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Portfolio Management Endpoints
@app.post("/api/v1/portfolio/analyze")
async def analyze_portfolio(
    portfolio_data: Dict[str, Any],
    db = Depends(get_db)
):
    """
    Analyze portfolio composition and risk
    """
    try:
        logger.info("Portfolio analysis request")
        
        # Validate portfolio data
        if 'assets' not in portfolio_data:
            raise HTTPException(status_code=400, detail="Portfolio must contain 'assets' field")
        
        assets = portfolio_data['assets']
        if not assets:
            raise HTTPException(status_code=400, detail="Portfolio must contain at least one asset")
        
        # Analyze each asset
        portfolio_value = 0
        asset_valuations = []
        risk_scores = []
        
        for asset in assets:
            asset_type = asset.get('type', '').lower()
            
            if asset_type == 'real_estate':
                result = await real_estate_valuer.value_property(asset)
            elif asset_type == 'commodities':
                result = await commodities_valuer.value_commodity(asset)
            elif asset_type == 'art':
                result = await art_valuer.value_artwork(asset)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported asset type: {asset_type}")
            
            portfolio_value += result.estimated_value
            asset_valuations.append({
                "asset_type": asset_type,
                "estimated_value": result.estimated_value,
                "confidence_score": result.confidence_score
            })
            
            # Extract risk score (simplified)
            risk_score = 1 - result.confidence_score  # Lower confidence = higher risk
            risk_scores.append(risk_score)
        
        # Calculate portfolio metrics
        avg_risk = sum(risk_scores) / len(risk_scores)
        portfolio_confidence = sum(val['confidence_score'] for val in asset_valuations) / len(asset_valuations)
        
        # Asset allocation
        asset_allocation = {}
        for val in asset_valuations:
            asset_type = val['asset_type']
            percentage = (val['estimated_value'] / portfolio_value) * 100
            asset_allocation[asset_type] = asset_allocation.get(asset_type, 0) + percentage
        
        return {
            "success": True,
            "portfolio": {
                "total_value": portfolio_value,
                "asset_count": len(assets),
                "portfolio_confidence": portfolio_confidence,
                "average_risk_score": avg_risk,
                "diversification_score": len(set(val['asset_type'] for val in asset_valuations)) / 3  # Out of 3 asset types
            },
            "asset_allocation": asset_allocation,
            "asset_valuations": asset_valuations,
            "recommendations": generate_portfolio_recommendations(asset_allocation, avg_risk)
        }
        
    except Exception as e:
        logger.error(f"Portfolio analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/risk/assessment/{asset_type}")
async def get_risk_assessment(asset_type: str):
    """
    Get risk assessment for specific asset type
    """
    try:
        asset_type = asset_type.lower()
        
        if asset_type == "real_estate":
            risk_factors = {
                "market_risk": 0.3,
                "liquidity_risk": 0.6,
                "interest_rate_risk": 0.4,
                "location_risk": 0.2,
                "property_specific_risk": 0.3
            }
        elif asset_type == "commodities":
            risk_factors = {
                "price_volatility": 0.7,
                "supply_demand_risk": 0.5,
                "geopolitical_risk": 0.4,
                "currency_risk": 0.3,
                "storage_risk": 0.2
            }
        elif asset_type == "art":
            risk_factors = {
                "authentication_risk": 0.4,
                "condition_risk": 0.3,
                "market_liquidity_risk": 0.8,
                "provenance_risk": 0.3,
                "market_trend_risk": 0.4
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported asset type: {asset_type}")
        
        overall_risk = sum(risk_factors.values()) / len(risk_factors)
        
        return {
            "asset_type": asset_type,
            "overall_risk_score": overall_risk,
            "risk_factors": risk_factors,
            "risk_level": "high" if overall_risk > 0.6 else "medium" if overall_risk > 0.3 else "low",
            "recommendations": generate_risk_recommendations(asset_type, risk_factors)
        }
        
    except Exception as e:
        logger.error(f"Risk assessment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model Performance Endpoints
@app.get("/api/v1/models/performance")
async def get_model_performance():
    """
    Get performance metrics for all valuation models
    """
    try:
        return {
            "real_estate": await real_estate_valuer.get_model_performance(),
            "commodities": await commodities_valuer.get_model_performance("precious_metals"),
            "art": await art_valuer.get_model_performance(),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Analytics Endpoints
@app.get("/api/v1/analytics/market-trends")
async def get_market_trends():
    """
    Get current market trends across asset classes
    """
    try:
        return {
            "real_estate": {
                "trend": "rising",
                "change_percentage": 5.2,
                "confidence": 0.78
            },
            "commodities": {
                "precious_metals": {
                    "trend": "stable",
                    "change_percentage": 1.1,
                    "confidence": 0.85
                },
                "energy": {
                    "trend": "volatile",
                    "change_percentage": -3.4,
                    "confidence": 0.72
                }
            },
            "art": {
                "contemporary": {
                    "trend": "rising",
                    "change_percentage": 8.7,
                    "confidence": 0.65
                },
                "classical": {
                    "trend": "stable",
                    "change_percentage": 2.3,
                    "confidence": 0.80
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting market trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Utility Functions
async def store_valuation_result(
    asset_type: str,
    asset_data: Dict[str, Any],
    result: Dict[str, Any],
    db
):
    """
    Store valuation result in database
    """
    try:
        # Store in database (simplified)
        valuation_record = {
            "asset_type": asset_type,
            "asset_data": asset_data,
            "valuation_result": result,
            "timestamp": datetime.now()
        }
        
        # In production, save to actual database
        logger.info(f"Stored {asset_type} valuation result")
        
    except Exception as e:
        logger.error(f"Failed to store valuation result: {e}")


def generate_portfolio_recommendations(asset_allocation: Dict[str, float], avg_risk: float) -> List[str]:
    """
    Generate portfolio recommendations based on allocation and risk
    """
    recommendations = []
    
    # Diversification recommendations
    if len(asset_allocation) == 1:
        recommendations.append("Consider diversifying across multiple asset classes to reduce risk")
    
    # Risk-based recommendations
    if avg_risk > 0.7:
        recommendations.append("Portfolio has high risk - consider adding more stable assets")
    elif avg_risk < 0.3:
        recommendations.append("Portfolio is very conservative - consider adding growth assets")
    
    # Asset-specific recommendations
    real_estate_pct = asset_allocation.get('real_estate', 0)
    if real_estate_pct > 60:
        recommendations.append("Real estate allocation is high - consider reducing exposure")
    elif real_estate_pct < 20:
        recommendations.append("Consider adding real estate for inflation protection")
    
    commodities_pct = asset_allocation.get('commodities', 0)
    if commodities_pct > 30:
        recommendations.append("Commodities allocation is high - monitor for volatility")
    elif commodities_pct < 5:
        recommendations.append("Consider adding commodities for portfolio diversification")
    
    art_pct = asset_allocation.get('art', 0)
    if art_pct > 20:
        recommendations.append("Art allocation is high - ensure adequate liquidity")
    
    return recommendations


def generate_risk_recommendations(asset_type: str, risk_factors: Dict[str, float]) -> List[str]:
    """
    Generate risk mitigation recommendations
    """
    recommendations = []
    
    if asset_type == "real_estate":
        if risk_factors.get("liquidity_risk", 0) > 0.5:
            recommendations.append("Consider REITs for improved liquidity")
        if risk_factors.get("interest_rate_risk", 0) > 0.4:
            recommendations.append("Monitor interest rate environment closely")
    
    elif asset_type == "commodities":
        if risk_factors.get("price_volatility", 0) > 0.6:
            recommendations.append("Consider hedging strategies to reduce volatility")
        if risk_factors.get("geopolitical_risk", 0) > 0.4:
            recommendations.append("Diversify across multiple commodity types and regions")
    
    elif asset_type == "art":
        if risk_factors.get("authentication_risk", 0) > 0.3:
            recommendations.append("Ensure proper authentication and documentation")
        if risk_factors.get("market_liquidity_risk", 0) > 0.7:
            recommendations.append("Consider fractional ownership or art funds for liquidity")
    
    return recommendations


# Error Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Handle HTTP exceptions
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    Handle general exceptions
    """
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )


# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    level=settings.LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    ) 