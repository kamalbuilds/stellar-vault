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
from services.bonds_valuer import BondsValuationService
from risk.portfolio_optimizer import PortfolioOptimizer
from risk.risk_assessor import RiskAssessor
from analytics.market_analytics import MarketAnalyticsService
from compliance.kyc_aml_service import KYCAMLService


# Initialize services
real_estate_valuer = RealEstateValuer()
commodities_valuer = CommoditiesValuer()
art_valuer = ArtValuer()
bonds_valuer = BondsValuationService()
portfolio_optimizer = PortfolioOptimizer()
risk_assessor = RiskAssessor()
market_analytics = MarketAnalyticsService()
kyc_aml_service = KYCAMLService()
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
        await bonds_valuer.initialize()
        await risk_assessor.initialize()
        await market_analytics.initialize()
        await kyc_aml_service.initialize()
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
            "bonds": getattr(bonds_valuer, 'is_initialized', True),
            "portfolio_optimizer": True,
            "risk_assessor": getattr(risk_assessor, 'is_initialized', True),
            "market_analytics": getattr(market_analytics, 'is_initialized', True),
            "kyc_aml": getattr(kyc_aml_service, 'is_initialized', True),
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
            "bonds": "/api/v1/valuation/bonds",
            "portfolio": "/api/v1/portfolio",
            "risk": "/api/v1/risk",
            "analytics": "/api/v1/analytics",
            "compliance": "/api/v1/compliance"
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


@app.post("/api/v1/valuation/bonds")
async def value_bond(
    bond_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db = Depends(get_db)
):
    """
    Value bonds using comprehensive fixed-income models
    """
    try:
        logger.info(f"Bond valuation request: {bond_data.get('issuer', 'Unknown')}")
        
        # Validate input
        required_fields = ['face_value', 'coupon_rate', 'maturity_date', 'credit_rating', 'issuer']
        missing_fields = [field for field in required_fields if field not in bond_data]
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {missing_fields}"
            )
        
        # Parse bond data
        from services.bonds_valuer import BondCharacteristics, BondType
        from datetime import datetime
        
        # Convert string dates to datetime objects
        maturity_date = datetime.fromisoformat(bond_data['maturity_date'].replace('Z', '+00:00'))
        issue_date = datetime.fromisoformat(bond_data.get('issue_date', '2020-01-01').replace('Z', '+00:00'))
        
        # Create bond characteristics
        bond_characteristics = BondCharacteristics(
            face_value=float(bond_data['face_value']),
            coupon_rate=float(bond_data['coupon_rate']),
            maturity_date=maturity_date,
            issue_date=issue_date,
            credit_rating=bond_data['credit_rating'],
            issuer=bond_data['issuer'],
            bond_type=BondType(bond_data.get('bond_type', 'corporate')),
            callable=bond_data.get('callable', False),
            call_price=bond_data.get('call_price'),
            call_date=datetime.fromisoformat(bond_data['call_date'].replace('Z', '+00:00')) if bond_data.get('call_date') else None,
            frequency=int(bond_data.get('frequency', 2))
        )
        
        # Perform valuation
        result = await bonds_valuer.valuate_bond(bond_characteristics)
        
        # Store in database
        background_tasks.add_task(
            store_valuation_result,
            "bonds",
            bond_data,
            result.__dict__,
            db
        )
        
        return {
            "success": True,
            "valuation": {
                "present_value": result.present_value,
                "clean_price": result.clean_price,
                "dirty_price": result.dirty_price,
                "accrued_interest": result.accrued_interest,
                "yield_to_maturity": result.yield_to_maturity,
                "modified_duration": result.modified_duration,
                "convexity": result.convexity,
                "credit_spread": result.credit_spread,
                "risk_premium": result.risk_premium,
                "confidence_score": result.confidence_score,
                "methodology": result.methodology,
                "valuation_date": result.valuation_date.isoformat()
            },
            "risk_metrics": {
                "interest_rate_sensitivity": result.modified_duration,
                "credit_risk": result.credit_spread,
                "price_volatility": result.convexity,
                "risk_premium": result.risk_premium
            },
            "market_data": result.market_data
        }
        
    except Exception as e:
        logger.error(f"Bond valuation error: {e}")
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


@app.post("/api/v1/portfolio/optimize")
async def optimize_portfolio(
    optimization_request: Dict[str, Any],
    db = Depends(get_db)
):
    """
    Optimize portfolio using advanced algorithms (MPT, Risk Parity, Black-Litterman)
    """
    try:
        logger.info("Portfolio optimization request")
        
        # Validate input
        required_fields = ['assets', 'method']
        missing_fields = [field for field in required_fields if field not in optimization_request]
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {missing_fields}"
            )
        
        # Import necessary classes
        from risk.portfolio_optimizer import OptimizationMethod, AssetData, PortfolioConstraints
        
        # Parse optimization method
        method_str = optimization_request['method'].upper()
        try:
            method = OptimizationMethod(method_str.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid optimization method: {method_str}. Supported: {[m.value for m in OptimizationMethod]}"
            )
        
        # Prepare asset data
        assets = []
        for asset_info in optimization_request['assets']:
            # Generate mock historical returns for demonstration
            # In production, this would fetch real historical data
            import numpy as np
            np.random.seed(42)  # For reproducible results
            returns = np.random.normal(0.08, 0.15, 252).tolist()  # Daily returns for 1 year
            prices = [100 * (1 + sum(returns[:i+1])) for i in range(len(returns))]
            
            asset = AssetData(
                symbol=asset_info.get('symbol', f"ASSET_{len(assets)}"),
                name=asset_info.get('name', 'Unknown Asset'),
                asset_class=asset_info.get('asset_class', 'unknown'),
                returns=returns,
                prices=prices,
                market_cap=asset_info.get('market_cap'),
                expected_return=asset_info.get('expected_return'),
                volatility=asset_info.get('volatility')
            )
            assets.append(asset)
        
        # Parse constraints
        constraints_data = optimization_request.get('constraints', {})
        constraints = PortfolioConstraints(
            min_weight=constraints_data.get('min_weight', 0.0),
            max_weight=constraints_data.get('max_weight', 1.0),
            target_return=constraints_data.get('target_return'),
            max_risk=constraints_data.get('max_risk')
        )
        
        # Parse investor views for Black-Litterman
        views = optimization_request.get('views', {})
        
        # Perform optimization
        result = await portfolio_optimizer.optimize_portfolio(
            assets=assets,
            method=method,
            constraints=constraints,
            views=views
        )
        
        return {
            "success": True,
            "optimization": {
                "method": result.optimization_method,
                "weights": result.weights,
                "expected_return": result.expected_return,
                "expected_volatility": result.expected_volatility,
                "sharpe_ratio": result.sharpe_ratio,
                "var_95": result.var_95,
                "var_99": result.var_99,
                "cvar_95": result.cvar_95,
                "maximum_drawdown": result.maximum_drawdown,
                "diversification_ratio": result.diversification_ratio,
                "confidence_score": result.confidence_score,
                "constraints_satisfied": result.constraints_satisfied,
                "optimization_date": result.optimization_date.isoformat()
            },
            "risk_metrics": {
                "risk_contribution": result.risk_contribution,
                "performance_metrics": result.performance_metrics
            }
        }
        
    except Exception as e:
        logger.error(f"Portfolio optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/portfolio/rebalance")
async def rebalance_portfolio(
    rebalance_request: Dict[str, Any],
    db = Depends(get_db)
):
    """
    Rebalance existing portfolio to target allocation
    """
    try:
        logger.info("Portfolio rebalance request")
        
        # Validate input
        required_fields = ['current_holdings', 'target_allocation']
        missing_fields = [field for field in required_fields if field not in rebalance_request]
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {missing_fields}"
            )
        
        current_holdings = rebalance_request['current_holdings']
        target_allocation = rebalance_request['target_allocation']
        
        # Calculate current portfolio value
        total_value = sum(holding.get('value', 0) for holding in current_holdings)
        
        # Calculate rebalancing trades
        trades = []
        for asset_symbol, target_weight in target_allocation.items():
            target_value = total_value * target_weight
            
            # Find current holding
            current_holding = next(
                (h for h in current_holdings if h.get('symbol') == asset_symbol),
                {'value': 0, 'quantity': 0}
            )
            current_value = current_holding.get('value', 0)
            
            # Calculate trade
            trade_value = target_value - current_value
            trade_type = "buy" if trade_value > 0 else "sell"
            
            if abs(trade_value) > total_value * 0.01:  # Only trades > 1% of portfolio
                trades.append({
                    "symbol": asset_symbol,
                    "action": trade_type,
                    "value": abs(trade_value),
                    "current_weight": current_value / total_value,
                    "target_weight": target_weight,
                    "weight_change": target_weight - (current_value / total_value)
                })
        
        # Calculate transaction costs (simplified)
        transaction_cost_rate = rebalance_request.get('transaction_cost_rate', 0.001)
        total_transaction_costs = sum(trade['value'] * transaction_cost_rate for trade in trades)
        
        return {
            "success": True,
            "rebalancing": {
                "current_portfolio_value": total_value,
                "target_allocation": target_allocation,
                "required_trades": trades,
                "total_transaction_costs": total_transaction_costs,
                "transaction_cost_percentage": (total_transaction_costs / total_value) * 100,
                "net_rebalancing_benefit": len(trades) * 0.02,  # Simplified benefit calculation
                "rebalance_recommended": len(trades) > 0
            }
        }
        
    except Exception as e:
        logger.error(f"Portfolio rebalance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/portfolio/methods")
async def get_optimization_methods():
    """
    Get available portfolio optimization methods and their descriptions
    """
    try:
        from risk.portfolio_optimizer import OptimizationMethod
        
        return {
            "optimization_methods": {
                "mean_variance": {
                    "name": "Modern Portfolio Theory (MPT)",
                    "description": "Optimizes portfolio for maximum return given risk level",
                    "best_for": "Balanced risk-return optimization",
                    "parameters": ["target_return", "risk_tolerance"]
                },
                "risk_parity": {
                    "name": "Risk Parity",
                    "description": "Allocates risk equally across all assets",
                    "best_for": "Risk diversification focus",
                    "parameters": ["risk_budget"]
                },
                "black_litterman": {
                    "name": "Black-Litterman",
                    "description": "Incorporates investor views into optimization",
                    "best_for": "When you have market views/predictions",
                    "parameters": ["investor_views", "confidence_levels"]
                },
                "minimum_variance": {
                    "name": "Minimum Variance",
                    "description": "Minimizes portfolio volatility",
                    "best_for": "Conservative, stability-focused portfolios",
                    "parameters": ["weight_constraints"]
                },
                "maximum_sharpe": {
                    "name": "Maximum Sharpe Ratio",
                    "description": "Maximizes risk-adjusted returns",
                    "best_for": "Efficient risk-return balance",
                    "parameters": ["risk_free_rate"]
                },
                "maximum_diversification": {
                    "name": "Maximum Diversification",
                    "description": "Maximizes diversification benefits",
                    "best_for": "Broad diversification strategy",
                    "parameters": ["correlation_matrix"]
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting optimization methods: {e}")
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


@app.post("/api/v1/risk/portfolio-analysis")
async def analyze_portfolio_risk(
    risk_request: Dict[str, Any],
    db = Depends(get_db)
):
    """
    Comprehensive portfolio risk analysis using advanced risk models
    """
    try:
        logger.info("Portfolio risk analysis request")
        
        # Validate input
        required_fields = ['portfolio_holdings']
        missing_fields = [field for field in required_fields if field not in risk_request]
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {missing_fields}"
            )
        
        # Import necessary classes
        from risk.risk_assessor import PortfolioData, AssetData
        
        # Prepare portfolio data
        holdings = risk_request['portfolio_holdings']
        assets = []
        
        for holding in holdings:
            # Generate mock price data for demonstration
            # In production, this would fetch real market data
            import numpy as np
            np.random.seed(hash(holding.get('symbol', 'ASSET')) % 2**32)
            price_data = np.random.normal(100, 15, 252)  # 252 trading days
            returns_data = np.diff(price_data) / price_data[:-1]
            
            asset = AssetData(
                symbol=holding.get('symbol', 'UNKNOWN'),
                asset_class=holding.get('asset_class', 'unknown'),
                price_data=price_data.tolist(),
                returns_data=returns_data.tolist(),
                market_value=holding.get('market_value', 0),
                quantity=holding.get('quantity', 0)
            )
            assets.append(asset)
        
        portfolio = PortfolioData(
            portfolio_id=risk_request.get('portfolio_id', 'PORTFOLIO_001'),
            assets=assets,
            base_currency=risk_request.get('base_currency', 'USD'),
            benchmark=risk_request.get('benchmark', 'SPY')
        )
        
        # Perform comprehensive risk analysis
        result = await risk_assessor.analyze_portfolio_risk(portfolio)
        
        return {
            "success": True,
            "risk_analysis": {
                "portfolio_id": result.portfolio_id,
                "total_portfolio_value": result.total_portfolio_value,
                "portfolio_volatility": result.portfolio_volatility,
                "sharpe_ratio": result.sharpe_ratio,
                "sortino_ratio": result.sortino_ratio,
                "maximum_drawdown": result.maximum_drawdown,
                "beta": result.beta,
                "analysis_date": result.analysis_date.isoformat()
            },
            "value_at_risk": {
                "var_95_1d": result.var_metrics['var_95_1d'],
                "var_99_1d": result.var_metrics['var_99_1d'],
                "var_95_10d": result.var_metrics['var_95_10d'],
                "cvar_95": result.var_metrics['cvar_95'],
                "confidence_level": result.var_metrics['confidence_level']
            },
            "risk_decomposition": result.risk_decomposition,
            "correlation_analysis": result.correlation_analysis,
            "risk_score": result.overall_risk_score,
            "recommendations": result.recommendations,
            "confidence_score": result.confidence_score
        }
        
    except Exception as e:
        logger.error(f"Portfolio risk analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/risk/stress-test")
async def perform_stress_test(
    stress_test_request: Dict[str, Any],
    db = Depends(get_db)
):
    """
    Perform stress testing on portfolio under various scenarios
    """
    try:
        logger.info("Stress test request")
        
        # Validate input
        required_fields = ['portfolio_holdings']
        missing_fields = [field for field in required_fields if field not in stress_test_request]
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {missing_fields}"
            )
        
        # Import necessary classes
        from risk.risk_assessor import PortfolioData, AssetData, StressTestScenario
        
        # Prepare portfolio data (similar to risk analysis)
        holdings = stress_test_request['portfolio_holdings']
        assets = []
        
        for holding in holdings:
            import numpy as np
            np.random.seed(hash(holding.get('symbol', 'ASSET')) % 2**32)
            price_data = np.random.normal(100, 15, 252)
            returns_data = np.diff(price_data) / price_data[:-1]
            
            asset = AssetData(
                symbol=holding.get('symbol', 'UNKNOWN'),
                asset_class=holding.get('asset_class', 'unknown'),
                price_data=price_data.tolist(),
                returns_data=returns_data.tolist(),
                market_value=holding.get('market_value', 0),
                quantity=holding.get('quantity', 0)
            )
            assets.append(asset)
        
        portfolio = PortfolioData(
            portfolio_id=stress_test_request.get('portfolio_id', 'PORTFOLIO_001'),
            assets=assets,
            base_currency=stress_test_request.get('base_currency', 'USD')
        )
        
        # Define stress test scenarios
        scenarios = stress_test_request.get('scenarios', ['market_crash', 'interest_rate_shock', 'inflation_surge'])
        
        # Perform stress testing
        result = await risk_assessor.perform_stress_test(portfolio, scenarios)
        
        return {
            "success": True,
            "stress_test": {
                "portfolio_id": result.portfolio_id,
                "base_portfolio_value": result.base_portfolio_value,
                "test_date": result.test_date.isoformat(),
                "scenarios_tested": scenarios
            },
            "scenario_results": result.scenario_results,
            "worst_case_scenario": result.worst_case_scenario,
            "portfolio_resilience_score": result.portfolio_resilience_score,
            "stress_test_summary": {
                "max_loss": max([s['portfolio_change_percent'] for s in result.scenario_results.values()]),
                "avg_loss": sum([s['portfolio_change_percent'] for s in result.scenario_results.values()]) / len(result.scenario_results),
                "scenarios_with_major_loss": len([s for s in result.scenario_results.values() if s['portfolio_change_percent'] < -20])
            },
            "recommendations": result.recommendations
        }
        
    except Exception as e:
        logger.error(f"Stress test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/risk/var-calculation/{portfolio_id}")
async def calculate_var(
    portfolio_id: str,
    confidence_level: float = 0.95,
    time_horizon: int = 1,
    method: str = "historical"
):
    """
    Calculate Value at Risk (VaR) for a specific portfolio
    """
    try:
        logger.info(f"VaR calculation request for portfolio: {portfolio_id}")
        
        # Validate parameters
        if confidence_level <= 0 or confidence_level >= 1:
            raise HTTPException(status_code=400, detail="Confidence level must be between 0 and 1")
        
        if time_horizon <= 0:
            raise HTTPException(status_code=400, detail="Time horizon must be positive")
        
        valid_methods = ["historical", "parametric", "monte_carlo"]
        if method not in valid_methods:
            raise HTTPException(status_code=400, detail=f"Method must be one of: {valid_methods}")
        
        # Mock portfolio data for demonstration
        # In production, fetch real portfolio data from database
        import numpy as np
        np.random.seed(hash(portfolio_id) % 2**32)
        portfolio_returns = np.random.normal(0.0008, 0.02, 1000)  # Daily returns
        portfolio_value = 1000000  # $1M portfolio
        
        # Perform VaR calculation using the risk assessor
        var_result = await risk_assessor.calculate_var(
            returns_data=portfolio_returns.tolist(),
            portfolio_value=portfolio_value,
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            method=method
        )
        
        return {
            "success": True,
            "var_calculation": {
                "portfolio_id": portfolio_id,
                "portfolio_value": portfolio_value,
                "confidence_level": confidence_level,
                "time_horizon_days": time_horizon,
                "method": method,
                "calculation_date": datetime.now().isoformat()
            },
            "var_metrics": {
                "var_absolute": var_result.var_absolute,
                "var_percentage": var_result.var_percentage,
                "cvar_absolute": var_result.cvar_absolute,
                "cvar_percentage": var_result.cvar_percentage,
                "confidence_interval": var_result.confidence_interval
            },
            "risk_interpretation": {
                "interpretation": f"There is a {(1-confidence_level)*100}% chance of losing more than ${var_result.var_absolute:,.2f} over {time_horizon} day(s)",
                "risk_level": "high" if var_result.var_percentage > 5 else "medium" if var_result.var_percentage > 2 else "low",
                "expected_shortfall": var_result.cvar_absolute
            },
            "model_diagnostics": var_result.model_diagnostics
        }
        
    except Exception as e:
        logger.error(f"VaR calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/risk/monitoring")
async def get_real_time_risk_monitoring():
    """
    Get real-time risk monitoring dashboard data
    """
    try:
        logger.info("Real-time risk monitoring request")
        
        # Get current market conditions and risk indicators
        risk_indicators = await risk_assessor.get_real_time_risk_indicators()
        
        return {
            "success": True,
            "monitoring": {
                "timestamp": datetime.now().isoformat(),
                "market_regime": risk_indicators.market_regime,
                "overall_market_risk": risk_indicators.overall_market_risk,
                "volatility_regime": risk_indicators.volatility_regime,
                "last_updated": risk_indicators.last_updated.isoformat()
            },
            "risk_indicators": {
                "vix_level": risk_indicators.vix_level,
                "yield_curve_slope": risk_indicators.yield_curve_slope,
                "credit_spreads": risk_indicators.credit_spreads,
                "currency_volatility": risk_indicators.currency_volatility,
                "commodity_volatility": risk_indicators.commodity_volatility
            },
            "alerts": risk_indicators.active_alerts,
            "risk_assessment": {
                "short_term_outlook": risk_indicators.short_term_outlook,
                "medium_term_outlook": risk_indicators.medium_term_outlook,
                "key_risks": risk_indicators.key_risks
            },
            "recommendations": risk_indicators.recommendations
        }
        
    except Exception as e:
        logger.error(f"Risk monitoring error: {e}")
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
    Get current market trends across asset classes using advanced analytics
    """
    try:
        from analytics.market_analytics import AssetClass
        
        # Define asset classes to analyze
        asset_classes = [AssetClass.REAL_ESTATE, AssetClass.COMMODITIES, AssetClass.ART_COLLECTIBLES, AssetClass.BONDS]
        
        # Generate comprehensive market trends analysis
        trends_analysis = await market_analytics._analyze_trends(asset_classes, "1M")
        
        # Format response
        trends_data = {}
        for asset_class in asset_classes:
            class_name = asset_class.value
            if class_name in trends_analysis:
                trend = trends_analysis[class_name]
                trends_data[class_name] = {
                    "direction": trend.direction.value,
                    "strength": trend.strength,
                    "duration_days": trend.duration_days,
                    "support_level": trend.support_level,
                    "resistance_level": trend.resistance_level,
                    "momentum_indicators": trend.momentum_indicators,
                    "reversal_probability": trend.reversal_probability,
                    "confidence": trend.confidence
                }
        
        return {
            "success": True,
            "market_trends": trends_data,
            "timestamp": datetime.now().isoformat(),
            "analysis_period": "1M",
            "data_quality_score": 0.85
        }
        
    except Exception as e:
        logger.error(f"Error getting market trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analytics/market-report")
async def generate_market_report(
    report_request: Dict[str, Any],
    db = Depends(get_db)
):
    """
    Generate comprehensive market analytics report
    """
    try:
        logger.info("Market report generation request")
        
        # Import necessary classes
        from analytics.market_analytics import AssetClass
        
        # Parse request parameters
        asset_classes_str = report_request.get('asset_classes', ['real_estate', 'commodities', 'art_collectibles', 'bonds'])
        time_horizon = report_request.get('time_horizon', '1M')
        include_predictions = report_request.get('include_predictions', True)
        
        # Convert string asset classes to enum
        asset_classes = []
        for ac_str in asset_classes_str:
            try:
                asset_classes.append(AssetClass(ac_str.lower()))
            except ValueError:
                logger.warning(f"Unknown asset class: {ac_str}")
        
        # Generate comprehensive market report
        report = await market_analytics.generate_market_report(
            asset_classes=asset_classes,
            time_horizon=time_horizon,
            include_predictions=include_predictions
        )
        
        return {
            "success": True,
            "report": {
                "report_id": report.report_id,
                "generation_time": report.generation_time.isoformat(),
                "market_overview": report.market_overview,
                "sentiment_analysis": {
                    "overall_sentiment": report.sentiment_analysis.overall_sentiment.value,
                    "sentiment_score": report.sentiment_analysis.sentiment_score,
                    "news_sentiment": report.sentiment_analysis.news_sentiment,
                    "social_sentiment": report.sentiment_analysis.social_sentiment,
                    "analyst_sentiment": report.sentiment_analysis.analyst_sentiment,
                    "sentiment_drivers": report.sentiment_analysis.sentiment_drivers,
                    "confidence": report.sentiment_analysis.confidence
                },
                "trend_analysis": {
                    ac.value: {
                        "direction": trend.direction.value,
                        "strength": trend.strength,
                        "duration_days": trend.duration_days,
                        "confidence": trend.confidence
                    } for ac, trend in report.trend_analysis.items()
                },
                "volatility_analysis": {
                    ac.value: {
                        "current_volatility": vol.current_volatility,
                        "historical_volatility": vol.historical_volatility,
                        "volatility_percentile": vol.volatility_percentile,
                        "volatility_regime": vol.volatility_regime,
                        "volatility_forecast": vol.volatility_forecast
                    } for ac, vol in report.volatility_analysis.items()
                },
                "correlation_analysis": {
                    "correlation_matrix": report.correlation_analysis.correlation_matrix,
                    "correlation_changes": report.correlation_analysis.correlation_changes,
                    "diversification_benefits": report.correlation_analysis.diversification_benefits,
                    "risk_concentrations": report.correlation_analysis.risk_concentrations,
                    "correlation_regime": report.correlation_analysis.correlation_regime
                },
                "market_insights": [
                    {
                        "insight_id": insight.insight_id,
                        "category": insight.category,
                        "priority": insight.priority,
                        "title": insight.title,
                        "description": insight.description,
                        "affected_assets": insight.affected_assets,
                        "recommendation": insight.recommendation,
                        "confidence": insight.confidence,
                        "created_at": insight.created_at.isoformat()
                    } for insight in report.market_insights
                ],
                "risk_alerts": report.risk_alerts,
                "opportunities": report.opportunities,
                "data_quality_score": report.data_quality_score
            }
        }
        
    except Exception as e:
        logger.error(f"Market report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analytics/sentiment/{asset_class}")
async def get_sentiment_analysis(asset_class: str):
    """
    Get detailed sentiment analysis for specific asset class
    """
    try:
        logger.info(f"Sentiment analysis request for: {asset_class}")
        
        # Import necessary classes
        from analytics.market_analytics import AssetClass
        
        # Validate asset class
        try:
            ac = AssetClass(asset_class.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid asset class: {asset_class}")
        
        # Perform sentiment analysis
        sentiment = await market_analytics._analyze_market_sentiment([ac])
        
        return {
            "success": True,
            "sentiment_analysis": {
                "asset_class": asset_class,
                "overall_sentiment": sentiment.overall_sentiment.value,
                "sentiment_score": sentiment.sentiment_score,
                "sentiment_breakdown": {
                    "news_sentiment": sentiment.news_sentiment,
                    "social_sentiment": sentiment.social_sentiment,
                    "analyst_sentiment": sentiment.analyst_sentiment
                },
                "sentiment_drivers": sentiment.sentiment_drivers,
                "confidence": sentiment.confidence,
                "data_quality": sentiment.data_quality
            },
            "interpretation": {
                "sentiment_label": sentiment.overall_sentiment.value,
                "market_mood": "bullish" if sentiment.sentiment_score > 0.1 else "bearish" if sentiment.sentiment_score < -0.1 else "neutral",
                "signal_strength": "strong" if abs(sentiment.sentiment_score) > 0.5 else "moderate" if abs(sentiment.sentiment_score) > 0.2 else "weak"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analytics/volatility/{asset_class}")
async def get_volatility_analysis(
    asset_class: str,
    time_period: str = "30D"
):
    """
    Get detailed volatility analysis for specific asset class
    """
    try:
        logger.info(f"Volatility analysis request for: {asset_class}")
        
        # Import necessary classes
        from analytics.market_analytics import AssetClass
        
        # Validate asset class
        try:
            ac = AssetClass(asset_class.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid asset class: {asset_class}")
        
        # Perform volatility analysis
        volatility = await market_analytics._analyze_volatility([ac])
        
        if asset_class.lower() not in volatility:
            raise HTTPException(status_code=404, detail=f"Volatility data not available for {asset_class}")
        
        vol_data = volatility[asset_class.lower()]
        
        return {
            "success": True,
            "volatility_analysis": {
                "asset_class": asset_class,
                "current_volatility": vol_data.current_volatility,
                "historical_volatility": vol_data.historical_volatility,
                "implied_volatility": vol_data.implied_volatility,
                "volatility_percentile": vol_data.volatility_percentile,
                "volatility_regime": vol_data.volatility_regime,
                "volatility_forecast": vol_data.volatility_forecast,
                "time_period": time_period
            },
            "volatility_interpretation": {
                "regime_description": {
                    "low": "Below historical average - stable market conditions",
                    "normal": "Within normal historical range",
                    "high": "Above historical average - increased uncertainty",
                    "extreme": "Exceptionally high - potential market stress"
                }.get(vol_data.volatility_regime, "Unknown regime"),
                "percentile_meaning": f"Current volatility is higher than {vol_data.volatility_percentile:.1%} of historical observations",
                "forecast_trend": "increasing" if vol_data.volatility_forecast.get('30d', 0) > vol_data.current_volatility else "decreasing"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Volatility analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analytics/correlations")
async def get_correlation_analysis(
    asset_classes: Optional[str] = None,
    time_period: str = "90D"
):
    """
    Get cross-asset correlation analysis
    """
    try:
        logger.info("Correlation analysis request")
        
        # Import necessary classes
        from analytics.market_analytics import AssetClass
        
        # Parse asset classes
        if asset_classes:
            ac_list = []
            for ac_str in asset_classes.split(','):
                try:
                    ac_list.append(AssetClass(ac_str.strip().lower()))
                except ValueError:
                    logger.warning(f"Unknown asset class: {ac_str}")
        else:
            ac_list = [AssetClass.REAL_ESTATE, AssetClass.COMMODITIES, AssetClass.ART_COLLECTIBLES, AssetClass.BONDS]
        
        # Perform correlation analysis
        correlation = await market_analytics._analyze_correlations(ac_list)
        
        return {
            "success": True,
            "correlation_analysis": {
                "correlation_matrix": correlation.correlation_matrix,
                "correlation_changes": correlation.correlation_changes,
                "diversification_benefits": correlation.diversification_benefits,
                "risk_concentrations": correlation.risk_concentrations,
                "correlation_regime": correlation.correlation_regime,
                "time_period": time_period
            },
            "insights": {
                "highest_correlation": max(
                    [(f"{k1}-{k2}", corr) for k1, corrs in correlation.correlation_matrix.items() 
                     for k2, corr in corrs.items() if k1 != k2],
                    key=lambda x: abs(x[1])
                )[0] if correlation.correlation_matrix else None,
                "diversification_score": sum(correlation.diversification_benefits.values()) / len(correlation.diversification_benefits) if correlation.diversification_benefits else 0,
                "concentration_risk": len(correlation.risk_concentrations) > 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Correlation analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analytics/predictions")
async def get_market_predictions(
    prediction_request: Dict[str, Any],
    db = Depends(get_db)
):
    """
    Get AI-powered market predictions for asset classes
    """
    try:
        logger.info("Market predictions request")
        
        # Import necessary classes
        from analytics.market_analytics import AssetClass
        
        # Parse request
        asset_classes_str = prediction_request.get('asset_classes', ['real_estate', 'commodities'])
        time_horizons = prediction_request.get('time_horizons', ['1D', '7D', '30D'])
        
        # Convert to asset class enums
        asset_classes = []
        for ac_str in asset_classes_str:
            try:
                asset_classes.append(AssetClass(ac_str.lower()))
            except ValueError:
                logger.warning(f"Unknown asset class: {ac_str}")
        
        # Generate predictions
        predictions = await market_analytics._generate_predictions(asset_classes)
        
        # Format predictions response
        predictions_data = {}
        for ac_str, prediction in predictions.items():
            predictions_data[ac_str] = {
                "model_type": prediction.model_type,
                "predictions": prediction.predictions,
                "confidence_intervals": {
                    horizon: {
                        "lower": ci[0],
                        "upper": ci[1]
                    } for horizon, ci in prediction.confidence_intervals.items()
                },
                "feature_importance": prediction.feature_importance,
                "model_accuracy": prediction.model_accuracy,
                "risk_scenarios": prediction.risk_scenarios
            }
        
        return {
            "success": True,
            "predictions": predictions_data,
            "metadata": {
                "prediction_date": datetime.now().isoformat(),
                "time_horizons": time_horizons,
                "model_version": "v1.0.0",
                "data_quality_score": 0.88
            },
            "disclaimers": [
                "Predictions are based on historical data and machine learning models",
                "Past performance does not guarantee future results",
                "Consider multiple factors when making investment decisions",
                "High volatility may affect prediction accuracy"
            ]
        }
        
    except Exception as e:
        logger.error(f"Market predictions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# KYC/AML Compliance Endpoints
@app.post("/api/v1/compliance/kyc/verify-customer")
async def verify_customer(
    customer_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db = Depends(get_db)
):
    """
    Perform comprehensive KYC verification for new customers
    """
    try:
        logger.info(f"KYC verification request for customer: {customer_data.get('customer_id', 'Unknown')}")
        
        # Validate input
        required_fields = ['customer_id', 'personal_info', 'documents']
        missing_fields = [field for field in required_fields if field not in customer_data]
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {missing_fields}"
            )
        
        # Import necessary classes
        from compliance.kyc_aml_service import CustomerData, DocumentData, PersonalInfo
        
        # Prepare customer data
        personal_info = PersonalInfo(
            first_name=customer_data['personal_info']['first_name'],
            last_name=customer_data['personal_info']['last_name'],
            date_of_birth=customer_data['personal_info']['date_of_birth'],
            nationality=customer_data['personal_info']['nationality'],
            address=customer_data['personal_info']['address'],
            phone_number=customer_data['personal_info'].get('phone_number'),
            email=customer_data['personal_info'].get('email')
        )
        
        documents = []
        for doc_data in customer_data['documents']:
            document = DocumentData(
                document_type=doc_data['type'],
                document_number=doc_data['number'],
                issuing_country=doc_data['issuing_country'],
                expiry_date=doc_data.get('expiry_date'),
                document_image_url=doc_data.get('image_url')
            )
            documents.append(document)
        
        customer = CustomerData(
            customer_id=customer_data['customer_id'],
            personal_info=personal_info,
            documents=documents,
            risk_profile=customer_data.get('risk_profile', 'medium'),
            customer_type=customer_data.get('customer_type', 'individual')
        )
        
        # Perform KYC verification
        verification_result = await kyc_aml_service.verify_customer(customer)
        
        # Store result in database
        background_tasks.add_task(
            store_compliance_result,
            "kyc_verification",
            customer_data,
            verification_result.__dict__,
            db
        )
        
        return {
            "success": True,
            "verification": {
                "customer_id": verification_result.customer_id,
                "verification_status": verification_result.verification_status.value,
                "risk_score": verification_result.risk_score,
                "risk_level": verification_result.risk_level.value,
                "verification_date": verification_result.verification_date.isoformat(),
                "expiry_date": verification_result.expiry_date.isoformat() if verification_result.expiry_date else None
            },
            "checks_performed": {
                "identity_verification": {
                    "status": verification_result.identity_verification.status.value,
                    "confidence_score": verification_result.identity_verification.confidence_score,
                    "checks_passed": verification_result.identity_verification.checks_passed,
                    "checks_failed": verification_result.identity_verification.checks_failed
                },
                "document_verification": {
                    "status": verification_result.document_verification.status.value,
                    "documents_verified": verification_result.document_verification.documents_verified,
                    "authenticity_score": verification_result.document_verification.authenticity_score
                },
                "address_verification": {
                    "status": verification_result.address_verification.status.value,
                    "verification_method": verification_result.address_verification.verification_method,
                    "confidence_score": verification_result.address_verification.confidence_score
                }
            },
            "sanctions_screening": verification_result.sanctions_screening,
            "recommendations": verification_result.recommendations,
            "next_review_date": verification_result.next_review_date.isoformat() if verification_result.next_review_date else None
        }
        
    except Exception as e:
        logger.error(f"KYC verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/compliance/aml/screen-sanctions")
async def screen_sanctions(
    screening_request: Dict[str, Any],
    db = Depends(get_db)
):
    """
    Screen customers and transactions against sanctions lists
    """
    try:
        logger.info("Sanctions screening request")
        
        # Validate input
        required_fields = ['entity_type', 'entity_data']
        missing_fields = [field for field in required_fields if field not in screening_request]
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {missing_fields}"
            )
        
        entity_type = screening_request['entity_type']  # 'person' or 'organization'
        entity_data = screening_request['entity_data']
        
        # Perform sanctions screening
        screening_result = await kyc_aml_service.screen_sanctions(entity_type, entity_data)
        
        return {
            "success": True,
            "sanctions_screening": {
                "entity_type": entity_type,
                "screening_date": screening_result.screening_date.isoformat(),
                "overall_risk_score": screening_result.overall_risk_score,
                "match_found": screening_result.match_found,
                "watchlist_matches": [
                    {
                        "list_name": match.list_name,
                        "match_type": match.match_type.value,
                        "confidence_score": match.confidence_score,
                        "matched_entity": match.matched_entity,
                        "risk_level": match.risk_level.value
                    } for match in screening_result.watchlist_matches
                ],
                "pep_screening": {
                    "is_pep": screening_result.pep_screening.is_pep,
                    "pep_category": screening_result.pep_screening.pep_category,
                    "risk_score": screening_result.pep_screening.risk_score,
                    "source_lists": screening_result.pep_screening.source_lists
                },
                "adverse_media": {
                    "negative_news_found": screening_result.adverse_media.negative_news_found,
                    "risk_score": screening_result.adverse_media.risk_score,
                    "articles_count": screening_result.adverse_media.articles_count,
                    "severity_level": screening_result.adverse_media.severity_level
                }
            },
            "recommendations": screening_result.recommendations,
            "next_screening_date": screening_result.next_screening_date.isoformat() if screening_result.next_screening_date else None
        }
        
    except Exception as e:
        logger.error(f"Sanctions screening error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/compliance/aml/monitor-transaction")
async def monitor_transaction(
    transaction_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db = Depends(get_db)
):
    """
    Monitor transactions for suspicious activity and AML compliance
    """
    try:
        logger.info(f"Transaction monitoring request: {transaction_data.get('transaction_id', 'Unknown')}")
        
        # Validate input
        required_fields = ['transaction_id', 'customer_id', 'amount', 'currency', 'transaction_type']
        missing_fields = [field for field in required_fields if field not in transaction_data]
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {missing_fields}"
            )
        
        # Import necessary classes
        from compliance.kyc_aml_service import TransactionData
        
        # Prepare transaction data
        transaction = TransactionData(
            transaction_id=transaction_data['transaction_id'],
            customer_id=transaction_data['customer_id'],
            amount=float(transaction_data['amount']),
            currency=transaction_data['currency'],
            transaction_type=transaction_data['transaction_type'],
            counterparty_id=transaction_data.get('counterparty_id'),
            transaction_date=datetime.fromisoformat(transaction_data.get('transaction_date', datetime.now().isoformat())),
            description=transaction_data.get('description'),
            asset_type=transaction_data.get('asset_type'),
            source_of_funds=transaction_data.get('source_of_funds')
        )
        
        # Perform transaction monitoring
        monitoring_result = await kyc_aml_service.monitor_transaction(transaction)
        
        # Store result for compliance reporting
        background_tasks.add_task(
            store_compliance_result,
            "transaction_monitoring",
            transaction_data,
            monitoring_result.__dict__,
            db
        )
        
        return {
            "success": True,
            "transaction_monitoring": {
                "transaction_id": monitoring_result.transaction_id,
                "monitoring_date": monitoring_result.monitoring_date.isoformat(),
                "risk_score": monitoring_result.risk_score,
                "risk_level": monitoring_result.risk_level.value,
                "suspicious_activity_detected": monitoring_result.suspicious_activity_detected,
                "alerts_triggered": [
                    {
                        "alert_type": alert.alert_type.value,
                        "severity": alert.severity.value,
                        "description": alert.description,
                        "triggered_by": alert.triggered_by,
                        "risk_score": alert.risk_score
                    } for alert in monitoring_result.alerts_triggered
                ],
                "compliance_checks": {
                    "structuring_check": {
                        "flagged": monitoring_result.compliance_checks.structuring_check.flagged,
                        "pattern_detected": monitoring_result.compliance_checks.structuring_check.pattern_detected,
                        "confidence_score": monitoring_result.compliance_checks.structuring_check.confidence_score
                    },
                    "velocity_check": {
                        "flagged": monitoring_result.compliance_checks.velocity_check.flagged,
                        "transaction_frequency": monitoring_result.compliance_checks.velocity_check.transaction_frequency,
                        "volume_increase": monitoring_result.compliance_checks.velocity_check.volume_increase
                    },
                    "geographic_check": {
                        "flagged": monitoring_result.compliance_checks.geographic_check.flagged,
                        "high_risk_jurisdiction": monitoring_result.compliance_checks.geographic_check.high_risk_jurisdiction,
                        "risk_score": monitoring_result.compliance_checks.geographic_check.risk_score
                    }
                }
            },
            "recommendations": monitoring_result.recommendations,
            "requires_investigation": monitoring_result.requires_investigation,
            "sar_filing_recommended": monitoring_result.sar_filing_recommended
        }
        
    except Exception as e:
        logger.error(f"Transaction monitoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/compliance/aml/suspicious-activities")
async def get_suspicious_activities(
    days_back: int = 30,
    risk_level: Optional[str] = None,
    investigation_status: Optional[str] = None
):
    """
    Get list of suspicious activities detected by AML monitoring
    """
    try:
        logger.info("Suspicious activities request")
        
        # Get suspicious activities from AML service
        activities = await kyc_aml_service.get_suspicious_activities(
            days_back=days_back,
            risk_level=risk_level,
            investigation_status=investigation_status
        )
        
        return {
            "success": True,
            "suspicious_activities": [
                {
                    "activity_id": activity.activity_id,
                    "customer_id": activity.customer_id,
                    "transaction_id": activity.transaction_id,
                    "detected_date": activity.detected_date.isoformat(),
                    "activity_type": activity.activity_type.value,
                    "risk_score": activity.risk_score,
                    "risk_level": activity.risk_level.value,
                    "description": activity.description,
                    "investigation_status": activity.investigation_status.value,
                    "assigned_investigator": activity.assigned_investigator,
                    "alerts_count": len(activity.related_alerts),
                    "last_updated": activity.last_updated.isoformat()
                } for activity in activities
            ],
            "summary": {
                "total_activities": len(activities),
                "high_risk_count": len([a for a in activities if a.risk_level.value == "high"]),
                "under_investigation": len([a for a in activities if a.investigation_status.value == "under_investigation"]),
                "sar_filed": len([a for a in activities if a.investigation_status.value == "sar_filed"])
            },
            "query_parameters": {
                "days_back": days_back,
                "risk_level": risk_level,
                "investigation_status": investigation_status
            }
        }
        
    except Exception as e:
        logger.error(f"Suspicious activities error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/compliance/reports/generate")
async def generate_compliance_report(
    report_request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db = Depends(get_db)
):
    """
    Generate regulatory compliance reports (SAR, CTR, etc.)
    """
    try:
        logger.info("Compliance report generation request")
        
        # Validate input
        required_fields = ['report_type', 'period_start', 'period_end']
        missing_fields = [field for field in required_fields if field not in report_request]
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {missing_fields}"
            )
        
        report_type = report_request['report_type']  # 'SAR', 'CTR', 'SUSPICIOUS_ACTIVITY'
        period_start = datetime.fromisoformat(report_request['period_start'])
        period_end = datetime.fromisoformat(report_request['period_end'])
        
        # Generate compliance report
        report = await kyc_aml_service.generate_compliance_report(
            report_type=report_type,
            period_start=period_start,
            period_end=period_end,
            filters=report_request.get('filters', {})
        )
        
        # Store report for audit trail
        background_tasks.add_task(
            store_compliance_result,
            "compliance_report",
            report_request,
            report.__dict__,
            db
        )
        
        return {
            "success": True,
            "compliance_report": {
                "report_id": report.report_id,
                "report_type": report.report_type,
                "generation_date": report.generation_date.isoformat(),
                "period_start": report.period_start.isoformat(),
                "period_end": report.period_end.isoformat(),
                "total_records": report.total_records,
                "summary_statistics": report.summary_statistics,
                "regulatory_thresholds": report.regulatory_thresholds,
                "compliance_status": report.compliance_status,
                "report_url": report.report_url,
                "file_format": report.file_format
            },
            "filing_requirements": {
                "filing_deadline": report.filing_deadline.isoformat() if report.filing_deadline else None,
                "regulatory_authority": report.regulatory_authority,
                "submission_method": report.submission_method,
                "required_approvals": report.required_approvals
            },
            "data_quality": {
                "completeness_score": report.data_quality_score,
                "validation_errors": report.validation_errors,
                "missing_data_fields": report.missing_data_fields
            }
        }
        
    except Exception as e:
        logger.error(f"Compliance report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/compliance/status/{customer_id}")
async def get_customer_compliance_status(customer_id: str):
    """
    Get comprehensive compliance status for a specific customer
    """
    try:
        logger.info(f"Compliance status request for customer: {customer_id}")
        
        # Get customer compliance status
        status = await kyc_aml_service.get_customer_compliance_status(customer_id)
        
        return {
            "success": True,
            "compliance_status": {
                "customer_id": status.customer_id,
                "overall_status": status.overall_status.value,
                "risk_level": status.risk_level.value,
                "last_updated": status.last_updated.isoformat(),
                "kyc_status": {
                    "status": status.kyc_status.status.value,
                    "completion_date": status.kyc_status.completion_date.isoformat() if status.kyc_status.completion_date else None,
                    "expiry_date": status.kyc_status.expiry_date.isoformat() if status.kyc_status.expiry_date else None,
                    "next_review_date": status.kyc_status.next_review_date.isoformat() if status.kyc_status.next_review_date else None,
                    "documents_status": status.kyc_status.documents_status
                },
                "aml_monitoring": {
                    "monitoring_status": status.aml_monitoring.monitoring_status.value,
                    "last_screening_date": status.aml_monitoring.last_screening_date.isoformat() if status.aml_monitoring.last_screening_date else None,
                    "next_screening_date": status.aml_monitoring.next_screening_date.isoformat() if status.aml_monitoring.next_screening_date else None,
                    "sanctions_status": status.aml_monitoring.sanctions_status.value,
                    "pep_status": status.aml_monitoring.pep_status.value,
                    "watchlist_alerts": status.aml_monitoring.watchlist_alerts
                },
                "transaction_monitoring": {
                    "monitoring_enabled": status.transaction_monitoring.monitoring_enabled,
                    "risk_score": status.transaction_monitoring.risk_score,
                    "alerts_last_30_days": status.transaction_monitoring.alerts_last_30_days,
                    "suspicious_activities_count": status.transaction_monitoring.suspicious_activities_count,
                    "last_transaction_date": status.transaction_monitoring.last_transaction_date.isoformat() if status.transaction_monitoring.last_transaction_date else None
                }
            },
            "action_items": status.action_items,
            "compliance_score": status.compliance_score,
            "recommendations": status.recommendations
        }
        
    except Exception as e:
        logger.error(f"Compliance status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def store_compliance_result(
    compliance_type: str,
    input_data: Dict[str, Any],
    result: Dict[str, Any],
    db
):
    """
    Store compliance result in database for audit trail
    """
    try:
        # Store in database (simplified)
        compliance_record = {
            "compliance_type": compliance_type,
            "input_data": input_data,
            "compliance_result": result,
            "timestamp": datetime.now()
        }
        
        # In production, save to actual database
        logger.info(f"Stored {compliance_type} compliance result")
        
    except Exception as e:
        logger.error(f"Failed to store compliance result: {e}")


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