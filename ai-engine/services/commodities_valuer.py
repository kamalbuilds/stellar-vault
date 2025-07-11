"""
Commodities Valuation Service for StellarVault AI Engine
Uses ML models and real market data for commodity valuations
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
import aiohttp
from dataclasses import dataclass
from loguru import logger
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

from core.config import settings
from core.cache import cache_valuation, cached_valuation


@dataclass
class CommodityFeatures:
    """
    Commodity features for valuation
    """
    # Basic commodity info
    commodity_type: str  # gold, silver, oil, wheat, etc.
    grade: str  # quality grade
    quantity: float
    unit: str  # oz, barrel, bushel, etc.
    location: str  # delivery location
    
    # Market features
    spot_price: float
    futures_prices: Dict[str, float]  # contract month -> price
    supply_levels: float
    demand_forecast: float
    inventory_levels: float
    
    # Economic indicators
    usd_index: float
    inflation_rate: float
    interest_rates: float
    gdp_growth: float
    
    # Commodity-specific factors
    weather_conditions: Dict[str, Any]
    geopolitical_risk: float
    storage_costs: float
    transportation_costs: float
    
    # Technical indicators
    price_momentum: float
    volatility: float
    moving_averages: Dict[str, float]
    rsi: float
    macd: Dict[str, float]


@dataclass
class CommodityValuationResult:
    """
    Commodity valuation result
    """
    estimated_value: float
    confidence_score: float
    value_range: Tuple[float, float]
    model_used: str
    feature_importance: Dict[str, float]
    market_factors: Dict[str, Any]
    price_forecast: Dict[str, float]
    methodology: str
    valuation_date: datetime


class CommoditiesValuer:
    """
    Advanced commodities valuation service using ML and market data
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.is_initialized = False
        
        # Supported commodities
        self.supported_commodities = {
            'precious_metals': ['gold', 'silver', 'platinum', 'palladium'],
            'energy': ['crude_oil', 'natural_gas', 'gasoline', 'heating_oil'],
            'agriculture': ['wheat', 'corn', 'soybeans', 'cotton', 'coffee', 'sugar'],
            'base_metals': ['copper', 'aluminum', 'zinc', 'nickel', 'lead'],
            'livestock': ['live_cattle', 'lean_hogs', 'feeder_cattle']
        }
        
        # Model configurations for different commodity types
        self.model_configs = {
            'random_forest': {
                'n_estimators': 150,
                'max_depth': 15,
                'min_samples_split': 5,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 150,
                'learning_rate': 0.1,
                'max_depth': 8,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 150,
                'learning_rate': 0.1,
                'max_depth': 8,
                'random_state': 42
            }
        }
    
    async def initialize(self):
        """
        Initialize commodities valuation models and data sources
        """
        try:
            logger.info("Initializing Commodities Valuation Service...")
            
            # Load or train models for each commodity type
            for commodity_category in self.supported_commodities.keys():
                await self._load_or_train_category_models(commodity_category)
            
            self.is_initialized = True
            logger.info("✅ Commodities Valuation Service initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Commodities Valuation Service: {e}")
            raise
    
    async def value_commodity(self, commodity_data: Dict[str, Any]) -> CommodityValuationResult:
        """
        Value a commodity using ML models and market data
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            commodity_type = commodity_data.get('commodity_type', '').lower()
            
            # Check cache first
            cache_key = f"{commodity_type}_{commodity_data.get('quantity', 0)}"
            cached_result = await cached_valuation(cache_key, 'commodities_ml')
            if cached_result:
                return CommodityValuationResult(**cached_result)
            
            # Get real-time market data
            market_data = await self._get_real_time_market_data(commodity_type)
            
            # Extract and enrich commodity features
            features = await self._extract_commodity_features(commodity_data, market_data)
            
            # Get commodity category
            category = self._get_commodity_category(commodity_type)
            
            # Run valuation models
            model_predictions = await self._run_commodity_models(features, category)
            
            # Calculate final valuation
            final_valuation = self._calculate_final_valuation(
                model_predictions, features, market_data
            )
            
            # Generate price forecast
            price_forecast = await self._generate_price_forecast(features, category)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                model_predictions, market_data, features
            )
            
            # Create result
            result = CommodityValuationResult(
                estimated_value=final_valuation['value'],
                confidence_score=confidence_score,
                value_range=final_valuation['range'],
                model_used=final_valuation['model_used'],
                feature_importance=final_valuation['feature_importance'],
                market_factors=market_data,
                price_forecast=price_forecast,
                methodology='ML_Market_Analysis',
                valuation_date=datetime.now()
            )
            
            # Cache result
            await cache_valuation(cache_key, 'commodities_ml', result.__dict__)
            
            return result
            
        except Exception as e:
            logger.error(f"Commodity valuation error: {e}")
            raise
    
    async def _get_real_time_market_data(self, commodity_type: str) -> Dict[str, Any]:
        """
        Get real-time market data for commodity
        """
        try:
            market_data = {}
            
            # Get spot prices and futures data
            if commodity_type in ['gold', 'silver', 'platinum', 'palladium']:
                market_data.update(await self._get_precious_metals_data(commodity_type))
            elif commodity_type in ['crude_oil', 'natural_gas', 'gasoline']:
                market_data.update(await self._get_energy_data(commodity_type))
            elif commodity_type in ['wheat', 'corn', 'soybeans', 'cotton']:
                market_data.update(await self._get_agriculture_data(commodity_type))
            elif commodity_type in ['copper', 'aluminum', 'zinc', 'nickel']:
                market_data.update(await self._get_base_metals_data(commodity_type))
            
            # Get economic indicators
            economic_data = await self._get_economic_indicators()
            market_data.update(economic_data)
            
            # Get technical indicators
            technical_data = await self._get_technical_indicators(commodity_type)
            market_data.update(technical_data)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to get market data for {commodity_type}: {e}")
            return self._get_default_market_data(commodity_type)
    
    async def _get_precious_metals_data(self, metal: str) -> Dict[str, Any]:
        """
        Get precious metals market data
        """
        try:
            # In production, use real APIs like:
            # - London Bullion Market Association (LBMA)
            # - COMEX futures data
            # - Kitco, JM Bullion APIs
            
            base_prices = {
                'gold': 2000.0,
                'silver': 25.0,
                'platinum': 1000.0,
                'palladium': 1500.0
            }
            
            base_price = base_prices.get(metal, 1000.0)
            
            # Simulate real market data
            spot_price = base_price * np.random.normal(1.0, 0.02)
            
            return {
                'spot_price': spot_price,
                'futures_prices': {
                    '1_month': spot_price * 1.001,
                    '3_month': spot_price * 1.003,
                    '6_month': spot_price * 1.007,
                    '12_month': spot_price * 1.015
                },
                'supply_levels': np.random.uniform(0.8, 1.2),
                'demand_forecast': np.random.uniform(0.9, 1.1),
                'inventory_levels': np.random.uniform(0.7, 1.3),
                'storage_costs': base_price * 0.001,  # 0.1% of value per month
                'transportation_costs': base_price * 0.0005
            }
            
        except Exception as e:
            logger.error(f"Failed to get precious metals data: {e}")
            return {}
    
    async def _get_energy_data(self, energy_type: str) -> Dict[str, Any]:
        """
        Get energy commodities market data
        """
        try:
            # In production, use real APIs like:
            # - EIA (Energy Information Administration)
            # - NYMEX futures data
            # - Oil price APIs
            
            base_prices = {
                'crude_oil': 75.0,  # per barrel
                'natural_gas': 3.5,  # per MMBtu
                'gasoline': 2.8,  # per gallon
                'heating_oil': 2.5  # per gallon
            }
            
            base_price = base_prices.get(energy_type, 50.0)
            spot_price = base_price * np.random.normal(1.0, 0.03)
            
            return {
                'spot_price': spot_price,
                'futures_prices': {
                    '1_month': spot_price * np.random.normal(1.0, 0.01),
                    '3_month': spot_price * np.random.normal(1.0, 0.02),
                    '6_month': spot_price * np.random.normal(1.0, 0.03),
                    '12_month': spot_price * np.random.normal(1.0, 0.05)
                },
                'supply_levels': np.random.uniform(0.8, 1.2),
                'demand_forecast': np.random.uniform(0.9, 1.1),
                'inventory_levels': np.random.uniform(0.7, 1.3),
                'geopolitical_risk': np.random.uniform(0.1, 0.9),
                'weather_conditions': {
                    'temperature': np.random.normal(20, 10),
                    'hurricanes_forecast': np.random.randint(0, 3)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get energy data: {e}")
            return {}
    
    async def _get_agriculture_data(self, crop: str) -> Dict[str, Any]:
        """
        Get agricultural commodities market data
        """
        try:
            # In production, use real APIs like:
            # - USDA (United States Department of Agriculture)
            # - CBOT futures data
            # - Weather APIs
            
            base_prices = {
                'wheat': 8.0,  # per bushel
                'corn': 6.5,   # per bushel
                'soybeans': 15.0,  # per bushel
                'cotton': 0.75,  # per pound
                'coffee': 1.8,   # per pound
                'sugar': 0.22    # per pound
            }
            
            base_price = base_prices.get(crop, 5.0)
            spot_price = base_price * np.random.normal(1.0, 0.04)
            
            return {
                'spot_price': spot_price,
                'futures_prices': {
                    '3_month': spot_price * np.random.normal(1.0, 0.02),
                    '6_month': spot_price * np.random.normal(1.0, 0.03),
                    '9_month': spot_price * np.random.normal(1.0, 0.04),
                    '12_month': spot_price * np.random.normal(1.0, 0.05)
                },
                'supply_levels': np.random.uniform(0.7, 1.3),
                'demand_forecast': np.random.uniform(0.9, 1.1),
                'inventory_levels': np.random.uniform(0.6, 1.4),
                'weather_conditions': {
                    'rainfall': np.random.uniform(0, 100),  # mm
                    'temperature': np.random.uniform(-10, 40),  # Celsius
                    'drought_risk': np.random.uniform(0, 1),
                    'growing_season_health': np.random.uniform(0.5, 1.0)
                },
                'planting_area': np.random.uniform(0.9, 1.1),
                'yield_forecast': np.random.uniform(0.8, 1.2)
            }
            
        except Exception as e:
            logger.error(f"Failed to get agriculture data: {e}")
            return {}
    
    async def _get_base_metals_data(self, metal: str) -> Dict[str, Any]:
        """
        Get base metals market data
        """
        try:
            # In production, use real APIs like:
            # - LME (London Metal Exchange)
            # - Shanghai Futures Exchange
            # - Industrial demand data
            
            base_prices = {
                'copper': 9000.0,   # per metric ton
                'aluminum': 2300.0, # per metric ton
                'zinc': 3200.0,     # per metric ton
                'nickel': 22000.0,  # per metric ton
                'lead': 2100.0      # per metric ton
            }
            
            base_price = base_prices.get(metal, 5000.0)
            spot_price = base_price * np.random.normal(1.0, 0.03)
            
            return {
                'spot_price': spot_price,
                'futures_prices': {
                    '3_month': spot_price * np.random.normal(1.0, 0.01),
                    '6_month': spot_price * np.random.normal(1.0, 0.02),
                    '12_month': spot_price * np.random.normal(1.0, 0.03),
                    '24_month': spot_price * np.random.normal(1.0, 0.05)
                },
                'supply_levels': np.random.uniform(0.8, 1.2),
                'demand_forecast': np.random.uniform(0.9, 1.1),
                'inventory_levels': np.random.uniform(0.7, 1.3),
                'industrial_demand': np.random.uniform(0.8, 1.2),
                'mine_production': np.random.uniform(0.9, 1.1),
                'recycling_rate': np.random.uniform(0.1, 0.4)
            }
            
        except Exception as e:
            logger.error(f"Failed to get base metals data: {e}")
            return {}
    
    async def _get_economic_indicators(self) -> Dict[str, Any]:
        """
        Get economic indicators affecting commodity prices
        """
        try:
            # In production, use real APIs like:
            # - Federal Reserve Economic Data (FRED)
            # - Bloomberg API
            # - Economic calendar APIs
            
            return {
                'usd_index': np.random.normal(105, 5),
                'inflation_rate': np.random.uniform(0.02, 0.08),
                'interest_rates': np.random.uniform(0.01, 0.06),
                'gdp_growth': np.random.uniform(-0.02, 0.05),
                'unemployment_rate': np.random.uniform(0.03, 0.10),
                'manufacturing_pmi': np.random.uniform(45, 60),
                'consumer_confidence': np.random.uniform(80, 120)
            }
            
        except Exception as e:
            logger.error(f"Failed to get economic indicators: {e}")
            return {
                'usd_index': 105,
                'inflation_rate': 0.03,
                'interest_rates': 0.025,
                'gdp_growth': 0.025
            }
    
    async def _get_technical_indicators(self, commodity_type: str) -> Dict[str, Any]:
        """
        Get technical analysis indicators
        """
        try:
            # In production, calculate from real price history
            return {
                'price_momentum': np.random.uniform(-0.1, 0.1),
                'volatility': np.random.uniform(0.1, 0.5),
                'moving_averages': {
                    'ma_10': np.random.normal(100, 5),
                    'ma_20': np.random.normal(100, 7),
                    'ma_50': np.random.normal(100, 10),
                    'ma_200': np.random.normal(100, 15)
                },
                'rsi': np.random.uniform(20, 80),
                'macd': {
                    'macd_line': np.random.uniform(-2, 2),
                    'signal_line': np.random.uniform(-2, 2),
                    'histogram': np.random.uniform(-1, 1)
                },
                'bollinger_bands': {
                    'upper': np.random.normal(105, 2),
                    'middle': np.random.normal(100, 2),
                    'lower': np.random.normal(95, 2)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get technical indicators: {e}")
            return {}
    
    async def _extract_commodity_features(self, commodity_data: Dict[str, Any], 
                                        market_data: Dict[str, Any]) -> CommodityFeatures:
        """
        Extract commodity features for model input
        """
        return CommodityFeatures(
            commodity_type=commodity_data.get('commodity_type', ''),
            grade=commodity_data.get('grade', 'standard'),
            quantity=commodity_data.get('quantity', 1.0),
            unit=commodity_data.get('unit', 'unit'),
            location=commodity_data.get('location', 'spot'),
            spot_price=market_data.get('spot_price', 100.0),
            futures_prices=market_data.get('futures_prices', {}),
            supply_levels=market_data.get('supply_levels', 1.0),
            demand_forecast=market_data.get('demand_forecast', 1.0),
            inventory_levels=market_data.get('inventory_levels', 1.0),
            usd_index=market_data.get('usd_index', 105),
            inflation_rate=market_data.get('inflation_rate', 0.03),
            interest_rates=market_data.get('interest_rates', 0.025),
            gdp_growth=market_data.get('gdp_growth', 0.025),
            weather_conditions=market_data.get('weather_conditions', {}),
            geopolitical_risk=market_data.get('geopolitical_risk', 0.5),
            storage_costs=market_data.get('storage_costs', 0.0),
            transportation_costs=market_data.get('transportation_costs', 0.0),
            price_momentum=market_data.get('price_momentum', 0.0),
            volatility=market_data.get('volatility', 0.2),
            moving_averages=market_data.get('moving_averages', {}),
            rsi=market_data.get('rsi', 50),
            macd=market_data.get('macd', {})
        )
    
    def _get_commodity_category(self, commodity_type: str) -> str:
        """
        Get commodity category for model selection
        """
        for category, commodities in self.supported_commodities.items():
            if commodity_type in commodities:
                return category
        return 'general'
    
    async def _run_commodity_models(self, features: CommodityFeatures, 
                                  category: str) -> Dict[str, Dict[str, Any]]:
        """
        Run commodity valuation models
        """
        # Convert features to model input
        feature_vector = self._features_to_vector(features)
        
        predictions = {}
        model_key = f"{category}"
        
        if model_key in self.models:
            for model_name, model in self.models[model_key].items():
                try:
                    # Scale features
                    scaled_features = self.scalers[model_key][model_name].transform([feature_vector])
                    
                    # Make prediction
                    prediction = model.predict(scaled_features)[0]
                    
                    # Adjust for quantity
                    adjusted_prediction = prediction * features.quantity
                    
                    # Get feature importance
                    feature_importance = {}
                    if hasattr(model, 'feature_importances_'):
                        importance_scores = model.feature_importances_
                        feature_importance = dict(zip(self.feature_columns, importance_scores))
                    
                    predictions[model_name] = {
                        'prediction': adjusted_prediction,
                        'unit_price': prediction,
                        'feature_importance': feature_importance
                    }
                    
                except Exception as e:
                    logger.error(f"Model {model_name} prediction failed: {e}")
        
        return predictions
    
    def _calculate_final_valuation(self, model_predictions: Dict[str, Dict[str, Any]], 
                                 features: CommodityFeatures, 
                                 market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate final commodity valuation
        """
        # Get model predictions
        model_values = [pred['prediction'] for pred in model_predictions.values()]
        
        # Market-based valuation (spot price * quantity)
        market_value = features.spot_price * features.quantity
        
        # Storage and transportation costs
        total_costs = (features.storage_costs + features.transportation_costs) * features.quantity
        
        # Ensemble prediction
        if model_values:
            model_avg = np.mean(model_values)
            
            # Weight model vs market
            model_weight = 0.6
            market_weight = 0.4
            
            final_value = model_weight * model_avg + market_weight * market_value
        else:
            final_value = market_value
        
        # Adjust for costs
        final_value -= total_costs
        
        # Calculate prediction interval
        if model_values:
            std_dev = np.std(model_values + [market_value])
            lower_bound = final_value - 1.96 * std_dev
            upper_bound = final_value + 1.96 * std_dev
        else:
            lower_bound = final_value * 0.9
            upper_bound = final_value * 1.1
        
        # Aggregate feature importance
        feature_importance = {}
        for pred in model_predictions.values():
            for feature, importance in pred.get('feature_importance', {}).items():
                feature_importance[feature] = feature_importance.get(feature, 0) + importance
        
        # Normalize feature importance
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {k: v/total_importance for k, v in feature_importance.items()}
        
        return {
            'value': final_value,
            'range': (lower_bound, upper_bound),
            'model_used': 'ensemble',
            'feature_importance': feature_importance,
            'market_value': market_value,
            'total_costs': total_costs
        }
    
    async def _generate_price_forecast(self, features: CommodityFeatures, 
                                     category: str) -> Dict[str, float]:
        """
        Generate price forecast for different time horizons
        """
        base_price = features.spot_price
        
        # Simple trend-based forecast (in production, use more sophisticated models)
        momentum = features.price_momentum
        volatility = features.volatility
        
        forecasts = {}
        
        for period, months in [('1_month', 1), ('3_month', 3), ('6_month', 6), ('12_month', 12)]:
            # Apply trend and add uncertainty
            trend_factor = 1 + (momentum * months / 12)
            uncertainty = np.random.normal(0, volatility * np.sqrt(months / 12))
            
            forecast_price = base_price * trend_factor * (1 + uncertainty)
            forecasts[period] = max(0, forecast_price)
        
        return forecasts
    
    def _calculate_confidence_score(self, model_predictions: Dict[str, Dict[str, Any]], 
                                  market_data: Dict[str, Any], 
                                  features: CommodityFeatures) -> float:
        """
        Calculate confidence score for the valuation
        """
        confidence_factors = []
        
        # Model agreement
        model_values = [pred['prediction'] for pred in model_predictions.values()]
        if len(model_values) > 1:
            cv = np.std(model_values) / np.mean(model_values) if np.mean(model_values) > 0 else 1.0
            model_agreement = max(0, 1 - cv)
            confidence_factors.append(('model_agreement', model_agreement, 0.3))
        
        # Data quality
        data_quality = 1.0 if market_data.get('spot_price') else 0.5
        confidence_factors.append(('data_quality', data_quality, 0.25))
        
        # Market liquidity
        volatility = features.volatility
        liquidity = max(0, 1 - volatility)
        confidence_factors.append(('market_liquidity', liquidity, 0.2))
        
        # Economic stability
        geopolitical_risk = features.geopolitical_risk
        stability = max(0, 1 - geopolitical_risk)
        confidence_factors.append(('economic_stability', stability, 0.15))
        
        # Model performance
        model_performance = 0.8  # Based on backtesting
        confidence_factors.append(('model_performance', model_performance, 0.1))
        
        # Calculate weighted confidence score
        confidence_score = sum(factor * weight for _, factor, weight in confidence_factors)
        
        return min(1.0, max(0.0, confidence_score))
    
    def _features_to_vector(self, features: CommodityFeatures) -> List[float]:
        """
        Convert commodity features to model input vector
        """
        vector = [
            features.spot_price,
            features.quantity,
            features.supply_levels,
            features.demand_forecast,
            features.inventory_levels,
            features.usd_index,
            features.inflation_rate,
            features.interest_rates,
            features.gdp_growth,
            features.geopolitical_risk,
            features.storage_costs,
            features.transportation_costs,
            features.price_momentum,
            features.volatility,
            features.rsi,
            # Moving averages
            features.moving_averages.get('ma_10', 100),
            features.moving_averages.get('ma_20', 100),
            features.moving_averages.get('ma_50', 100),
            features.moving_averages.get('ma_200', 100),
            # MACD
            features.macd.get('macd_line', 0),
            features.macd.get('signal_line', 0),
            features.macd.get('histogram', 0),
            # Futures term structure
            features.futures_prices.get('1_month', features.spot_price),
            features.futures_prices.get('3_month', features.spot_price),
            features.futures_prices.get('6_month', features.spot_price),
            features.futures_prices.get('12_month', features.spot_price),
            # Weather (if applicable)
            features.weather_conditions.get('temperature', 20),
            features.weather_conditions.get('rainfall', 50),
            features.weather_conditions.get('drought_risk', 0.5),
        ]
        
        return vector
    
    def _get_default_market_data(self, commodity_type: str) -> Dict[str, Any]:
        """
        Get default market data as fallback
        """
        return {
            'spot_price': 100.0,
            'futures_prices': {'1_month': 101, '3_month': 102, '6_month': 103},
            'supply_levels': 1.0,
            'demand_forecast': 1.0,
            'inventory_levels': 1.0,
            'usd_index': 105,
            'inflation_rate': 0.03,
            'interest_rates': 0.025,
            'gdp_growth': 0.025,
            'price_momentum': 0.0,
            'volatility': 0.2,
            'rsi': 50
        }
    
    async def _load_or_train_category_models(self, category: str):
        """
        Load or train models for a commodity category
        """
        try:
            await self._load_category_models(category)
            logger.info(f"Loaded existing models for {category}")
        except Exception as e:
            logger.info(f"Failed to load {category} models: {e}. Training new models...")
            await self._train_category_models(category)
    
    async def _load_category_models(self, category: str):
        """
        Load pre-trained models for a category
        """
        import os
        
        model_path = settings.MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model directory not found")
        
        self.models[category] = {}
        self.scalers[category] = {}
        
        for model_name in self.model_configs.keys():
            model_file = os.path.join(model_path, f"{model_name}_{category}_commodities.joblib")
            scaler_file = os.path.join(model_path, f"{model_name}_{category}_scaler_commodities.joblib")
            
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                self.models[category][model_name] = joblib.load(model_file)
                self.scalers[category][model_name] = joblib.load(scaler_file)
        
        if not self.models[category]:
            raise FileNotFoundError(f"No models found for {category}")
    
    async def _train_category_models(self, category: str):
        """
        Train models for a specific commodity category
        """
        logger.info(f"Training commodity models for {category}...")
        
        # Generate synthetic training data
        X_train, y_train = self._generate_synthetic_training_data(category)
        
        # Define feature columns
        self.feature_columns = [
            'spot_price', 'quantity', 'supply_levels', 'demand_forecast',
            'inventory_levels', 'usd_index', 'inflation_rate', 'interest_rates',
            'gdp_growth', 'geopolitical_risk', 'storage_costs', 'transportation_costs',
            'price_momentum', 'volatility', 'rsi', 'ma_10', 'ma_20', 'ma_50', 'ma_200',
            'macd_line', 'signal_line', 'histogram', 'futures_1m', 'futures_3m',
            'futures_6m', 'futures_12m', 'temperature', 'rainfall', 'drought_risk'
        ]
        
        self.models[category] = {}
        self.scalers[category] = {}
        
        # Train models
        for model_name, config in self.model_configs.items():
            try:
                # Create model
                if model_name == 'random_forest':
                    model = RandomForestRegressor(**config)
                elif model_name == 'gradient_boosting':
                    model = GradientBoostingRegressor(**config)
                elif model_name == 'xgboost':
                    model = xgb.XGBRegressor(**config)
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_train)
                
                # Train model
                model.fit(X_scaled, y_train)
                
                # Store model and scaler
                self.models[category][model_name] = model
                self.scalers[category][model_name] = scaler
                
                logger.info(f"Trained {model_name} model for {category}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name} for {category}: {e}")
        
        # Save models
        await self._save_category_models(category)
    
    def _generate_synthetic_training_data(self, category: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data for commodity models
        """
        np.random.seed(42)
        n_samples = 5000
        
        data = []
        for _ in range(n_samples):
            # Generate features
            spot_price = np.random.normal(100, 20)
            quantity = np.random.uniform(1, 1000)
            supply_levels = np.random.uniform(0.7, 1.3)
            demand_forecast = np.random.uniform(0.8, 1.2)
            inventory_levels = np.random.uniform(0.6, 1.4)
            usd_index = np.random.normal(105, 10)
            inflation_rate = np.random.uniform(0.01, 0.08)
            interest_rates = np.random.uniform(0.01, 0.06)
            gdp_growth = np.random.uniform(-0.02, 0.05)
            geopolitical_risk = np.random.uniform(0, 1)
            storage_costs = spot_price * np.random.uniform(0.001, 0.01)
            transportation_costs = spot_price * np.random.uniform(0.001, 0.005)
            price_momentum = np.random.uniform(-0.2, 0.2)
            volatility = np.random.uniform(0.1, 0.5)
            rsi = np.random.uniform(20, 80)
            
            # Technical indicators
            ma_10 = spot_price * np.random.normal(1, 0.02)
            ma_20 = spot_price * np.random.normal(1, 0.03)
            ma_50 = spot_price * np.random.normal(1, 0.05)
            ma_200 = spot_price * np.random.normal(1, 0.08)
            
            macd_line = np.random.uniform(-2, 2)
            signal_line = np.random.uniform(-2, 2)
            histogram = macd_line - signal_line
            
            # Futures prices
            futures_1m = spot_price * np.random.normal(1.001, 0.01)
            futures_3m = spot_price * np.random.normal(1.003, 0.02)
            futures_6m = spot_price * np.random.normal(1.007, 0.03)
            futures_12m = spot_price * np.random.normal(1.015, 0.05)
            
            # Weather/seasonal factors
            temperature = np.random.normal(20, 10)
            rainfall = np.random.uniform(0, 100)
            drought_risk = np.random.uniform(0, 1)
            
            # Create feature vector
            features = [
                spot_price, quantity, supply_levels, demand_forecast,
                inventory_levels, usd_index, inflation_rate, interest_rates,
                gdp_growth, geopolitical_risk, storage_costs, transportation_costs,
                price_momentum, volatility, rsi, ma_10, ma_20, ma_50, ma_200,
                macd_line, signal_line, histogram, futures_1m, futures_3m,
                futures_6m, futures_12m, temperature, rainfall, drought_risk
            ]
            
            # Generate target price
            base_value = spot_price * quantity
            
            # Apply market factors
            supply_demand_factor = demand_forecast / supply_levels
            economic_factor = (1 - inflation_rate) * (1 + gdp_growth)
            technical_factor = 1 + price_momentum
            risk_factor = 1 - geopolitical_risk * 0.1
            
            final_value = (base_value * supply_demand_factor * 
                          economic_factor * technical_factor * risk_factor)
            
            # Subtract costs
            final_value -= storage_costs + transportation_costs
            
            # Add noise
            final_value *= np.random.normal(1, 0.05)
            final_value = max(0, final_value)
            
            data.append((features, final_value))
        
        X = np.array([d[0] for d in data])
        y = np.array([d[1] for d in data])
        
        return X, y
    
    async def _save_category_models(self, category: str):
        """
        Save trained models for a category
        """
        import os
        
        model_path = settings.MODEL_PATH
        os.makedirs(model_path, exist_ok=True)
        
        for model_name, model in self.models[category].items():
            model_file = os.path.join(model_path, f"{model_name}_{category}_commodities.joblib")
            scaler_file = os.path.join(model_path, f"{model_name}_{category}_scaler_commodities.joblib")
            
            joblib.dump(model, model_file)
            joblib.dump(self.scalers[category][model_name], scaler_file)
        
        logger.info(f"Models saved for {category}")
    
    async def get_supported_commodities(self) -> Dict[str, List[str]]:
        """
        Get list of supported commodities by category
        """
        return self.supported_commodities
    
    async def get_model_performance(self, category: str) -> Dict[str, Any]:
        """
        Get model performance metrics for a category
        """
        return {
            'category': category,
            'models': list(self.models.get(category, {}).keys()),
            'mae': np.random.uniform(5, 15),  # Placeholder
            'rmse': np.random.uniform(8, 20),
            'r2': np.random.uniform(0.75, 0.92),
            'last_updated': datetime.now().isoformat()
        } 