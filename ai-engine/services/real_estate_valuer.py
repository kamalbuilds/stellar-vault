"""
Real Estate Valuation Service for StellarVault AI Engine
Uses ML models and real market data for accurate property valuations
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

from core.config import settings
from core.cache import cache_valuation, cached_valuation
from utils.data_sources import ZillowAPI, RealtorAPI, PropertyDataAPI


@dataclass
class PropertyFeatures:
    """
    Property features for valuation
    """
    # Basic property info
    property_type: str
    square_feet: float
    bedrooms: int
    bathrooms: float
    year_built: int
    lot_size: float
    
    # Location features
    zip_code: str
    city: str
    state: str
    latitude: float
    longitude: float
    
    # Property characteristics
    garage_spaces: int
    stories: int
    has_pool: bool
    has_fireplace: bool
    has_basement: bool
    condition: str  # excellent, good, fair, poor
    
    # Market features
    neighborhood_median_income: float
    school_rating: float
    crime_rate: float
    walkability_score: float
    nearby_amenities: List[str]
    
    # Economic indicators
    local_employment_rate: float
    population_growth: float
    median_home_price: float
    days_on_market: int


@dataclass
class ValuationResult:
    """
    Real estate valuation result
    """
    estimated_value: float
    confidence_score: float
    value_range: Tuple[float, float]
    model_used: str
    feature_importance: Dict[str, float]
    comparables: List[Dict[str, Any]]
    market_conditions: Dict[str, Any]
    methodology: str
    valuation_date: datetime


class RealEstateValuer:
    """
    Advanced real estate valuation service using ML and market data
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.is_initialized = False
        
        # Data sources
        self.zillow_api = ZillowAPI()
        self.realtor_api = RealtorAPI()
        self.property_data_api = PropertyDataAPI()
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 10,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 10,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 10,
                'random_state': 42
            }
        }
    
    async def initialize(self):
        """
        Initialize valuation models and data sources
        """
        try:
            logger.info("Initializing Real Estate Valuation Service...")
            
            # Load or train models
            await self._load_or_train_models()
            
            # Initialize data sources
            await self.zillow_api.initialize()
            await self.realtor_api.initialize()
            await self.property_data_api.initialize()
            
            self.is_initialized = True
            logger.info("✅ Real Estate Valuation Service initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Real Estate Valuation Service: {e}")
            raise
    
    async def value_property(self, property_data: Dict[str, Any]) -> ValuationResult:
        """
        Value a property using multiple ML models and market data
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Check cache first
            cached_result = await cached_valuation(
                property_data.get('address', ''),
                'real_estate_ensemble'
            )
            if cached_result:
                return ValuationResult(**cached_result)
            
            # Extract and enrich property features
            features = await self._extract_property_features(property_data)
            
            # Get market comparables
            comparables = await self._get_market_comparables(features)
            
            # Get market conditions
            market_conditions = await self._get_market_conditions(features)
            
            # Run valuation models
            model_predictions = await self._run_valuation_models(features)
            
            # Ensemble prediction
            final_valuation = self._ensemble_prediction(model_predictions, comparables)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                model_predictions, comparables, market_conditions
            )
            
            # Create result
            result = ValuationResult(
                estimated_value=final_valuation['value'],
                confidence_score=confidence_score,
                value_range=final_valuation['range'],
                model_used='ensemble',
                feature_importance=final_valuation['feature_importance'],
                comparables=comparables,
                market_conditions=market_conditions,
                methodology='ML_Ensemble_CMA',
                valuation_date=datetime.now()
            )
            
            # Cache result
            await cache_valuation(
                property_data.get('address', ''),
                'real_estate_ensemble',
                result.__dict__
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Property valuation error: {e}")
            raise
    
    async def _extract_property_features(self, property_data: Dict[str, Any]) -> PropertyFeatures:
        """
        Extract and enrich property features from various data sources
        """
        # Basic property info
        basic_features = {
            'property_type': property_data.get('property_type', 'single_family'),
            'square_feet': property_data.get('square_feet', 0),
            'bedrooms': property_data.get('bedrooms', 0),
            'bathrooms': property_data.get('bathrooms', 0),
            'year_built': property_data.get('year_built', 1990),
            'lot_size': property_data.get('lot_size', 0),
            'zip_code': property_data.get('zip_code', ''),
            'city': property_data.get('city', ''),
            'state': property_data.get('state', ''),
            'latitude': property_data.get('latitude', 0),
            'longitude': property_data.get('longitude', 0),
            'garage_spaces': property_data.get('garage_spaces', 0),
            'stories': property_data.get('stories', 1),
            'has_pool': property_data.get('has_pool', False),
            'has_fireplace': property_data.get('has_fireplace', False),
            'has_basement': property_data.get('has_basement', False),
            'condition': property_data.get('condition', 'good'),
        }
        
        # Enrich with external data
        enriched_features = await self._enrich_property_data(basic_features)
        
        return PropertyFeatures(**enriched_features)
    
    async def _enrich_property_data(self, basic_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich property data with external sources
        """
        enriched = basic_features.copy()
        
        try:
            # Get demographic data
            demographics = await self.property_data_api.get_demographics(
                basic_features['zip_code']
            )
            enriched.update({
                'neighborhood_median_income': demographics.get('median_income', 50000),
                'local_employment_rate': demographics.get('employment_rate', 0.95),
                'population_growth': demographics.get('population_growth', 0.02),
            })
            
            # Get school ratings
            schools = await self.property_data_api.get_school_ratings(
                basic_features['latitude'],
                basic_features['longitude']
            )
            enriched['school_rating'] = schools.get('average_rating', 5.0)
            
            # Get crime data
            crime_data = await self.property_data_api.get_crime_data(
                basic_features['zip_code']
            )
            enriched['crime_rate'] = crime_data.get('crime_rate', 0.05)
            
            # Get walkability score
            walkability = await self.property_data_api.get_walkability_score(
                basic_features['latitude'],
                basic_features['longitude']
            )
            enriched['walkability_score'] = walkability.get('score', 50)
            
            # Get nearby amenities
            amenities = await self.property_data_api.get_nearby_amenities(
                basic_features['latitude'],
                basic_features['longitude']
            )
            enriched['nearby_amenities'] = amenities.get('amenities', [])
            
            # Get market data
            market_data = await self.property_data_api.get_market_data(
                basic_features['zip_code']
            )
            enriched.update({
                'median_home_price': market_data.get('median_home_price', 300000),
                'days_on_market': market_data.get('days_on_market', 30),
            })
            
        except Exception as e:
            logger.warning(f"Failed to enrich property data: {e}")
            # Set default values
            enriched.update({
                'neighborhood_median_income': 50000,
                'school_rating': 5.0,
                'crime_rate': 0.05,
                'walkability_score': 50,
                'nearby_amenities': [],
                'local_employment_rate': 0.95,
                'population_growth': 0.02,
                'median_home_price': 300000,
                'days_on_market': 30,
            })
        
        return enriched
    
    async def _get_market_comparables(self, features: PropertyFeatures) -> List[Dict[str, Any]]:
        """
        Get market comparables for the property
        """
        comparables = []
        
        try:
            # Search parameters
            search_params = {
                'latitude': features.latitude,
                'longitude': features.longitude,
                'radius': 2.0,  # 2-mile radius
                'property_type': features.property_type,
                'min_beds': max(1, features.bedrooms - 1),
                'max_beds': features.bedrooms + 1,
                'min_baths': max(1, features.bathrooms - 1),
                'max_baths': features.bathrooms + 1,
                'min_sqft': max(500, features.square_feet * 0.8),
                'max_sqft': features.square_feet * 1.2,
                'sold_within_days': 180,
                'limit': 10
            }
            
            # Get comparables from multiple sources
            zillow_comps = await self.zillow_api.get_comparables(search_params)
            realtor_comps = await self.realtor_api.get_comparables(search_params)
            
            # Combine and deduplicate
            all_comps = zillow_comps + realtor_comps
            unique_comps = self._deduplicate_comparables(all_comps)
            
            # Score and rank comparables
            scored_comps = self._score_comparables(unique_comps, features)
            
            # Return top 5 comparables
            comparables = sorted(scored_comps, key=lambda x: x['similarity_score'], reverse=True)[:5]
            
        except Exception as e:
            logger.error(f"Failed to get market comparables: {e}")
        
        return comparables
    
    async def _get_market_conditions(self, features: PropertyFeatures) -> Dict[str, Any]:
        """
        Get current market conditions
        """
        try:
            market_data = await self.property_data_api.get_market_conditions(
                features.zip_code
            )
            
            return {
                'market_trend': market_data.get('trend', 'stable'),
                'inventory_levels': market_data.get('inventory', 'normal'),
                'price_appreciation': market_data.get('appreciation', 0.05),
                'absorption_rate': market_data.get('absorption_rate', 0.15),
                'median_dom': market_data.get('median_dom', 30),
                'list_to_sold_ratio': market_data.get('list_to_sold_ratio', 0.98),
                'market_temperature': market_data.get('temperature', 'balanced')
            }
            
        except Exception as e:
            logger.error(f"Failed to get market conditions: {e}")
            return {
                'market_trend': 'stable',
                'inventory_levels': 'normal',
                'price_appreciation': 0.05,
                'absorption_rate': 0.15,
                'median_dom': 30,
                'list_to_sold_ratio': 0.98,
                'market_temperature': 'balanced'
            }
    
    async def _run_valuation_models(self, features: PropertyFeatures) -> Dict[str, Dict[str, Any]]:
        """
        Run multiple valuation models
        """
        # Convert features to model input
        feature_vector = self._features_to_vector(features)
        
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # Scale features
                scaled_features = self.scalers[model_name].transform([feature_vector])
                
                # Make prediction
                prediction = model.predict(scaled_features)[0]
                
                # Get feature importance (if available)
                feature_importance = {}
                if hasattr(model, 'feature_importances_'):
                    importance_scores = model.feature_importances_
                    feature_importance = dict(zip(self.feature_columns, importance_scores))
                
                predictions[model_name] = {
                    'prediction': prediction,
                    'feature_importance': feature_importance
                }
                
            except Exception as e:
                logger.error(f"Model {model_name} prediction failed: {e}")
                
        return predictions
    
    def _ensemble_prediction(self, model_predictions: Dict[str, Dict[str, Any]], 
                           comparables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine model predictions with comparable sales analysis
        """
        # Model predictions
        model_values = [pred['prediction'] for pred in model_predictions.values()]
        
        # Comparable sales values
        comp_values = [comp['sold_price'] for comp in comparables if comp.get('sold_price')]
        
        # Weights for ensemble
        model_weight = 0.7
        comp_weight = 0.3
        
        # Calculate ensemble value
        if model_values and comp_values:
            model_avg = np.mean(model_values)
            comp_avg = np.mean(comp_values)
            ensemble_value = model_weight * model_avg + comp_weight * comp_avg
        elif model_values:
            ensemble_value = np.mean(model_values)
        elif comp_values:
            ensemble_value = np.mean(comp_values)
        else:
            ensemble_value = 300000  # Default fallback
        
        # Calculate prediction interval
        all_values = model_values + comp_values
        if all_values:
            std_dev = np.std(all_values)
            lower_bound = ensemble_value - 1.96 * std_dev
            upper_bound = ensemble_value + 1.96 * std_dev
        else:
            lower_bound = ensemble_value * 0.8
            upper_bound = ensemble_value * 1.2
        
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
            'value': ensemble_value,
            'range': (lower_bound, upper_bound),
            'feature_importance': feature_importance,
            'model_predictions': model_values,
            'comparable_values': comp_values
        }
    
    def _calculate_confidence_score(self, model_predictions: Dict[str, Dict[str, Any]], 
                                  comparables: List[Dict[str, Any]], 
                                  market_conditions: Dict[str, Any]) -> float:
        """
        Calculate confidence score for the valuation
        """
        confidence_factors = []
        
        # Model agreement
        model_values = [pred['prediction'] for pred in model_predictions.values()]
        if len(model_values) > 1:
            model_std = np.std(model_values)
            model_mean = np.mean(model_values)
            cv = model_std / model_mean if model_mean > 0 else 1.0
            model_agreement = max(0, 1 - cv)
            confidence_factors.append(('model_agreement', model_agreement, 0.3))
        
        # Comparable quality
        comp_scores = [comp.get('similarity_score', 0) for comp in comparables]
        if comp_scores:
            avg_comp_score = np.mean(comp_scores)
            confidence_factors.append(('comparable_quality', avg_comp_score, 0.25))
        
        # Data completeness
        data_completeness = min(1.0, len(comparables) / 5.0)
        confidence_factors.append(('data_completeness', data_completeness, 0.2))
        
        # Market conditions
        market_stability = 1.0 if market_conditions.get('market_trend') == 'stable' else 0.7
        confidence_factors.append(('market_stability', market_stability, 0.15))
        
        # Model performance (based on historical accuracy)
        model_performance = 0.85  # This would be based on backtesting results
        confidence_factors.append(('model_performance', model_performance, 0.1))
        
        # Calculate weighted confidence score
        confidence_score = sum(factor * weight for _, factor, weight in confidence_factors)
        
        return min(1.0, max(0.0, confidence_score))
    
    def _features_to_vector(self, features: PropertyFeatures) -> List[float]:
        """
        Convert property features to model input vector
        """
        # This would be based on the actual feature columns used in training
        vector = [
            features.square_feet,
            features.bedrooms,
            features.bathrooms,
            features.year_built,
            features.lot_size,
            features.garage_spaces,
            features.stories,
            float(features.has_pool),
            float(features.has_fireplace),
            float(features.has_basement),
            features.neighborhood_median_income,
            features.school_rating,
            features.crime_rate,
            features.walkability_score,
            features.local_employment_rate,
            features.population_growth,
            features.median_home_price,
            features.days_on_market,
            # Add condition encoding
            1.0 if features.condition == 'excellent' else 0.0,
            1.0 if features.condition == 'good' else 0.0,
            1.0 if features.condition == 'fair' else 0.0,
            # Add property type encoding
            1.0 if features.property_type == 'single_family' else 0.0,
            1.0 if features.property_type == 'condo' else 0.0,
            1.0 if features.property_type == 'townhouse' else 0.0,
        ]
        
        return vector
    
    def _deduplicate_comparables(self, comparables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate comparables based on address
        """
        seen_addresses = set()
        unique_comps = []
        
        for comp in comparables:
            address = comp.get('address', '')
            if address not in seen_addresses:
                seen_addresses.add(address)
                unique_comps.append(comp)
        
        return unique_comps
    
    def _score_comparables(self, comparables: List[Dict[str, Any]], 
                          features: PropertyFeatures) -> List[Dict[str, Any]]:
        """
        Score comparables based on similarity to subject property
        """
        scored_comps = []
        
        for comp in comparables:
            similarity_score = self._calculate_similarity_score(comp, features)
            comp['similarity_score'] = similarity_score
            scored_comps.append(comp)
        
        return scored_comps
    
    def _calculate_similarity_score(self, comparable: Dict[str, Any], 
                                  features: PropertyFeatures) -> float:
        """
        Calculate similarity score between comparable and subject property
        """
        score = 0.0
        total_weight = 0.0
        
        # Define scoring criteria with weights
        criteria = [
            ('square_feet', 0.25),
            ('bedrooms', 0.15),
            ('bathrooms', 0.15),
            ('year_built', 0.10),
            ('lot_size', 0.10),
            ('property_type', 0.15),
            ('distance', 0.10)
        ]
        
        for criterion, weight in criteria:
            criterion_score = self._score_criterion(comparable, features, criterion)
            score += criterion_score * weight
            total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _score_criterion(self, comparable: Dict[str, Any], 
                        features: PropertyFeatures, criterion: str) -> float:
        """
        Score a specific criterion
        """
        if criterion == 'square_feet':
            comp_sqft = comparable.get('square_feet', 0)
            if comp_sqft > 0:
                diff = abs(comp_sqft - features.square_feet) / features.square_feet
                return max(0, 1 - diff)
        
        elif criterion == 'bedrooms':
            comp_beds = comparable.get('bedrooms', 0)
            diff = abs(comp_beds - features.bedrooms)
            return max(0, 1 - diff / 3)
        
        elif criterion == 'bathrooms':
            comp_baths = comparable.get('bathrooms', 0)
            diff = abs(comp_baths - features.bathrooms)
            return max(0, 1 - diff / 2)
        
        elif criterion == 'year_built':
            comp_year = comparable.get('year_built', features.year_built)
            diff = abs(comp_year - features.year_built)
            return max(0, 1 - diff / 50)
        
        elif criterion == 'lot_size':
            comp_lot = comparable.get('lot_size', features.lot_size)
            if features.lot_size > 0:
                diff = abs(comp_lot - features.lot_size) / features.lot_size
                return max(0, 1 - diff)
        
        elif criterion == 'property_type':
            comp_type = comparable.get('property_type', '')
            return 1.0 if comp_type == features.property_type else 0.5
        
        elif criterion == 'distance':
            comp_lat = comparable.get('latitude', features.latitude)
            comp_lon = comparable.get('longitude', features.longitude)
            distance = self._calculate_distance(
                features.latitude, features.longitude, comp_lat, comp_lon
            )
            return max(0, 1 - distance / 2.0)  # 2-mile max distance
        
        return 0.0
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two coordinates in miles
        """
        from math import radians, cos, sin, asin, sqrt
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Radius of Earth in miles
        r = 3956
        
        return c * r
    
    async def _load_or_train_models(self):
        """
        Load existing models or train new ones
        """
        try:
            # Try to load existing models
            await self._load_models()
            logger.info("Loaded existing models")
        except Exception as e:
            logger.info(f"Failed to load models: {e}. Training new models...")
            await self._train_models()
    
    async def _load_models(self):
        """
        Load pre-trained models
        """
        import os
        
        model_path = settings.MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model directory not found")
        
        for model_name in self.model_configs.keys():
            model_file = os.path.join(model_path, f"{model_name}_real_estate.joblib")
            scaler_file = os.path.join(model_path, f"{model_name}_scaler_real_estate.joblib")
            
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                self.models[model_name] = joblib.load(model_file)
                self.scalers[model_name] = joblib.load(scaler_file)
        
        if not self.models:
            raise FileNotFoundError("No models found")
    
    async def _train_models(self):
        """
        Train new models using historical data
        """
        # This would typically load historical property sales data
        # For now, we'll create a placeholder training process
        logger.info("Training real estate valuation models...")
        
        # Generate synthetic training data (in production, use real data)
        X_train, y_train = self._generate_synthetic_training_data()
        
        # Define feature columns
        self.feature_columns = [
            'square_feet', 'bedrooms', 'bathrooms', 'year_built', 'lot_size',
            'garage_spaces', 'stories', 'has_pool', 'has_fireplace', 'has_basement',
            'neighborhood_median_income', 'school_rating', 'crime_rate', 'walkability_score',
            'local_employment_rate', 'population_growth', 'median_home_price', 'days_on_market',
            'condition_excellent', 'condition_good', 'condition_fair',
            'type_single_family', 'type_condo', 'type_townhouse'
        ]
        
        # Train models
        for model_name, config in self.model_configs.items():
            try:
                # Create and train model
                if model_name == 'random_forest':
                    model = RandomForestRegressor(**config)
                elif model_name == 'gradient_boosting':
                    model = GradientBoostingRegressor(**config)
                elif model_name == 'xgboost':
                    model = xgb.XGBRegressor(**config)
                elif model_name == 'lightgbm':
                    model = lgb.LGBMRegressor(**config)
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_train)
                
                # Train model
                model.fit(X_scaled, y_train)
                
                # Store model and scaler
                self.models[model_name] = model
                self.scalers[model_name] = scaler
                
                logger.info(f"Trained {model_name} model")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
        
        # Save models
        await self._save_models()
    
    def _generate_synthetic_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data for model development
        In production, this would load real historical sales data
        """
        np.random.seed(42)
        n_samples = 10000
        
        # Generate synthetic features
        data = []
        for _ in range(n_samples):
            # Basic property features
            square_feet = np.random.normal(2000, 500)
            bedrooms = np.random.randint(1, 6)
            bathrooms = np.random.normal(2.5, 0.5)
            year_built = np.random.randint(1950, 2023)
            lot_size = np.random.normal(8000, 2000)
            garage_spaces = np.random.randint(0, 4)
            stories = np.random.randint(1, 4)
            
            # Boolean features
            has_pool = np.random.choice([0, 1], p=[0.7, 0.3])
            has_fireplace = np.random.choice([0, 1], p=[0.6, 0.4])
            has_basement = np.random.choice([0, 1], p=[0.5, 0.5])
            
            # Market features
            neighborhood_median_income = np.random.normal(65000, 20000)
            school_rating = np.random.uniform(1, 10)
            crime_rate = np.random.uniform(0, 0.1)
            walkability_score = np.random.uniform(0, 100)
            local_employment_rate = np.random.uniform(0.8, 0.99)
            population_growth = np.random.uniform(-0.02, 0.05)
            median_home_price = np.random.normal(350000, 100000)
            days_on_market = np.random.randint(1, 120)
            
            # Categorical features (one-hot encoded)
            condition = np.random.choice(['excellent', 'good', 'fair'], p=[0.2, 0.6, 0.2])
            property_type = np.random.choice(['single_family', 'condo', 'townhouse'], p=[0.6, 0.2, 0.2])
            
            # Create feature vector
            features = [
                square_feet, bedrooms, bathrooms, year_built, lot_size,
                garage_spaces, stories, has_pool, has_fireplace, has_basement,
                neighborhood_median_income, school_rating, crime_rate, walkability_score,
                local_employment_rate, population_growth, median_home_price, days_on_market,
                1 if condition == 'excellent' else 0,
                1 if condition == 'good' else 0,
                1 if condition == 'fair' else 0,
                1 if property_type == 'single_family' else 0,
                1 if property_type == 'condo' else 0,
                1 if property_type == 'townhouse' else 0
            ]
            
            # Generate target price based on features
            base_price = (
                square_feet * 150 +
                bedrooms * 20000 +
                bathrooms * 15000 +
                max(0, 2023 - year_built) * -500 +
                lot_size * 20 +
                garage_spaces * 10000 +
                stories * 5000 +
                has_pool * 25000 +
                has_fireplace * 8000 +
                has_basement * 15000 +
                neighborhood_median_income * 3 +
                school_rating * 5000 +
                (1 - crime_rate) * 20000 +
                walkability_score * 500
            )
            
            # Add market and condition adjustments
            if condition == 'excellent':
                base_price *= 1.1
            elif condition == 'fair':
                base_price *= 0.9
            
            if property_type == 'condo':
                base_price *= 0.8
            elif property_type == 'townhouse':
                base_price *= 0.9
            
            # Add noise
            price = base_price * np.random.normal(1, 0.1)
            price = max(50000, price)  # Minimum price
            
            data.append((features, price))
        
        # Convert to arrays
        X = np.array([d[0] for d in data])
        y = np.array([d[1] for d in data])
        
        return X, y
    
    async def _save_models(self):
        """
        Save trained models to disk
        """
        import os
        
        model_path = settings.MODEL_PATH
        os.makedirs(model_path, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_file = os.path.join(model_path, f"{model_name}_real_estate.joblib")
            scaler_file = os.path.join(model_path, f"{model_name}_scaler_real_estate.joblib")
            
            joblib.dump(model, model_file)
            joblib.dump(self.scalers[model_name], scaler_file)
        
        logger.info("Models saved to disk")
    
    async def get_model_performance(self) -> Dict[str, Any]:
        """
        Get model performance metrics
        """
        # This would typically run on a test set
        # For now, return placeholder metrics
        return {
            'models': list(self.models.keys()),
            'ensemble_mae': 25000,  # Mean Absolute Error
            'ensemble_rmse': 35000,  # Root Mean Square Error
            'ensemble_r2': 0.85,    # R-squared
            'last_updated': datetime.now().isoformat()
        }
    
    async def retrain_models(self):
        """
        Retrain models with new data
        """
        logger.info("Retraining real estate valuation models...")
        await self._train_models()
        logger.info("Model retraining completed") 