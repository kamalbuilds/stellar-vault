"""
Market Analytics Service for StellarVault AI Engine

This service provides comprehensive market analysis including:
- Real-time market insights and trend analysis
- Sentiment analysis from news and social media
- Predictive modeling for asset prices
- Market volatility and correlation analysis
- Economic indicator monitoring
- Cross-asset class analysis for RWA markets
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import statistics
from collections import defaultdict
from loguru import logger

from ..core.config import settings
from ..utils.data_sources import DataSourceClient


class TrendDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"


class SentimentScore(Enum):
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class MarketRegime(Enum):
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    CONSOLIDATION = "consolidation"
    CRISIS = "crisis"
    RECOVERY = "recovery"


class AssetClass(Enum):
    REAL_ESTATE = "real_estate"
    COMMODITIES = "commodities"
    ART_COLLECTIBLES = "art_collectibles"
    BONDS = "bonds"
    EQUITIES = "equities"
    CRYPTO = "crypto"
    CURRENCIES = "currencies"


@dataclass
class MarketIndicator:
    """Market indicator data"""
    name: str
    value: float
    previous_value: float
    change_percent: float
    timestamp: datetime
    source: str
    confidence: float


@dataclass
class SentimentAnalysis:
    """Sentiment analysis result"""
    overall_sentiment: SentimentScore
    sentiment_score: float  # -1 to 1
    news_sentiment: float
    social_sentiment: float
    analyst_sentiment: float
    sentiment_drivers: List[str]
    confidence: float
    data_quality: float


@dataclass
class TrendAnalysis:
    """Trend analysis result"""
    direction: TrendDirection
    strength: float  # 0 to 1
    duration_days: int
    support_level: Optional[float]
    resistance_level: Optional[float]
    momentum_indicators: Dict[str, float]
    reversal_probability: float
    confidence: float


@dataclass
class VolatilityAnalysis:
    """Volatility analysis result"""
    current_volatility: float
    historical_volatility: float
    implied_volatility: Optional[float]
    volatility_percentile: float
    volatility_regime: str  # low, normal, high, extreme
    volatility_forecast: Dict[str, float]  # 1d, 7d, 30d forecasts


@dataclass
class CorrelationAnalysis:
    """Cross-asset correlation analysis"""
    correlation_matrix: Dict[str, Dict[str, float]]
    correlation_changes: Dict[str, float]
    diversification_benefits: Dict[str, float]
    risk_concentrations: List[str]
    correlation_regime: str


@dataclass
class PredictiveModel:
    """Predictive model result"""
    model_type: str
    predictions: Dict[str, float]  # timeframe -> predicted value
    confidence_intervals: Dict[str, Tuple[float, float]]
    feature_importance: Dict[str, float]
    model_accuracy: float
    risk_scenarios: Dict[str, float]


@dataclass
class MarketInsight:
    """Market insight"""
    insight_id: str
    category: str
    priority: str  # high, medium, low
    title: str
    description: str
    affected_assets: List[str]
    recommendation: str
    confidence: float
    created_at: datetime
    expires_at: Optional[datetime]


@dataclass
class MarketAnalyticsReport:
    """Comprehensive market analytics report"""
    report_id: str
    generation_time: datetime
    market_overview: Dict[str, Any]
    sentiment_analysis: SentimentAnalysis
    trend_analysis: Dict[str, TrendAnalysis]
    volatility_analysis: Dict[str, VolatilityAnalysis]
    correlation_analysis: CorrelationAnalysis
    predictive_models: Dict[str, PredictiveModel]
    market_insights: List[MarketInsight]
    risk_alerts: List[str]
    opportunities: List[str]
    data_quality_score: float


class MarketAnalyticsService:
    """
    Advanced market analytics service for RWA markets
    """
    
    def __init__(self):
        self.data_client = DataSourceClient()
        self.sentiment_cache = {}
        self.trend_cache = {}
        self.volatility_cache = {}
        self.correlation_cache = {}
        self.last_update = None
        
    async def initialize(self):
        """Initialize the market analytics service"""
        try:
            await self._update_market_data()
            await self._initialize_models()
            logger.info("Market analytics service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize market analytics service: {e}")
            raise
    
    async def generate_market_report(
        self,
        asset_classes: List[AssetClass] = None,
        time_horizon: str = "1M",
        include_predictions: bool = True
    ) -> MarketAnalyticsReport:
        """
        Generate comprehensive market analytics report
        
        Args:
            asset_classes: Asset classes to analyze
            time_horizon: Analysis time horizon (1D, 1W, 1M, 3M, 1Y)
            include_predictions: Whether to include predictive models
            
        Returns:
            MarketAnalyticsReport with comprehensive analysis
        """
        try:
            if not asset_classes:
                asset_classes = list(AssetClass)
            
            # Ensure fresh data
            await self._ensure_fresh_data()
            
            # Market overview
            market_overview = await self._generate_market_overview(asset_classes)
            
            # Sentiment analysis
            sentiment_analysis = await self._analyze_market_sentiment(asset_classes)
            
            # Trend analysis
            trend_analysis = await self._analyze_trends(asset_classes, time_horizon)
            
            # Volatility analysis
            volatility_analysis = await self._analyze_volatility(asset_classes)
            
            # Correlation analysis
            correlation_analysis = await self._analyze_correlations(asset_classes)
            
            # Predictive models
            predictive_models = {}
            if include_predictions:
                predictive_models = await self._generate_predictions(asset_classes)
            
            # Generate insights
            market_insights = await self._generate_market_insights(
                sentiment_analysis, trend_analysis, volatility_analysis, correlation_analysis
            )
            
            # Risk alerts
            risk_alerts = await self._identify_risk_alerts(
                trend_analysis, volatility_analysis, correlation_analysis
            )
            
            # Opportunities
            opportunities = await self._identify_opportunities(
                sentiment_analysis, trend_analysis, predictive_models
            )
            
            # Data quality assessment
            data_quality = await self._assess_data_quality()
            
            return MarketAnalyticsReport(
                report_id=f"MAR_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                generation_time=datetime.now(),
                market_overview=market_overview,
                sentiment_analysis=sentiment_analysis,
                trend_analysis=trend_analysis,
                volatility_analysis=volatility_analysis,
                correlation_analysis=correlation_analysis,
                predictive_models=predictive_models,
                market_insights=market_insights,
                risk_alerts=risk_alerts,
                opportunities=opportunities,
                data_quality_score=data_quality
            )
            
        except Exception as e:
            logger.error(f"Market report generation failed: {e}")
            raise
    
    async def _generate_market_overview(self, asset_classes: List[AssetClass]) -> Dict[str, Any]:
        """Generate market overview"""
        try:
            overview = {
                "market_regime": await self._determine_market_regime(),
                "risk_on_off": await self._assess_risk_sentiment(),
                "key_indicators": await self._get_key_indicators(),
                "market_performance": await self._calculate_market_performance(asset_classes),
                "economic_backdrop": await self._analyze_economic_backdrop()
            }
            
            return overview
            
        except Exception as e:
            logger.error(f"Market overview generation failed: {e}")
            return {
                "market_regime": MarketRegime.CONSOLIDATION.value,
                "risk_on_off": "neutral",
                "key_indicators": {},
                "market_performance": {},
                "economic_backdrop": {}
            }
    
    async def _analyze_market_sentiment(self, asset_classes: List[AssetClass]) -> SentimentAnalysis:
        """Analyze market sentiment across multiple sources"""
        try:
            # News sentiment analysis
            news_sentiment = await self._analyze_news_sentiment(asset_classes)
            
            # Social media sentiment
            social_sentiment = await self._analyze_social_sentiment(asset_classes)
            
            # Analyst sentiment
            analyst_sentiment = await self._analyze_analyst_sentiment(asset_classes)
            
            # Combined sentiment score
            sentiment_weights = {"news": 0.4, "social": 0.3, "analyst": 0.3}
            overall_score = (
                news_sentiment * sentiment_weights["news"] +
                social_sentiment * sentiment_weights["social"] +
                analyst_sentiment * sentiment_weights["analyst"]
            )
            
            # Determine sentiment category
            if overall_score <= -0.6:
                overall_sentiment = SentimentScore.VERY_NEGATIVE
            elif overall_score <= -0.2:
                overall_sentiment = SentimentScore.NEGATIVE
            elif overall_score <= 0.2:
                overall_sentiment = SentimentScore.NEUTRAL
            elif overall_score <= 0.6:
                overall_sentiment = SentimentScore.POSITIVE
            else:
                overall_sentiment = SentimentScore.VERY_POSITIVE
            
            # Identify sentiment drivers
            sentiment_drivers = await self._identify_sentiment_drivers(
                news_sentiment, social_sentiment, analyst_sentiment
            )
            
            return SentimentAnalysis(
                overall_sentiment=overall_sentiment,
                sentiment_score=overall_score,
                news_sentiment=news_sentiment,
                social_sentiment=social_sentiment,
                analyst_sentiment=analyst_sentiment,
                sentiment_drivers=sentiment_drivers,
                confidence=0.8,
                data_quality=0.9
            )
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return SentimentAnalysis(
                overall_sentiment=SentimentScore.NEUTRAL,
                sentiment_score=0.0,
                news_sentiment=0.0,
                social_sentiment=0.0,
                analyst_sentiment=0.0,
                sentiment_drivers=["No data available"],
                confidence=0.5,
                data_quality=0.5
            )
    
    async def _analyze_trends(
        self, 
        asset_classes: List[AssetClass], 
        time_horizon: str
    ) -> Dict[str, TrendAnalysis]:
        """Analyze trends for each asset class"""
        try:
            trend_analyses = {}
            
            for asset_class in asset_classes:
                # Get price data
                price_data = await self._get_price_data(asset_class, time_horizon)
                
                # Calculate trend indicators
                trend_direction = await self._calculate_trend_direction(price_data)
                trend_strength = await self._calculate_trend_strength(price_data)
                duration = await self._calculate_trend_duration(price_data)
                
                # Support and resistance levels
                support, resistance = await self._calculate_support_resistance(price_data)
                
                # Momentum indicators
                momentum = await self._calculate_momentum_indicators(price_data)
                
                # Reversal probability
                reversal_prob = await self._calculate_reversal_probability(
                    price_data, trend_direction, trend_strength
                )
                
                trend_analyses[asset_class.value] = TrendAnalysis(
                    direction=trend_direction,
                    strength=trend_strength,
                    duration_days=duration,
                    support_level=support,
                    resistance_level=resistance,
                    momentum_indicators=momentum,
                    reversal_probability=reversal_prob,
                    confidence=0.85
                )
            
            return trend_analyses
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {}
    
    async def _analyze_volatility(self, asset_classes: List[AssetClass]) -> Dict[str, VolatilityAnalysis]:
        """Analyze volatility for each asset class"""
        try:
            volatility_analyses = {}
            
            for asset_class in asset_classes:
                # Get price data
                price_data = await self._get_price_data(asset_class, "3M")
                
                # Calculate volatilities
                current_vol = await self._calculate_current_volatility(price_data)
                historical_vol = await self._calculate_historical_volatility(price_data)
                
                # Volatility percentile
                vol_percentile = await self._calculate_volatility_percentile(
                    current_vol, price_data
                )
                
                # Volatility regime
                vol_regime = await self._determine_volatility_regime(vol_percentile)
                
                # Volatility forecasts
                vol_forecast = await self._forecast_volatility(price_data)
                
                volatility_analyses[asset_class.value] = VolatilityAnalysis(
                    current_volatility=current_vol,
                    historical_volatility=historical_vol,
                    implied_volatility=None,  # Would need options data
                    volatility_percentile=vol_percentile,
                    volatility_regime=vol_regime,
                    volatility_forecast=vol_forecast
                )
            
            return volatility_analyses
            
        except Exception as e:
            logger.error(f"Volatility analysis failed: {e}")
            return {}
    
    async def _analyze_correlations(self, asset_classes: List[AssetClass]) -> CorrelationAnalysis:
        """Analyze cross-asset correlations"""
        try:
            # Get return data for all asset classes
            returns_data = {}
            for asset_class in asset_classes:
                price_data = await self._get_price_data(asset_class, "1Y")
                returns = np.diff(np.log(price_data)) if len(price_data) > 1 else [0.0]
                returns_data[asset_class.value] = returns
            
            # Calculate correlation matrix
            correlation_matrix = await self._calculate_correlation_matrix(returns_data)
            
            # Calculate correlation changes
            correlation_changes = await self._calculate_correlation_changes(returns_data)
            
            # Diversification benefits
            diversification_benefits = await self._calculate_diversification_benefits(
                correlation_matrix
            )
            
            # Risk concentrations
            risk_concentrations = await self._identify_risk_concentrations(
                correlation_matrix
            )
            
            # Correlation regime
            correlation_regime = await self._determine_correlation_regime(
                correlation_matrix
            )
            
            return CorrelationAnalysis(
                correlation_matrix=correlation_matrix,
                correlation_changes=correlation_changes,
                diversification_benefits=diversification_benefits,
                risk_concentrations=risk_concentrations,
                correlation_regime=correlation_regime
            )
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return CorrelationAnalysis(
                correlation_matrix={},
                correlation_changes={},
                diversification_benefits={},
                risk_concentrations=[],
                correlation_regime="normal"
            )
    
    async def _generate_predictions(self, asset_classes: List[AssetClass]) -> Dict[str, PredictiveModel]:
        """Generate predictive models for asset classes"""
        try:
            predictions = {}
            
            for asset_class in asset_classes:
                # Get training data
                price_data = await self._get_price_data(asset_class, "2Y")
                features = await self._extract_features(asset_class, price_data)
                
                # Train model
                model_result = await self._train_prediction_model(
                    asset_class, price_data, features
                )
                
                predictions[asset_class.value] = model_result
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            return {}
    
    async def _generate_market_insights(
        self,
        sentiment: SentimentAnalysis,
        trends: Dict[str, TrendAnalysis],
        volatility: Dict[str, VolatilityAnalysis],
        correlations: CorrelationAnalysis
    ) -> List[MarketInsight]:
        """Generate actionable market insights"""
        try:
            insights = []
            
            # Sentiment-based insights
            if sentiment.sentiment_score < -0.5:
                insights.append(MarketInsight(
                    insight_id=f"SENT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    category="sentiment",
                    priority="high",
                    title="Negative Market Sentiment Detected",
                    description=f"Market sentiment is {sentiment.overall_sentiment.value} with score {sentiment.sentiment_score:.2f}",
                    affected_assets=["all"],
                    recommendation="Consider defensive positioning and wait for sentiment improvement",
                    confidence=sentiment.confidence,
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(hours=24)
                ))
            
            # Trend-based insights
            bullish_trends = [asset for asset, trend in trends.items() 
                            if trend.direction == TrendDirection.BULLISH and trend.strength > 0.7]
            
            if bullish_trends:
                insights.append(MarketInsight(
                    insight_id=f"TREND_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    category="trend",
                    priority="medium",
                    title="Strong Bullish Trends Identified",
                    description=f"Strong uptrends detected in {', '.join(bullish_trends)}",
                    affected_assets=bullish_trends,
                    recommendation="Consider increasing allocation to trending assets",
                    confidence=0.8,
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(days=7)
                ))
            
            # Volatility-based insights
            high_vol_assets = [asset for asset, vol in volatility.items()
                             if vol.volatility_regime == "extreme"]
            
            if high_vol_assets:
                insights.append(MarketInsight(
                    insight_id=f"VOL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    category="volatility",
                    priority="high",
                    title="Extreme Volatility Warning",
                    description=f"Extreme volatility detected in {', '.join(high_vol_assets)}",
                    affected_assets=high_vol_assets,
                    recommendation="Reduce position sizes and implement volatility management strategies",
                    confidence=0.9,
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(hours=12)
                ))
            
            # Correlation-based insights
            if correlations.correlation_regime == "crisis":
                insights.append(MarketInsight(
                    insight_id=f"CORR_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    category="correlation",
                    priority="critical",
                    title="Correlation Breakdown Detected",
                    description="Asset correlations have increased significantly, reducing diversification benefits",
                    affected_assets=["all"],
                    recommendation="Review portfolio diversification and consider alternative assets",
                    confidence=0.85,
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(days=3)
                ))
            
            return insights
            
        except Exception as e:
            logger.error(f"Market insights generation failed: {e}")
            return []
    
    async def _identify_risk_alerts(
        self,
        trends: Dict[str, TrendAnalysis],
        volatility: Dict[str, VolatilityAnalysis],
        correlations: CorrelationAnalysis
    ) -> List[str]:
        """Identify risk alerts"""
        try:
            alerts = []
            
            # Trend reversal risks
            reversal_risks = [asset for asset, trend in trends.items()
                            if trend.reversal_probability > 0.7]
            if reversal_risks:
                alerts.append(f"High trend reversal probability in {', '.join(reversal_risks)}")
            
            # Volatility spikes
            vol_spikes = [asset for asset, vol in volatility.items()
                         if vol.volatility_percentile > 95]
            if vol_spikes:
                alerts.append(f"Volatility spike alert for {', '.join(vol_spikes)}")
            
            # Correlation regime changes
            if correlations.correlation_regime in ["crisis", "stress"]:
                alerts.append("Correlation regime change: diversification benefits reduced")
            
            return alerts
            
        except Exception as e:
            logger.error(f"Risk alerts identification failed: {e}")
            return ["Risk assessment unavailable"]
    
    async def _identify_opportunities(
        self,
        sentiment: SentimentAnalysis,
        trends: Dict[str, TrendAnalysis],
        predictions: Dict[str, PredictiveModel]
    ) -> List[str]:
        """Identify market opportunities"""
        try:
            opportunities = []
            
            # Contrarian opportunities
            if sentiment.sentiment_score < -0.4:
                opportunities.append("Contrarian opportunity: Oversold market conditions may present buying opportunities")
            
            # Trend continuation opportunities
            strong_trends = [asset for asset, trend in trends.items()
                           if trend.strength > 0.8 and trend.reversal_probability < 0.3]
            if strong_trends:
                opportunities.append(f"Trend continuation opportunity in {', '.join(strong_trends)}")
            
            # Prediction-based opportunities
            for asset, model in predictions.items():
                if "1M" in model.predictions and model.predictions["1M"] > 0.05:  # 5% predicted gain
                    if model.model_accuracy > 0.7:
                        opportunities.append(f"Predictive model suggests upside potential in {asset}")
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Opportunities identification failed: {e}")
            return ["Opportunity analysis unavailable"]
    
    # Helper methods for various calculations
    
    async def _determine_market_regime(self) -> str:
        """Determine overall market regime"""
        try:
            # Simplified regime detection
            # In production, would use more sophisticated models
            return MarketRegime.CONSOLIDATION.value
        except Exception:
            return MarketRegime.CONSOLIDATION.value
    
    async def _assess_risk_sentiment(self) -> str:
        """Assess risk-on vs risk-off sentiment"""
        try:
            # Simplified risk assessment
            return "neutral"
        except Exception:
            return "neutral"
    
    async def _get_key_indicators(self) -> Dict[str, MarketIndicator]:
        """Get key market indicators"""
        try:
            # Simulate key indicators
            return {
                "vix": MarketIndicator("VIX", 18.5, 20.1, -8.0, datetime.now(), "CBOE", 0.95),
                "10y_yield": MarketIndicator("10Y Treasury", 4.2, 4.1, 2.4, datetime.now(), "Federal Reserve", 0.98),
                "dxy": MarketIndicator("Dollar Index", 103.2, 102.8, 0.4, datetime.now(), "ICE", 0.92)
            }
        except Exception:
            return {}
    
    async def _calculate_market_performance(self, asset_classes: List[AssetClass]) -> Dict[str, float]:
        """Calculate market performance metrics"""
        try:
            performance = {}
            for asset_class in asset_classes:
                # Simulate performance data
                performance[asset_class.value] = np.random.normal(0.05, 0.15)  # 5% return, 15% vol
            return performance
        except Exception:
            return {}
    
    async def _analyze_economic_backdrop(self) -> Dict[str, Any]:
        """Analyze economic backdrop"""
        try:
            return {
                "gdp_growth": 2.1,
                "inflation": 3.2,
                "unemployment": 3.7,
                "central_bank_policy": "neutral"
            }
        except Exception:
            return {}
    
    async def _analyze_news_sentiment(self, asset_classes: List[AssetClass]) -> float:
        """Analyze news sentiment"""
        try:
            # Simulate news sentiment analysis
            # In production, would use NLP models on news feeds
            return np.random.normal(0, 0.3)
        except Exception:
            return 0.0
    
    async def _analyze_social_sentiment(self, asset_classes: List[AssetClass]) -> float:
        """Analyze social media sentiment"""
        try:
            # Simulate social sentiment analysis
            return np.random.normal(0, 0.4)
        except Exception:
            return 0.0
    
    async def _analyze_analyst_sentiment(self, asset_classes: List[AssetClass]) -> float:
        """Analyze analyst sentiment"""
        try:
            # Simulate analyst sentiment
            return np.random.normal(0.1, 0.2)
        except Exception:
            return 0.0
    
    async def _identify_sentiment_drivers(
        self, 
        news_sentiment: float, 
        social_sentiment: float, 
        analyst_sentiment: float
    ) -> List[str]:
        """Identify key sentiment drivers"""
        try:
            drivers = []
            
            if abs(news_sentiment) > 0.3:
                drivers.append("News flow impact")
            if abs(social_sentiment) > 0.3:
                drivers.append("Social media sentiment")
            if abs(analyst_sentiment) > 0.2:
                drivers.append("Analyst recommendations")
            
            if not drivers:
                drivers.append("Stable sentiment environment")
            
            return drivers
        except Exception:
            return ["Sentiment analysis unavailable"]
    
    async def _get_price_data(self, asset_class: AssetClass, time_horizon: str) -> np.ndarray:
        """Get price data for asset class"""
        try:
            # Simulate price data
            # In production, would fetch real market data
            days = {"1D": 1, "1W": 7, "1M": 30, "3M": 90, "1Y": 365, "2Y": 730}
            n_days = days.get(time_horizon, 30)
            
            # Generate realistic price series
            returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
            prices = 100 * np.exp(np.cumsum(returns))  # Price series starting at 100
            
            return prices
        except Exception:
            return np.array([100.0])  # Fallback
    
    async def _calculate_trend_direction(self, price_data: np.ndarray) -> TrendDirection:
        """Calculate trend direction"""
        try:
            if len(price_data) < 5:
                return TrendDirection.SIDEWAYS
            
            # Simple trend calculation
            start_price = np.mean(price_data[:5])
            end_price = np.mean(price_data[-5:])
            change = (end_price - start_price) / start_price
            
            if change > 0.05:
                return TrendDirection.BULLISH
            elif change < -0.05:
                return TrendDirection.BEARISH
            else:
                return TrendDirection.SIDEWAYS
        except Exception:
            return TrendDirection.SIDEWAYS
    
    async def _calculate_trend_strength(self, price_data: np.ndarray) -> float:
        """Calculate trend strength"""
        try:
            if len(price_data) < 10:
                return 0.5
            
            # Calculate R-squared of price vs time
            x = np.arange(len(price_data))
            y = price_data
            correlation = np.corrcoef(x, y)[0, 1]
            r_squared = correlation ** 2
            
            return min(abs(r_squared), 1.0)
        except Exception:
            return 0.5
    
    async def _calculate_trend_duration(self, price_data: np.ndarray) -> int:
        """Calculate trend duration in days"""
        try:
            # Simplified duration calculation
            return len(price_data)
        except Exception:
            return 0
    
    async def _calculate_support_resistance(self, price_data: np.ndarray) -> Tuple[float, float]:
        """Calculate support and resistance levels"""
        try:
            if len(price_data) < 10:
                return None, None
            
            # Simple support/resistance calculation
            support = np.percentile(price_data, 10)
            resistance = np.percentile(price_data, 90)
            
            return support, resistance
        except Exception:
            return None, None
    
    async def _calculate_momentum_indicators(self, price_data: np.ndarray) -> Dict[str, float]:
        """Calculate momentum indicators"""
        try:
            if len(price_data) < 14:
                return {"rsi": 50.0, "momentum": 0.0}
            
            # Simple RSI calculation
            returns = np.diff(price_data)
            gains = returns[returns > 0]
            losses = -returns[returns < 0]
            
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            # Momentum
            momentum = (price_data[-1] - price_data[-10]) / price_data[-10] if len(price_data) >= 10 else 0
            
            return {"rsi": rsi, "momentum": momentum}
        except Exception:
            return {"rsi": 50.0, "momentum": 0.0}
    
    async def _calculate_reversal_probability(
        self, 
        price_data: np.ndarray, 
        direction: TrendDirection, 
        strength: float
    ) -> float:
        """Calculate trend reversal probability"""
        try:
            # Simplified reversal probability
            # Higher probability for stronger trends (due to mean reversion)
            base_prob = 0.3
            strength_factor = strength * 0.4
            
            return min(base_prob + strength_factor, 0.9)
        except Exception:
            return 0.5
    
    async def _calculate_current_volatility(self, price_data: np.ndarray) -> float:
        """Calculate current volatility"""
        try:
            if len(price_data) < 2:
                return 0.2
            
            returns = np.diff(np.log(price_data))
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            return volatility
        except Exception:
            return 0.2
    
    async def _calculate_historical_volatility(self, price_data: np.ndarray) -> float:
        """Calculate historical volatility"""
        try:
            # Same as current for simplicity
            return await self._calculate_current_volatility(price_data)
        except Exception:
            return 0.2
    
    async def _calculate_volatility_percentile(self, current_vol: float, price_data: np.ndarray) -> float:
        """Calculate volatility percentile"""
        try:
            # Simulate historical volatility distribution
            historical_vols = np.random.lognormal(np.log(0.15), 0.5, 252)
            percentile = stats.percentileofscore(historical_vols, current_vol)
            
            return percentile
        except Exception:
            return 50.0
    
    async def _determine_volatility_regime(self, percentile: float) -> str:
        """Determine volatility regime"""
        try:
            if percentile < 25:
                return "low"
            elif percentile < 75:
                return "normal"
            elif percentile < 95:
                return "high"
            else:
                return "extreme"
        except Exception:
            return "normal"
    
    async def _forecast_volatility(self, price_data: np.ndarray) -> Dict[str, float]:
        """Forecast volatility"""
        try:
            current_vol = await self._calculate_current_volatility(price_data)
            
            # Simple volatility forecast (mean reversion)
            long_term_vol = 0.20
            
            return {
                "1d": current_vol,
                "7d": current_vol * 0.9 + long_term_vol * 0.1,
                "30d": current_vol * 0.7 + long_term_vol * 0.3
            }
        except Exception:
            return {"1d": 0.2, "7d": 0.2, "30d": 0.2}
    
    async def _calculate_correlation_matrix(self, returns_data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix"""
        try:
            assets = list(returns_data.keys())
            corr_matrix = {}
            
            for asset1 in assets:
                corr_matrix[asset1] = {}
                for asset2 in assets:
                    if asset1 == asset2:
                        corr = 1.0
                    else:
                        returns1 = returns_data[asset1]
                        returns2 = returns_data[asset2]
                        
                        min_len = min(len(returns1), len(returns2))
                        if min_len > 1:
                            corr = np.corrcoef(returns1[-min_len:], returns2[-min_len:])[0, 1]
                            if np.isnan(corr):
                                corr = 0.0
                        else:
                            corr = 0.0
                    
                    corr_matrix[asset1][asset2] = corr
            
            return corr_matrix
        except Exception:
            return {}
    
    async def _calculate_correlation_changes(self, returns_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate correlation changes"""
        try:
            # Simplified correlation change calculation
            return {"average_correlation_change": 0.05}
        except Exception:
            return {}
    
    async def _calculate_diversification_benefits(self, correlation_matrix: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate diversification benefits"""
        try:
            if not correlation_matrix:
                return {}
            
            # Calculate average correlation
            all_corrs = []
            assets = list(correlation_matrix.keys())
            
            for i, asset1 in enumerate(assets):
                for j, asset2 in enumerate(assets):
                    if i < j:  # Avoid double counting and self-correlation
                        corr = correlation_matrix[asset1][asset2]
                        if not np.isnan(corr):
                            all_corrs.append(corr)
            
            avg_corr = np.mean(all_corrs) if all_corrs else 0.5
            diversification_ratio = 1 - avg_corr
            
            return {"diversification_ratio": diversification_ratio}
        except Exception:
            return {"diversification_ratio": 0.5}
    
    async def _identify_risk_concentrations(self, correlation_matrix: Dict[str, Dict[str, float]]) -> List[str]:
        """Identify risk concentrations"""
        try:
            concentrations = []
            
            if correlation_matrix:
                assets = list(correlation_matrix.keys())
                high_corr_pairs = []
                
                for i, asset1 in enumerate(assets):
                    for j, asset2 in enumerate(assets):
                        if i < j:
                            corr = correlation_matrix[asset1][asset2]
                            if corr > 0.8:
                                high_corr_pairs.append(f"{asset1}-{asset2}")
                
                if high_corr_pairs:
                    concentrations.append(f"High correlation pairs: {', '.join(high_corr_pairs)}")
            
            return concentrations
        except Exception:
            return []
    
    async def _determine_correlation_regime(self, correlation_matrix: Dict[str, Dict[str, float]]) -> str:
        """Determine correlation regime"""
        try:
            if not correlation_matrix:
                return "normal"
            
            # Calculate average correlation
            all_corrs = []
            assets = list(correlation_matrix.keys())
            
            for i, asset1 in enumerate(assets):
                for j, asset2 in enumerate(assets):
                    if i < j:
                        corr = correlation_matrix[asset1][asset2]
                        if not np.isnan(corr):
                            all_corrs.append(corr)
            
            avg_corr = np.mean(all_corrs) if all_corrs else 0.5
            
            if avg_corr > 0.8:
                return "crisis"
            elif avg_corr > 0.6:
                return "stress"
            elif avg_corr < 0.2:
                return "decoupled"
            else:
                return "normal"
        except Exception:
            return "normal"
    
    async def _extract_features(self, asset_class: AssetClass, price_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features for prediction models"""
        try:
            features = {}
            
            if len(price_data) > 1:
                # Price-based features
                returns = np.diff(np.log(price_data))
                features["returns"] = returns
                features["volatility"] = np.std(returns)
                features["momentum"] = returns[-5:].mean() if len(returns) >= 5 else 0
                
                # Technical indicators
                if len(price_data) >= 20:
                    sma_20 = np.convolve(price_data, np.ones(20)/20, mode='valid')
                    features["sma_signal"] = price_data[-1] / sma_20[-1] - 1 if len(sma_20) > 0 else 0
            
            return features
        except Exception:
            return {}
    
    async def _train_prediction_model(
        self, 
        asset_class: AssetClass, 
        price_data: np.ndarray, 
        features: Dict[str, Any]
    ) -> PredictiveModel:
        """Train prediction model"""
        try:
            # Simplified prediction model
            # In production, would use sophisticated ML models
            
            # Generate predictions
            current_price = price_data[-1] if len(price_data) > 0 else 100
            predictions = {
                "1D": current_price * (1 + np.random.normal(0, 0.01)),
                "1W": current_price * (1 + np.random.normal(0, 0.03)),
                "1M": current_price * (1 + np.random.normal(0, 0.05))
            }
            
            # Confidence intervals
            confidence_intervals = {}
            for timeframe, pred in predictions.items():
                margin = pred * 0.1  # 10% margin
                confidence_intervals[timeframe] = (pred - margin, pred + margin)
            
            # Feature importance
            feature_importance = {
                "momentum": 0.3,
                "volatility": 0.2,
                "sma_signal": 0.25,
                "returns": 0.25
            }
            
            # Risk scenarios
            risk_scenarios = {
                "bull_case": predictions["1M"] * 1.2,
                "base_case": predictions["1M"],
                "bear_case": predictions["1M"] * 0.8
            }
            
            return PredictiveModel(
                model_type="ensemble",
                predictions=predictions,
                confidence_intervals=confidence_intervals,
                feature_importance=feature_importance,
                model_accuracy=0.75,
                risk_scenarios=risk_scenarios
            )
            
        except Exception as e:
            logger.error(f"Prediction model training failed: {e}")
            return PredictiveModel(
                model_type="fallback",
                predictions={},
                confidence_intervals={},
                feature_importance={},
                model_accuracy=0.5,
                risk_scenarios={}
            )
    
    async def _update_market_data(self):
        """Update cached market data"""
        try:
            # In production, would fetch real market data
            self.last_update = datetime.now()
            logger.info("Market data updated successfully")
        except Exception as e:
            logger.error(f"Market data update failed: {e}")
    
    async def _initialize_models(self):
        """Initialize prediction models"""
        try:
            # In production, would load pre-trained models
            logger.info("Prediction models initialized")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
    
    async def _ensure_fresh_data(self):
        """Ensure market data is fresh"""
        if not self.last_update or (datetime.now() - self.last_update).seconds > 3600:
            await self._update_market_data()
    
    async def _assess_data_quality(self) -> float:
        """Assess overall data quality"""
        try:
            # In production, would assess real data quality metrics
            return 0.9
        except Exception:
            return 0.5


# Export the service
__all__ = [
    'MarketAnalyticsService', 'MarketAnalyticsReport', 'SentimentAnalysis',
    'TrendAnalysis', 'VolatilityAnalysis', 'CorrelationAnalysis', 'PredictiveModel',
    'MarketInsight', 'TrendDirection', 'SentimentScore', 'MarketRegime', 'AssetClass'
] 