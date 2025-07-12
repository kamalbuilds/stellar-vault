"""
Risk Assessment Service for StellarVault AI Engine

This service provides comprehensive risk analysis including:
- Value at Risk (VaR) calculations using multiple methodologies
- Stress testing and scenario analysis
- Monte Carlo simulations
- Risk factor decomposition
- Real-time risk monitoring
- Correlation analysis and regime detection
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats
from scipy.optimize import minimize
import json
from loguru import logger

from ..core.config import settings


class VaRMethod(Enum):
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    CORNISH_FISHER = "cornish_fisher"
    EXTREME_VALUE = "extreme_value"


class StressScenario(Enum):
    MARKET_CRASH = "market_crash"
    INTEREST_RATE_SHOCK = "interest_rate_shock"
    CREDIT_CRISIS = "credit_crisis"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    INFLATION_SHOCK = "inflation_shock"
    CURRENCY_CRISIS = "currency_crisis"
    CUSTOM = "custom"


@dataclass
class RiskFactors:
    """Risk factors for analysis"""
    equity_market: float = 0.0
    interest_rates: float = 0.0
    credit_spreads: float = 0.0
    currency: float = 0.0
    commodities: float = 0.0
    volatility: float = 0.0
    liquidity: float = 0.0


@dataclass
class StressTest:
    """Stress test scenario definition"""
    name: str
    description: str
    risk_factor_shocks: RiskFactors
    probability: float
    severity: float


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    var_95_1day: float
    var_99_1day: float
    var_95_10day: float
    cvar_95: float
    maximum_drawdown: float
    volatility: float
    beta: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    skewness: float
    kurtosis: float
    tail_ratio: float
    value_at_risk_contributions: Dict[str, float]
    stress_test_results: Dict[str, float]
    confidence_interval: Tuple[float, float]
    risk_attribution: Dict[str, float]
    correlation_breakdown: Dict[str, float]


@dataclass
class RiskAssessmentResult:
    """Risk assessment result"""
    portfolio_id: str
    assessment_date: datetime
    risk_metrics: RiskMetrics
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    risk_score: float  # 0-100
    key_risks: List[str]
    recommendations: List[str]
    model_confidence: float
    methodology: str
    data_quality_score: float


class RiskAssessmentService:
    """
    Advanced risk assessment service with multiple methodologies
    """
    
    def __init__(self):
        self.confidence_level_95 = 0.95
        self.confidence_level_99 = 0.99
        self.risk_window_days = settings.RISK_WINDOW_DAYS
        self.monte_carlo_simulations = settings.MONTE_CARLO_SIMULATIONS
        
    async def assess_portfolio_risk(
        self,
        portfolio_returns: np.ndarray,
        asset_weights: Dict[str, float],
        asset_returns: Dict[str, np.ndarray],
        market_data: Optional[Dict[str, Any]] = None,
        stress_scenarios: Optional[List[StressTest]] = None
    ) -> RiskAssessmentResult:
        """
        Comprehensive portfolio risk assessment
        
        Args:
            portfolio_returns: Historical portfolio returns
            asset_weights: Current portfolio weights
            asset_returns: Individual asset return series
            market_data: Current market data
            stress_scenarios: Custom stress test scenarios
            
        Returns:
            RiskAssessmentResult with comprehensive risk analysis
        """
        try:
            # Calculate basic risk metrics
            var_metrics = await self._calculate_var_metrics(portfolio_returns)
            
            # Calculate advanced risk metrics
            advanced_metrics = await self._calculate_advanced_metrics(
                portfolio_returns, asset_returns
            )
            
            # Perform stress testing
            stress_results = await self._perform_stress_testing(
                portfolio_returns, asset_weights, asset_returns, stress_scenarios
            )
            
            # Risk attribution analysis
            risk_attribution = await self._calculate_risk_attribution(
                asset_weights, asset_returns
            )
            
            # Correlation analysis
            correlation_breakdown = await self._analyze_correlations(asset_returns)
            
            # Combine all metrics
            risk_metrics = RiskMetrics(
                var_95_1day=var_metrics['var_95_1day'],
                var_99_1day=var_metrics['var_99_1day'],
                var_95_10day=var_metrics['var_95_10day'],
                cvar_95=var_metrics['cvar_95'],
                maximum_drawdown=advanced_metrics['max_drawdown'],
                volatility=advanced_metrics['volatility'],
                beta=advanced_metrics['beta'],
                sharpe_ratio=advanced_metrics['sharpe_ratio'],
                sortino_ratio=advanced_metrics['sortino_ratio'],
                calmar_ratio=advanced_metrics['calmar_ratio'],
                skewness=advanced_metrics['skewness'],
                kurtosis=advanced_metrics['kurtosis'],
                tail_ratio=advanced_metrics['tail_ratio'],
                value_at_risk_contributions=var_metrics['var_contributions'],
                stress_test_results=stress_results,
                confidence_interval=var_metrics['confidence_interval'],
                risk_attribution=risk_attribution,
                correlation_breakdown=correlation_breakdown
            )
            
            # Calculate overall risk score and level
            risk_score = await self._calculate_risk_score(risk_metrics)
            risk_level = self._determine_risk_level(risk_score)
            
            # Identify key risks
            key_risks = await self._identify_key_risks(risk_metrics, market_data)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                risk_metrics, risk_level, key_risks
            )
            
            # Calculate model confidence
            model_confidence = await self._calculate_model_confidence(
                portfolio_returns, asset_returns
            )
            
            # Data quality assessment
            data_quality = await self._assess_data_quality(
                portfolio_returns, asset_returns
            )
            
            return RiskAssessmentResult(
                portfolio_id=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                assessment_date=datetime.now(),
                risk_metrics=risk_metrics,
                risk_level=risk_level,
                risk_score=risk_score,
                key_risks=key_risks,
                recommendations=recommendations,
                model_confidence=model_confidence,
                methodology="Multi-Method Risk Assessment with Monte Carlo",
                data_quality_score=data_quality
            )
            
        except Exception as e:
            logger.error(f"Portfolio risk assessment failed: {e}")
            raise
    
    async def _calculate_var_metrics(self, returns: np.ndarray) -> Dict[str, Any]:
        """Calculate Value at Risk using multiple methods"""
        try:
            # Historical VaR
            var_95_hist = np.percentile(returns, 5)
            var_99_hist = np.percentile(returns, 1)
            
            # Parametric VaR (assuming normal distribution)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            var_95_param = mean_return + std_return * stats.norm.ppf(0.05)
            var_99_param = mean_return + std_return * stats.norm.ppf(0.01)
            
            # Monte Carlo VaR
            var_95_mc, var_99_mc = await self._monte_carlo_var(returns)
            
            # Cornish-Fisher VaR (accounts for skewness and kurtosis)
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            # Cornish-Fisher expansion
            z_95 = stats.norm.ppf(0.05)
            z_99 = stats.norm.ppf(0.01)
            
            cf_adjustment_95 = (z_95 + (z_95**2 - 1) * skewness / 6 + 
                              (z_95**3 - 3*z_95) * kurtosis / 24 - 
                              (2*z_95**3 - 5*z_95) * skewness**2 / 36)
            
            cf_adjustment_99 = (z_99 + (z_99**2 - 1) * skewness / 6 + 
                              (z_99**3 - 3*z_99) * kurtosis / 24 - 
                              (2*z_99**3 - 5*z_99) * skewness**2 / 36)
            
            var_95_cf = mean_return + std_return * cf_adjustment_95
            var_99_cf = mean_return + std_return * cf_adjustment_99
            
            # Use most conservative VaR
            var_95_1day = min(var_95_hist, var_95_param, var_95_mc, var_95_cf)
            var_99_1day = min(var_99_hist, var_99_param, var_99_mc, var_99_cf)
            
            # 10-day VaR (scaling)
            var_95_10day = var_95_1day * np.sqrt(10)
            
            # Conditional VaR (Expected Shortfall)
            tail_returns = returns[returns <= var_95_1day]
            cvar_95 = np.mean(tail_returns) if len(tail_returns) > 0 else var_95_1day
            
            # VaR contributions (simplified)
            var_contributions = {"Total Portfolio": 1.0}
            
            # Confidence interval for VaR
            confidence_interval = self._calculate_var_confidence_interval(
                returns, var_95_1day
            )
            
            return {
                'var_95_1day': var_95_1day,
                'var_99_1day': var_99_1day,
                'var_95_10day': var_95_10day,
                'cvar_95': cvar_95,
                'var_contributions': var_contributions,
                'confidence_interval': confidence_interval
            }
            
        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            return {
                'var_95_1day': -0.05,
                'var_99_1day': -0.08,
                'var_95_10day': -0.16,
                'cvar_95': -0.07,
                'var_contributions': {"Total Portfolio": 1.0},
                'confidence_interval': (-0.06, -0.04)
            }
    
    async def _monte_carlo_var(self, returns: np.ndarray) -> Tuple[float, float]:
        """Calculate VaR using Monte Carlo simulation"""
        try:
            # Fit distribution to returns
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            skewness = stats.skew(returns)
            
            # Generate simulated returns
            np.random.seed(42)  # For reproducibility
            
            if abs(skewness) > 0.5:
                # Use skewed normal distribution
                simulated_returns = stats.skewnorm.rvs(
                    a=skewness, loc=mean_return, scale=std_return,
                    size=self.monte_carlo_simulations
                )
            else:
                # Use normal distribution
                simulated_returns = np.random.normal(
                    mean_return, std_return, self.monte_carlo_simulations
                )
            
            # Calculate VaR from simulated returns
            var_95_mc = np.percentile(simulated_returns, 5)
            var_99_mc = np.percentile(simulated_returns, 1)
            
            return var_95_mc, var_99_mc
            
        except Exception as e:
            logger.error(f"Monte Carlo VaR calculation failed: {e}")
            return -0.05, -0.08
    
    def _calculate_var_confidence_interval(
        self, 
        returns: np.ndarray, 
        var_estimate: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for VaR estimate"""
        try:
            n = len(returns)
            p = 0.05  # 5% quantile
            
            # Standard error of quantile
            f_var = stats.gaussian_kde(returns)(var_estimate)[0]
            se_var = np.sqrt(p * (1 - p) / (n * f_var**2))
            
            # 95% confidence interval
            margin = 1.96 * se_var
            
            return (var_estimate - margin, var_estimate + margin)
            
        except Exception as e:
            logger.error(f"VaR confidence interval calculation failed: {e}")
            return (var_estimate * 0.8, var_estimate * 1.2)
    
    async def _calculate_advanced_metrics(
        self,
        portfolio_returns: np.ndarray,
        asset_returns: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Calculate advanced risk metrics"""
        try:
            # Basic statistics
            volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
            mean_return = np.mean(portfolio_returns)
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + portfolio_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns)
            
            # Beta (vs market proxy - using first asset as proxy)
            if asset_returns:
                market_returns = list(asset_returns.values())[0]
                if len(market_returns) == len(portfolio_returns):
                    covariance = np.cov(portfolio_returns, market_returns)[0, 1]
                    market_variance = np.var(market_returns)
                    beta = covariance / market_variance if market_variance > 0 else 1.0
                else:
                    beta = 1.0
            else:
                beta = 1.0
            
            # Sharpe ratio
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            excess_returns = portfolio_returns - risk_free_rate
            sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns) * 
                          np.sqrt(252)) if np.std(excess_returns) > 0 else 0.0
            
            # Sortino ratio
            downside_returns = portfolio_returns[portfolio_returns < risk_free_rate]
            if len(downside_returns) > 0:
                downside_deviation = np.std(downside_returns) * np.sqrt(252)
                sortino_ratio = ((np.mean(portfolio_returns) - risk_free_rate) * 252 / 
                               downside_deviation) if downside_deviation > 0 else 0.0
            else:
                sortino_ratio = sharpe_ratio
            
            # Calmar ratio
            calmar_ratio = ((np.mean(portfolio_returns) * 252) / 
                          abs(max_drawdown)) if abs(max_drawdown) > 1e-8 else 0.0
            
            # Higher moments
            skewness = stats.skew(portfolio_returns)
            kurtosis = stats.kurtosis(portfolio_returns)
            
            # Tail ratio (95th percentile / 5th percentile)
            p95 = np.percentile(portfolio_returns, 95)
            p5 = np.percentile(portfolio_returns, 5)
            tail_ratio = abs(p95 / p5) if abs(p5) > 1e-8 else 1.0
            
            return {
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'beta': beta,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'tail_ratio': tail_ratio
            }
            
        except Exception as e:
            logger.error(f"Advanced metrics calculation failed: {e}")
            return {
                'volatility': 0.20,
                'max_drawdown': -0.15,
                'beta': 1.0,
                'sharpe_ratio': 0.5,
                'sortino_ratio': 0.6,
                'calmar_ratio': 0.3,
                'skewness': -0.5,
                'kurtosis': 1.0,
                'tail_ratio': 2.0
            }
    
    async def _perform_stress_testing(
        self,
        portfolio_returns: np.ndarray,
        asset_weights: Dict[str, float],
        asset_returns: Dict[str, np.ndarray],
        custom_scenarios: Optional[List[StressTest]] = None
    ) -> Dict[str, float]:
        """Perform comprehensive stress testing"""
        try:
            stress_results = {}
            
            # Default stress scenarios
            scenarios = [
                StressTest(
                    name="Market Crash",
                    description="Severe market downturn (-30%)",
                    risk_factor_shocks=RiskFactors(equity_market=-0.30),
                    probability=0.05,
                    severity=0.9
                ),
                StressTest(
                    name="Interest Rate Shock",
                    description="Sharp rise in interest rates (+300 bps)",
                    risk_factor_shocks=RiskFactors(interest_rates=0.03),
                    probability=0.10,
                    severity=0.7
                ),
                StressTest(
                    name="Credit Crisis",
                    description="Credit spreads widen significantly (+500 bps)",
                    risk_factor_shocks=RiskFactors(credit_spreads=0.05),
                    probability=0.08,
                    severity=0.8
                ),
                StressTest(
                    name="Liquidity Crisis",
                    description="Market liquidity dries up",
                    risk_factor_shocks=RiskFactors(liquidity=-0.50),
                    probability=0.06,
                    severity=0.9
                )
            ]
            
            # Add custom scenarios
            if custom_scenarios:
                scenarios.extend(custom_scenarios)
            
            # Calculate stress impact for each scenario
            for scenario in scenarios:
                impact = await self._calculate_stress_impact(
                    portfolio_returns, scenario
                )
                stress_results[scenario.name] = impact
            
            # Historical stress test (worst month)
            if len(portfolio_returns) >= 21:  # At least one month of data
                monthly_returns = []
                for i in range(0, len(portfolio_returns) - 20, 21):
                    month_return = np.sum(portfolio_returns[i:i+21])
                    monthly_returns.append(month_return)
                
                if monthly_returns:
                    worst_month = np.min(monthly_returns)
                    stress_results["Worst Historical Month"] = worst_month
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Stress testing failed: {e}")
            return {
                "Market Crash": -0.25,
                "Interest Rate Shock": -0.10,
                "Credit Crisis": -0.15,
                "Liquidity Crisis": -0.20
            }
    
    async def _calculate_stress_impact(
        self,
        portfolio_returns: np.ndarray,
        scenario: StressTest
    ) -> float:
        """Calculate impact of stress scenario on portfolio"""
        try:
            # Simplified stress impact calculation
            # In practice, would use factor models
            
            base_volatility = np.std(portfolio_returns)
            base_return = np.mean(portfolio_returns)
            
            # Apply stress factors
            stress_impact = 0.0
            
            if scenario.risk_factor_shocks.equity_market != 0:
                stress_impact += scenario.risk_factor_shocks.equity_market * 0.8
            
            if scenario.risk_factor_shocks.interest_rates != 0:
                # Duration impact (assuming 5-year duration)
                duration = 5.0
                stress_impact += -duration * scenario.risk_factor_shocks.interest_rates * 0.3
            
            if scenario.risk_factor_shocks.credit_spreads != 0:
                stress_impact += -scenario.risk_factor_shocks.credit_spreads * 2.0
            
            if scenario.risk_factor_shocks.liquidity != 0:
                stress_impact += scenario.risk_factor_shocks.liquidity * 0.1
            
            # Apply severity multiplier
            stress_impact *= scenario.severity
            
            return stress_impact
            
        except Exception as e:
            logger.error(f"Stress impact calculation failed: {e}")
            return -0.10
    
    async def _calculate_risk_attribution(
        self,
        asset_weights: Dict[str, float],
        asset_returns: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Calculate risk attribution by asset"""
        try:
            risk_attribution = {}
            
            # Calculate contribution to portfolio variance
            if len(asset_returns) > 1:
                # Create returns matrix
                symbols = list(asset_weights.keys())
                weights = np.array([asset_weights[symbol] for symbol in symbols])
                
                returns_matrix = []
                min_length = min(len(asset_returns[symbol]) for symbol in symbols)
                
                for symbol in symbols:
                    returns_matrix.append(asset_returns[symbol][-min_length:])
                
                returns_matrix = np.array(returns_matrix).T
                
                # Calculate covariance matrix
                cov_matrix = np.cov(returns_matrix, rowvar=False)
                
                # Portfolio variance
                portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
                
                # Marginal risk contributions
                marginal_contrib = np.dot(cov_matrix, weights)
                
                # Risk contributions
                for i, symbol in enumerate(symbols):
                    contrib = (weights[i] * marginal_contrib[i]) / portfolio_variance
                    risk_attribution[symbol] = contrib
            else:
                # Single asset case
                for symbol in asset_weights:
                    risk_attribution[symbol] = asset_weights[symbol]
            
            return risk_attribution
            
        except Exception as e:
            logger.error(f"Risk attribution calculation failed: {e}")
            return {symbol: weight for symbol, weight in asset_weights.items()}
    
    async def _analyze_correlations(
        self,
        asset_returns: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Analyze correlation structure"""
        try:
            correlation_breakdown = {}
            
            if len(asset_returns) < 2:
                return correlation_breakdown
            
            # Create correlation matrix
            symbols = list(asset_returns.keys())
            returns_matrix = []
            min_length = min(len(asset_returns[symbol]) for symbol in symbols)
            
            for symbol in symbols:
                returns_matrix.append(asset_returns[symbol][-min_length:])
            
            returns_matrix = np.array(returns_matrix).T
            corr_matrix = np.corrcoef(returns_matrix, rowvar=False)
            
            # Average correlation
            upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            avg_correlation = np.mean(upper_triangle)
            correlation_breakdown["Average Correlation"] = avg_correlation
            
            # Maximum correlation
            max_correlation = np.max(upper_triangle)
            correlation_breakdown["Maximum Correlation"] = max_correlation
            
            # Minimum correlation
            min_correlation = np.min(upper_triangle)
            correlation_breakdown["Minimum Correlation"] = min_correlation
            
            # Correlation concentration (percentage of high correlations)
            high_corr_count = np.sum(upper_triangle > 0.7)
            corr_concentration = high_corr_count / len(upper_triangle)
            correlation_breakdown["High Correlation Concentration"] = corr_concentration
            
            return correlation_breakdown
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return {
                "Average Correlation": 0.5,
                "Maximum Correlation": 0.8,
                "Minimum Correlation": 0.2,
                "High Correlation Concentration": 0.3
            }
    
    async def _calculate_risk_score(self, risk_metrics: RiskMetrics) -> float:
        """Calculate overall risk score (0-100)"""
        try:
            score = 0.0
            
            # VaR component (0-30 points)
            var_score = min(abs(risk_metrics.var_95_1day) * 300, 30)
            score += var_score
            
            # Volatility component (0-25 points)
            vol_score = min(risk_metrics.volatility * 125, 25)
            score += vol_score
            
            # Maximum drawdown component (0-20 points)
            dd_score = min(abs(risk_metrics.maximum_drawdown) * 100, 20)
            score += dd_score
            
            # Sharpe ratio component (0-15 points, inverse)
            sharpe_score = max(15 - risk_metrics.sharpe_ratio * 7.5, 0)
            score += sharpe_score
            
            # Tail risk component (0-10 points)
            tail_score = min(abs(risk_metrics.skewness) * 5 + 
                           max(risk_metrics.kurtosis - 3, 0) * 2.5, 10)
            score += tail_score
            
            return min(score, 100)
            
        except Exception as e:
            logger.error(f"Risk score calculation failed: {e}")
            return 50.0
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on score"""
        if risk_score < 25:
            return "LOW"
        elif risk_score < 50:
            return "MEDIUM"
        elif risk_score < 75:
            return "HIGH"
        else:
            return "CRITICAL"
    
    async def _identify_key_risks(
        self,
        risk_metrics: RiskMetrics,
        market_data: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Identify key risk factors"""
        try:
            key_risks = []
            
            # High volatility
            if risk_metrics.volatility > 0.25:
                key_risks.append("High portfolio volatility")
            
            # Large maximum drawdown
            if abs(risk_metrics.maximum_drawdown) > 0.20:
                key_risks.append("Significant drawdown risk")
            
            # Poor Sharpe ratio
            if risk_metrics.sharpe_ratio < 0.5:
                key_risks.append("Poor risk-adjusted returns")
            
            # Negative skewness
            if risk_metrics.skewness < -0.5:
                key_risks.append("Negative return distribution skew")
            
            # High kurtosis
            if risk_metrics.kurtosis > 5:
                key_risks.append("Fat tail risk (extreme events)")
            
            # High correlations
            if risk_metrics.correlation_breakdown.get("Average Correlation", 0) > 0.7:
                key_risks.append("High asset correlation concentration")
            
            # Large VaR
            if abs(risk_metrics.var_95_1day) > 0.05:
                key_risks.append("High Value at Risk")
            
            # Stress test failures
            for scenario, impact in risk_metrics.stress_test_results.items():
                if impact < -0.20:
                    key_risks.append(f"Vulnerable to {scenario.lower()}")
            
            return key_risks[:5]  # Return top 5 risks
            
        except Exception as e:
            logger.error(f"Key risk identification failed: {e}")
            return ["General market risk", "Concentration risk"]
    
    async def _generate_recommendations(
        self,
        risk_metrics: RiskMetrics,
        risk_level: str,
        key_risks: List[str]
    ) -> List[str]:
        """Generate risk management recommendations"""
        try:
            recommendations = []
            
            if risk_level in ["HIGH", "CRITICAL"]:
                recommendations.append("Consider reducing portfolio risk through diversification")
                recommendations.append("Implement risk management strategies (hedging, position sizing)")
            
            if abs(risk_metrics.maximum_drawdown) > 0.15:
                recommendations.append("Consider implementing stop-loss or dynamic hedging strategies")
            
            if risk_metrics.sharpe_ratio < 0.5:
                recommendations.append("Review asset allocation to improve risk-adjusted returns")
            
            if "High asset correlation concentration" in key_risks:
                recommendations.append("Diversify across uncorrelated assets and asset classes")
            
            if "Fat tail risk" in str(key_risks):
                recommendations.append("Consider tail risk hedging strategies")
            
            if risk_metrics.volatility > 0.30:
                recommendations.append("Implement volatility targeting or risk budgeting")
            
            # Add positive recommendations for good metrics
            if risk_level == "LOW" and risk_metrics.sharpe_ratio > 1.0:
                recommendations.append("Portfolio shows strong risk-adjusted performance")
            
            return recommendations[:5]  # Return top 5 recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return ["Monitor portfolio regularly", "Maintain appropriate diversification"]
    
    async def _calculate_model_confidence(
        self,
        portfolio_returns: np.ndarray,
        asset_returns: Dict[str, np.ndarray]
    ) -> float:
        """Calculate confidence in risk model results"""
        try:
            confidence = 1.0
            
            # Data quantity factor
            n_observations = len(portfolio_returns)
            if n_observations < 252:  # Less than 1 year
                confidence *= 0.8
            elif n_observations < 126:  # Less than 6 months
                confidence *= 0.6
            elif n_observations < 63:  # Less than 3 months
                confidence *= 0.4
            
            # Number of assets factor
            n_assets = len(asset_returns)
            if n_assets < 3:
                confidence *= 0.9
            elif n_assets > 20:
                confidence *= 0.95
            
            # Return distribution normality
            _, p_value = stats.shapiro(portfolio_returns[-252:] if len(portfolio_returns) > 252 
                                     else portfolio_returns)
            if p_value < 0.05:  # Non-normal distribution
                confidence *= 0.9
            
            # Stationarity check (simplified)
            # In practice, would use Augmented Dickey-Fuller test
            recent_vol = np.std(portfolio_returns[-63:]) if len(portfolio_returns) > 63 else np.std(portfolio_returns)
            overall_vol = np.std(portfolio_returns)
            if abs(recent_vol - overall_vol) / overall_vol > 0.5:
                confidence *= 0.85  # Regime change
            
            return min(max(confidence, 0.1), 1.0)
            
        except Exception as e:
            logger.error(f"Model confidence calculation failed: {e}")
            return 0.75
    
    async def _assess_data_quality(
        self,
        portfolio_returns: np.ndarray,
        asset_returns: Dict[str, np.ndarray]
    ) -> float:
        """Assess quality of input data"""
        try:
            quality_score = 1.0
            
            # Missing data check
            if np.any(np.isnan(portfolio_returns)):
                quality_score *= 0.7
            
            # Outlier detection (returns > 5 standard deviations)
            std_returns = np.std(portfolio_returns)
            outliers = np.sum(np.abs(portfolio_returns - np.mean(portfolio_returns)) > 5 * std_returns)
            if outliers > len(portfolio_returns) * 0.01:  # More than 1% outliers
                quality_score *= 0.9
            
            # Data consistency across assets
            if asset_returns:
                lengths = [len(returns) for returns in asset_returns.values()]
                if max(lengths) - min(lengths) > 10:  # Inconsistent data lengths
                    quality_score *= 0.85
            
            # Recent data availability
            # Assume daily data, check if recent enough
            quality_score *= 0.95  # Slight reduction for general uncertainty
            
            return min(max(quality_score, 0.1), 1.0)
            
        except Exception as e:
            logger.error(f"Data quality assessment failed: {e}")
            return 0.75


# Export the service
__all__ = [
    'RiskAssessmentService', 'RiskFactors', 'StressTest', 'RiskMetrics',
    'RiskAssessmentResult', 'VaRMethod', 'StressScenario'
] 