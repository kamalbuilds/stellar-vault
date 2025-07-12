"""
Risk Management Package for StellarVault AI Engine

This package provides comprehensive risk management capabilities including:
- Portfolio optimization and risk-parity algorithms
- Value at Risk (VaR) calculations and risk assessment
- Stress testing and scenario analysis
- Monte Carlo simulations for risk modeling
"""

from .portfolio_optimizer import (
    PortfolioOptimizer,
    AssetData,
    PortfolioConstraints,
    OptimizationResult,
    OptimizationMethod,
    RiskMetric
)

from .risk_assessor import (
    RiskAssessmentService,
    RiskFactors,
    StressTest,
    RiskMetrics,
    RiskAssessmentResult,
    VaRMethod,
    StressScenario
)

__all__ = [
    # Portfolio Optimization
    'PortfolioOptimizer',
    'AssetData',
    'PortfolioConstraints',
    'OptimizationResult',
    'OptimizationMethod',
    'RiskMetric',
    
    # Risk Assessment
    'RiskAssessmentService',
    'RiskFactors',
    'StressTest',
    'RiskMetrics',
    'RiskAssessmentResult',
    'VaRMethod',
    'StressScenario'
] 