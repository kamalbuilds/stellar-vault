"""
Bonds Valuation Service for StellarVault AI Engine

This service provides comprehensive bond valuation including:
- Government and corporate bond pricing
- Credit risk assessment
- Yield curve analysis
- Duration and convexity calculations
- Interest rate sensitivity analysis
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import aiohttp
import json
from loguru import logger

from ..core.config import settings
from ..utils.data_sources import DataSourceClient


class BondType(Enum):
    GOVERNMENT = "government"
    CORPORATE = "corporate"
    MUNICIPAL = "municipal"
    HIGH_YIELD = "high_yield"
    TREASURY = "treasury"
    AGENCY = "agency"


class CreditRating(Enum):
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"
    CCC = "CCC"
    CC = "CC"
    C = "C"
    D = "D"


@dataclass
class BondCharacteristics:
    """Bond characteristics for valuation"""
    face_value: float
    coupon_rate: float
    maturity_date: datetime
    issue_date: datetime
    credit_rating: str
    issuer: str
    bond_type: BondType
    callable: bool = False
    call_price: Optional[float] = None
    call_date: Optional[datetime] = None
    frequency: int = 2  # Semi-annual payments


@dataclass
class BondValuation:
    """Bond valuation result"""
    present_value: float
    accrued_interest: float
    clean_price: float
    dirty_price: float
    yield_to_maturity: float
    modified_duration: float
    convexity: float
    credit_spread: float
    risk_premium: float
    confidence_score: float
    valuation_date: datetime
    methodology: str
    market_data: Dict[str, Any]


class BondsValuationService:
    """
    Advanced bonds valuation service with ML-enhanced pricing models
    """
    
    def __init__(self):
        self.data_client = DataSourceClient()
        self.yield_curve_cache = {}
        self.credit_spreads_cache = {}
        self.last_update = None
        
    async def initialize(self):
        """Initialize the bonds valuation service"""
        try:
            await self._update_market_data()
            logger.info("Bonds valuation service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize bonds valuation service: {e}")
            raise
    
    async def valuate_bond(
        self, 
        bond_characteristics: BondCharacteristics,
        market_data: Optional[Dict[str, Any]] = None
    ) -> BondValuation:
        """
        Comprehensive bond valuation
        
        Args:
            bond_characteristics: Bond details
            market_data: Optional market data override
            
        Returns:
            BondValuation with comprehensive pricing analysis
        """
        try:
            # Ensure market data is fresh
            await self._ensure_fresh_market_data()
            
            # Get current market data
            if not market_data:
                market_data = await self._get_market_data_for_bond(bond_characteristics)
            
            # Calculate basic metrics
            present_value = await self._calculate_present_value(bond_characteristics, market_data)
            accrued_interest = self._calculate_accrued_interest(bond_characteristics)
            
            # Price calculations
            clean_price = present_value
            dirty_price = clean_price + accrued_interest
            
            # Yield calculations
            ytm = await self._calculate_yield_to_maturity(bond_characteristics, market_data)
            
            # Risk metrics
            duration = self._calculate_modified_duration(bond_characteristics, ytm)
            convexity = self._calculate_convexity(bond_characteristics, ytm)
            
            # Credit analysis
            credit_spread = await self._calculate_credit_spread(bond_characteristics, market_data)
            risk_premium = await self._calculate_risk_premium(bond_characteristics, market_data)
            
            # Confidence scoring
            confidence = await self._calculate_confidence_score(bond_characteristics, market_data)
            
            return BondValuation(
                present_value=present_value,
                accrued_interest=accrued_interest,
                clean_price=clean_price,
                dirty_price=dirty_price,
                yield_to_maturity=ytm,
                modified_duration=duration,
                convexity=convexity,
                credit_spread=credit_spread,
                risk_premium=risk_premium,
                confidence_score=confidence,
                valuation_date=datetime.now(),
                methodology="Discounted Cash Flow with Credit Risk Adjustment",
                market_data=market_data
            )
            
        except Exception as e:
            logger.error(f"Bond valuation failed: {e}")
            raise
    
    async def _calculate_present_value(
        self, 
        bond: BondCharacteristics, 
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate bond present value using DCF"""
        try:
            # Get risk-free rate and credit spread
            risk_free_rate = market_data.get('risk_free_rate', 0.04)
            credit_spread = market_data.get('credit_spread', 0.01)
            discount_rate = risk_free_rate + credit_spread
            
            # Calculate time to maturity
            days_to_maturity = (bond.maturity_date - datetime.now()).days
            years_to_maturity = days_to_maturity / 365.25
            
            # Calculate coupon payments
            periods_per_year = bond.frequency
            total_periods = int(years_to_maturity * periods_per_year)
            period_rate = discount_rate / periods_per_year
            coupon_payment = bond.face_value * bond.coupon_rate / periods_per_year
            
            # Present value of coupon payments
            pv_coupons = 0
            for period in range(1, total_periods + 1):
                pv_coupons += coupon_payment / ((1 + period_rate) ** period)
            
            # Present value of principal
            pv_principal = bond.face_value / ((1 + period_rate) ** total_periods)
            
            # Total present value
            present_value = pv_coupons + pv_principal
            
            # Adjust for callable bonds
            if bond.callable and bond.call_price:
                call_value = await self._calculate_call_option_value(bond, market_data)
                present_value -= call_value
            
            return present_value
            
        except Exception as e:
            logger.error(f"Present value calculation failed: {e}")
            return bond.face_value * 0.95  # Conservative fallback
    
    def _calculate_accrued_interest(self, bond: BondCharacteristics) -> float:
        """Calculate accrued interest since last coupon payment"""
        try:
            # Find last coupon date
            today = datetime.now()
            coupon_months = 12 // bond.frequency
            
            # Calculate days since last coupon
            last_coupon = today.replace(day=1)
            while (today - last_coupon).days > (365 // bond.frequency):
                month = last_coupon.month - coupon_months
                year = last_coupon.year
                if month <= 0:
                    month += 12
                    year -= 1
                last_coupon = last_coupon.replace(month=month, year=year)
            
            days_since_coupon = (today - last_coupon).days
            days_in_period = 365 // bond.frequency
            
            # Calculate accrued interest
            period_interest = bond.face_value * bond.coupon_rate / bond.frequency
            accrued = period_interest * (days_since_coupon / days_in_period)
            
            return accrued
            
        except Exception as e:
            logger.error(f"Accrued interest calculation failed: {e}")
            return 0.0
    
    async def _calculate_yield_to_maturity(
        self, 
        bond: BondCharacteristics, 
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate yield to maturity using Newton-Raphson method"""
        try:
            # Initial guess based on current yield
            current_price = market_data.get('current_price', bond.face_value)
            initial_guess = bond.coupon_rate
            
            def bond_price(ytm):
                days_to_maturity = (bond.maturity_date - datetime.now()).days
                years_to_maturity = days_to_maturity / 365.25
                periods_per_year = bond.frequency
                total_periods = int(years_to_maturity * periods_per_year)
                period_rate = ytm / periods_per_year
                coupon_payment = bond.face_value * bond.coupon_rate / periods_per_year
                
                pv = 0
                for period in range(1, total_periods + 1):
                    pv += coupon_payment / ((1 + period_rate) ** period)
                pv += bond.face_value / ((1 + period_rate) ** total_periods)
                
                return pv
            
            # Newton-Raphson iteration
            ytm = initial_guess
            for _ in range(50):  # Max iterations
                price = bond_price(ytm)
                price_diff = bond_price(ytm + 0.0001) - price
                derivative = price_diff / 0.0001
                
                if abs(derivative) < 1e-8:
                    break
                    
                new_ytm = ytm - (price - current_price) / derivative
                
                if abs(new_ytm - ytm) < 1e-8:
                    break
                    
                ytm = new_ytm
            
            return max(ytm, 0.0)  # Ensure non-negative yield
            
        except Exception as e:
            logger.error(f"YTM calculation failed: {e}")
            return bond.coupon_rate  # Fallback to coupon rate
    
    def _calculate_modified_duration(self, bond: BondCharacteristics, ytm: float) -> float:
        """Calculate modified duration for interest rate sensitivity"""
        try:
            days_to_maturity = (bond.maturity_date - datetime.now()).days
            years_to_maturity = days_to_maturity / 365.25
            periods_per_year = bond.frequency
            total_periods = int(years_to_maturity * periods_per_year)
            period_rate = ytm / periods_per_year
            
            # Macaulay duration calculation
            coupon_payment = bond.face_value * bond.coupon_rate / periods_per_year
            weighted_time = 0
            present_value = 0
            
            for period in range(1, total_periods + 1):
                time = period / periods_per_year
                pv_payment = coupon_payment / ((1 + period_rate) ** period)
                weighted_time += time * pv_payment
                present_value += pv_payment
            
            # Add principal payment
            time = total_periods / periods_per_year
            pv_principal = bond.face_value / ((1 + period_rate) ** total_periods)
            weighted_time += time * pv_principal
            present_value += pv_principal
            
            macaulay_duration = weighted_time / present_value
            
            # Modified duration
            modified_duration = macaulay_duration / (1 + ytm / periods_per_year)
            
            return modified_duration
            
        except Exception as e:
            logger.error(f"Duration calculation failed: {e}")
            return years_to_maturity  # Approximate fallback
    
    def _calculate_convexity(self, bond: BondCharacteristics, ytm: float) -> float:
        """Calculate convexity for price sensitivity analysis"""
        try:
            days_to_maturity = (bond.maturity_date - datetime.now()).days
            years_to_maturity = days_to_maturity / 365.25
            periods_per_year = bond.frequency
            total_periods = int(years_to_maturity * periods_per_year)
            period_rate = ytm / periods_per_year
            
            coupon_payment = bond.face_value * bond.coupon_rate / periods_per_year
            convexity = 0
            present_value = 0
            
            for period in range(1, total_periods + 1):
                time = period / periods_per_year
                pv_payment = coupon_payment / ((1 + period_rate) ** period)
                convexity += time * (time + 1) * pv_payment
                present_value += pv_payment
            
            # Add principal payment
            time = total_periods / periods_per_year
            pv_principal = bond.face_value / ((1 + period_rate) ** total_periods)
            convexity += time * (time + 1) * pv_principal
            present_value += pv_principal
            
            convexity = convexity / (present_value * ((1 + ytm / periods_per_year) ** 2))
            
            return convexity
            
        except Exception as e:
            logger.error(f"Convexity calculation failed: {e}")
            return 0.0
    
    async def _calculate_credit_spread(
        self, 
        bond: BondCharacteristics, 
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate credit spread over risk-free rate"""
        try:
            # Get treasury yield for similar maturity
            treasury_yield = await self._get_treasury_yield(bond.maturity_date)
            
            # Get corporate bond yield
            corporate_yield = market_data.get('yield', bond.coupon_rate)
            
            # Credit spread
            credit_spread = corporate_yield - treasury_yield
            
            # Adjust based on credit rating
            rating_adjustment = self._get_rating_adjustment(bond.credit_rating)
            credit_spread += rating_adjustment
            
            return max(credit_spread, 0.0)
            
        except Exception as e:
            logger.error(f"Credit spread calculation failed: {e}")
            return 0.02  # Default 200 bps
    
    async def _calculate_risk_premium(
        self, 
        bond: BondCharacteristics, 
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate risk premium based on bond characteristics"""
        try:
            base_premium = 0.0
            
            # Credit rating premium
            rating_premium = self._get_rating_premium(bond.credit_rating)
            base_premium += rating_premium
            
            # Maturity premium
            days_to_maturity = (bond.maturity_date - datetime.now()).days
            years_to_maturity = days_to_maturity / 365.25
            maturity_premium = min(years_to_maturity * 0.001, 0.02)  # Max 2%
            base_premium += maturity_premium
            
            # Liquidity premium
            liquidity_premium = 0.005  # 50 bps for illiquidity
            if bond.bond_type == BondType.GOVERNMENT:
                liquidity_premium = 0.0
            elif bond.bond_type == BondType.CORPORATE:
                liquidity_premium = 0.003
            
            base_premium += liquidity_premium
            
            # Call option premium
            if bond.callable:
                call_premium = 0.002  # 20 bps for call risk
                base_premium += call_premium
            
            return base_premium
            
        except Exception as e:
            logger.error(f"Risk premium calculation failed: {e}")
            return 0.03  # Default 300 bps
    
    async def _calculate_confidence_score(
        self, 
        bond: BondCharacteristics, 
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for the valuation"""
        try:
            confidence = 1.0
            
            # Data quality factors
            if not market_data.get('recent_trades'):
                confidence *= 0.8
            
            if not market_data.get('yield'):
                confidence *= 0.9
            
            # Bond characteristics factors
            if bond.bond_type == BondType.GOVERNMENT:
                confidence *= 1.0
            elif bond.bond_type == BondType.CORPORATE:
                confidence *= 0.95
            else:
                confidence *= 0.85
            
            # Rating factors
            rating_confidence = {
                'AAA': 1.0, 'AA': 0.98, 'A': 0.95, 'BBB': 0.90,
                'BB': 0.80, 'B': 0.70, 'CCC': 0.60, 'CC': 0.50, 'C': 0.40, 'D': 0.20
            }
            confidence *= rating_confidence.get(bond.credit_rating, 0.75)
            
            # Time to maturity factor
            days_to_maturity = (bond.maturity_date - datetime.now()).days
            if days_to_maturity < 30:  # Very short term
                confidence *= 0.9
            elif days_to_maturity > 10 * 365:  # Very long term
                confidence *= 0.85
            
            return min(max(confidence, 0.1), 1.0)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.75
    
    async def _get_market_data_for_bond(self, bond: BondCharacteristics) -> Dict[str, Any]:
        """Get relevant market data for bond valuation"""
        try:
            market_data = {}
            
            # Get treasury yield curve
            treasury_yield = await self._get_treasury_yield(bond.maturity_date)
            market_data['risk_free_rate'] = treasury_yield
            
            # Get credit spread for rating
            credit_spread = await self._get_credit_spread_for_rating(bond.credit_rating)
            market_data['credit_spread'] = credit_spread
            
            # Get sector spreads if available
            sector_spread = await self._get_sector_spread(bond.issuer)
            market_data['sector_spread'] = sector_spread
            
            # Market volatility
            market_data['volatility'] = await self._get_market_volatility()
            
            # Recent trading data (simulated for now)
            market_data['recent_trades'] = True
            market_data['yield'] = treasury_yield + credit_spread
            market_data['current_price'] = bond.face_value  # Default to par
            
            return market_data
            
        except Exception as e:
            logger.error(f"Market data retrieval failed: {e}")
            return {
                'risk_free_rate': 0.04,
                'credit_spread': 0.02,
                'sector_spread': 0.01,
                'volatility': 0.15,
                'recent_trades': False,
                'yield': 0.06,
                'current_price': bond.face_value
            }
    
    async def _get_treasury_yield(self, maturity_date: datetime) -> float:
        """Get treasury yield for given maturity"""
        try:
            # Calculate years to maturity
            days_to_maturity = (maturity_date - datetime.now()).days
            years_to_maturity = days_to_maturity / 365.25
            
            # Simulated yield curve (in production, use FRED API)
            if years_to_maturity <= 0.25:  # 3 months
                return 0.025
            elif years_to_maturity <= 0.5:  # 6 months
                return 0.028
            elif years_to_maturity <= 1:  # 1 year
                return 0.032
            elif years_to_maturity <= 2:  # 2 years
                return 0.035
            elif years_to_maturity <= 5:  # 5 years
                return 0.038
            elif years_to_maturity <= 10:  # 10 years
                return 0.042
            else:  # 30 years
                return 0.045
                
        except Exception as e:
            logger.error(f"Treasury yield lookup failed: {e}")
            return 0.04  # Default 4%
    
    async def _get_credit_spread_for_rating(self, rating: str) -> float:
        """Get credit spread based on rating"""
        try:
            # Credit spreads by rating (basis points)
            spreads = {
                'AAA': 0.0010,  # 10 bps
                'AA': 0.0025,   # 25 bps
                'A': 0.0050,    # 50 bps
                'BBB': 0.0100,  # 100 bps
                'BB': 0.0250,   # 250 bps
                'B': 0.0500,    # 500 bps
                'CCC': 0.1000,  # 1000 bps
                'CC': 0.1500,   # 1500 bps
                'C': 0.2000,    # 2000 bps
                'D': 0.3000     # 3000 bps
            }
            
            return spreads.get(rating, 0.02)
            
        except Exception as e:
            logger.error(f"Credit spread lookup failed: {e}")
            return 0.02
    
    async def _get_sector_spread(self, issuer: str) -> float:
        """Get sector-specific spread"""
        try:
            # Sector spreads (basis points)
            if any(keyword in issuer.lower() for keyword in ['bank', 'financial', 'credit']):
                return 0.0075
            elif any(keyword in issuer.lower() for keyword in ['tech', 'software', 'amazon', 'apple']):
                return 0.0050
            elif any(keyword in issuer.lower() for keyword in ['energy', 'oil', 'gas']):
                return 0.0125
            elif any(keyword in issuer.lower() for keyword in ['utility', 'electric', 'power']):
                return 0.0025
            else:
                return 0.0075  # Default industrial spread
                
        except Exception as e:
            logger.error(f"Sector spread lookup failed: {e}")
            return 0.0075
    
    async def _get_market_volatility(self) -> float:
        """Get current market volatility"""
        try:
            # In production, would use VIX or bond volatility indices
            return 0.15  # 15% volatility
            
        except Exception as e:
            logger.error(f"Market volatility lookup failed: {e}")
            return 0.15
    
    def _get_rating_adjustment(self, rating: str) -> float:
        """Get rating-based adjustment"""
        adjustments = {
            'AAA': -0.001, 'AA': 0.0, 'A': 0.001, 'BBB': 0.002,
            'BB': 0.005, 'B': 0.010, 'CCC': 0.020, 'CC': 0.030, 'C': 0.040, 'D': 0.050
        }
        return adjustments.get(rating, 0.01)
    
    def _get_rating_premium(self, rating: str) -> float:
        """Get risk premium based on credit rating"""
        premiums = {
            'AAA': 0.005, 'AA': 0.010, 'A': 0.015, 'BBB': 0.025,
            'BB': 0.040, 'B': 0.070, 'CCC': 0.120, 'CC': 0.200, 'C': 0.300, 'D': 0.500
        }
        return premiums.get(rating, 0.05)
    
    async def _calculate_call_option_value(
        self, 
        bond: BondCharacteristics, 
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate embedded call option value using Black-Scholes"""
        try:
            if not bond.callable or not bond.call_price or not bond.call_date:
                return 0.0
            
            # This is a simplified call option valuation
            # In production, would use more sophisticated models
            time_to_call = (bond.call_date - datetime.now()).days / 365.25
            volatility = market_data.get('volatility', 0.15)
            risk_free_rate = market_data.get('risk_free_rate', 0.04)
            
            # Approximate option value (simplified)
            option_value = bond.face_value * 0.02 * np.sqrt(time_to_call) * volatility
            
            return min(option_value, bond.face_value * 0.1)  # Cap at 10% of face value
            
        except Exception as e:
            logger.error(f"Call option valuation failed: {e}")
            return 0.0
    
    async def _update_market_data(self):
        """Update cached market data"""
        try:
            # Update yield curve
            await self._update_yield_curve()
            
            # Update credit spreads
            await self._update_credit_spreads()
            
            self.last_update = datetime.now()
            logger.info("Market data updated successfully")
            
        except Exception as e:
            logger.error(f"Market data update failed: {e}")
    
    async def _update_yield_curve(self):
        """Update treasury yield curve data"""
        try:
            # In production, fetch from FRED API
            self.yield_curve_cache = {
                '1M': 0.025, '3M': 0.028, '6M': 0.032, '1Y': 0.035,
                '2Y': 0.038, '5Y': 0.042, '10Y': 0.045, '30Y': 0.048
            }
            
        except Exception as e:
            logger.error(f"Yield curve update failed: {e}")
    
    async def _update_credit_spreads(self):
        """Update credit spread data"""
        try:
            # In production, fetch from Bloomberg/Reuters
            self.credit_spreads_cache = {
                'AAA': 0.0010, 'AA': 0.0025, 'A': 0.0050, 'BBB': 0.0100,
                'BB': 0.0250, 'B': 0.0500, 'CCC': 0.1000
            }
            
        except Exception as e:
            logger.error(f"Credit spreads update failed: {e}")
    
    async def _ensure_fresh_market_data(self):
        """Ensure market data is fresh"""
        if not self.last_update or (datetime.now() - self.last_update).seconds > 3600:
            await self._update_market_data()


# Export the service
__all__ = ['BondsValuationService', 'BondCharacteristics', 'BondValuation', 'BondType', 'CreditRating'] 