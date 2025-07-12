"""
Data Models for StellarVault AI Engine

This module defines comprehensive data models for:
- Asset representations across all asset classes
- Transaction and portfolio models
- Valuation and analysis results
- User and customer models
- Risk and compliance data structures
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid


# Enums for standardized categorization

class AssetType(Enum):
    REAL_ESTATE = "real_estate"
    COMMODITIES = "commodities"
    ART_COLLECTIBLES = "art_collectibles"
    BONDS = "bonds"
    EQUITIES = "equities"
    CRYPTO = "crypto"
    PRECIOUS_METALS = "precious_metals"
    INFRASTRUCTURE = "infrastructure"
    PRIVATE_EQUITY = "private_equity"
    INTELLECTUAL_PROPERTY = "intellectual_property"


class AssetStatus(Enum):
    ACTIVE = "active"
    PENDING = "pending"
    TOKENIZED = "tokenized"
    REDEEMED = "redeemed"
    SUSPENDED = "suspended"
    DELISTED = "delisted"


class TransactionStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DISPUTED = "disputed"


class UserRole(Enum):
    INDIVIDUAL = "individual"
    INSTITUTIONAL = "institutional"
    ACCREDITED_INVESTOR = "accredited_investor"
    QUALIFIED_PURCHASER = "qualified_purchaser"
    FAMILY_OFFICE = "family_office"
    ASSET_MANAGER = "asset_manager"
    COMPLIANCE_OFFICER = "compliance_officer"
    ADMINISTRATOR = "administrator"


class GeographicRegion(Enum):
    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    LATIN_AMERICA = "latin_america"
    MIDDLE_EAST = "middle_east"
    AFRICA = "africa"
    GLOBAL = "global"


# Core Asset Models

@dataclass
class BaseAsset:
    """Base asset model with common properties"""
    asset_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    asset_type: AssetType = AssetType.REAL_ESTATE
    status: AssetStatus = AssetStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Ownership and custody
    owner_id: str = ""
    custodian: Optional[str] = None
    legal_structure: Optional[str] = None
    
    # Geographic information
    country: str = ""
    region: GeographicRegion = GeographicRegion.GLOBAL
    jurisdiction: str = ""
    
    # Financial information
    purchase_price: Optional[float] = None
    current_valuation: Optional[float] = None
    currency: str = "USD"
    
    # Tokenization details
    is_tokenized: bool = False
    token_symbol: Optional[str] = None
    total_tokens: Optional[int] = None
    tokens_outstanding: Optional[int] = None
    stellar_asset_code: Optional[str] = None
    stellar_issuer: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert asset to dictionary"""
        return {
            "asset_id": self.asset_id,
            "name": self.name,
            "description": self.description,
            "asset_type": self.asset_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "owner_id": self.owner_id,
            "custodian": self.custodian,
            "legal_structure": self.legal_structure,
            "country": self.country,
            "region": self.region.value,
            "jurisdiction": self.jurisdiction,
            "purchase_price": self.purchase_price,
            "current_valuation": self.current_valuation,
            "currency": self.currency,
            "is_tokenized": self.is_tokenized,
            "token_symbol": self.token_symbol,
            "total_tokens": self.total_tokens,
            "tokens_outstanding": self.tokens_outstanding,
            "stellar_asset_code": self.stellar_asset_code,
            "stellar_issuer": self.stellar_issuer,
            "metadata": self.metadata,
            "tags": self.tags
        }


@dataclass
class RealEstateAsset(BaseAsset):
    """Real estate specific asset model"""
    asset_type: AssetType = AssetType.REAL_ESTATE
    
    # Property details
    property_type: str = ""  # residential, commercial, industrial, land
    address: str = ""
    city: str = ""
    state_province: str = ""
    postal_code: str = ""
    coordinates: Optional[Dict[str, float]] = None  # lat, lng
    
    # Physical characteristics
    total_area_sqft: Optional[float] = None
    lot_size_sqft: Optional[float] = None
    year_built: Optional[int] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[float] = None
    parking_spaces: Optional[int] = None
    
    # Financial details
    annual_rental_income: Optional[float] = None
    operating_expenses: Optional[float] = None
    net_operating_income: Optional[float] = None
    cap_rate: Optional[float] = None
    cash_on_cash_return: Optional[float] = None
    
    # Market data
    comparable_sales: List[Dict[str, Any]] = field(default_factory=list)
    neighborhood_data: Dict[str, Any] = field(default_factory=dict)
    market_trends: Dict[str, Any] = field(default_factory=dict)
    
    # Legal and regulatory
    zoning: Optional[str] = None
    property_taxes: Optional[float] = None
    hoa_fees: Optional[float] = None
    deed_restrictions: List[str] = field(default_factory=list)


@dataclass
class CommodityAsset(BaseAsset):
    """Commodity specific asset model"""
    asset_type: AssetType = AssetType.COMMODITIES
    
    # Commodity details
    commodity_type: str = ""  # gold, silver, oil, wheat, etc.
    grade_quality: str = ""
    exchange: str = ""
    contract_specification: str = ""
    
    # Physical characteristics
    quantity: float = 0.0
    unit_of_measure: str = ""  # ounces, barrels, bushels, etc.
    purity: Optional[float] = None
    
    # Storage and custody
    storage_location: str = ""
    storage_costs: Optional[float] = None
    insurance_costs: Optional[float] = None
    warehouse_receipts: List[str] = field(default_factory=list)
    
    # Market data
    spot_price: Optional[float] = None
    futures_prices: Dict[str, float] = field(default_factory=dict)
    price_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Supply and demand factors
    production_data: Dict[str, Any] = field(default_factory=dict)
    consumption_data: Dict[str, Any] = field(default_factory=dict)
    inventory_levels: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtAsset(BaseAsset):
    """Art and collectibles specific asset model"""
    asset_type: AssetType = AssetType.ART_COLLECTIBLES
    
    # Artwork details
    artist_name: str = ""
    creation_year: Optional[int] = None
    medium: str = ""
    dimensions: str = ""
    edition_size: Optional[int] = None
    edition_number: Optional[int] = None
    
    # Provenance and authenticity
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    authentication_certificates: List[str] = field(default_factory=list)
    condition_report: Optional[str] = None
    conservation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Market data
    auction_history: List[Dict[str, Any]] = field(default_factory=list)
    comparable_sales: List[Dict[str, Any]] = field(default_factory=list)
    artist_market_data: Dict[str, Any] = field(default_factory=dict)
    
    # Storage and insurance
    storage_location: str = ""
    insurance_value: Optional[float] = None
    security_measures: List[str] = field(default_factory=list)


@dataclass
class BondAsset(BaseAsset):
    """Bond specific asset model"""
    asset_type: AssetType = AssetType.BONDS
    
    # Bond details
    issuer: str = ""
    bond_type: str = ""  # government, corporate, municipal
    credit_rating: str = ""
    face_value: float = 0.0
    coupon_rate: float = 0.0
    
    # Dates
    issue_date: Optional[date] = None
    maturity_date: Optional[date] = None
    first_call_date: Optional[date] = None
    
    # Payment details
    payment_frequency: int = 2  # Semi-annual
    next_payment_date: Optional[date] = None
    accrued_interest: Optional[float] = None
    
    # Market data
    yield_to_maturity: Optional[float] = None
    yield_to_call: Optional[float] = None
    duration: Optional[float] = None
    convexity: Optional[float] = None
    credit_spread: Optional[float] = None
    
    # Features
    callable: bool = False
    puttable: bool = False
    convertible: bool = False
    floating_rate: bool = False


# Transaction Models

@dataclass
class Transaction:
    """Base transaction model"""
    transaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    asset_id: str = ""
    transaction_type: str = ""  # buy, sell, transfer, dividend, etc.
    status: TransactionStatus = TransactionStatus.PENDING
    
    # Financial details
    amount: float = 0.0
    quantity: float = 0.0
    price_per_unit: float = 0.0
    currency: str = "USD"
    fees: float = 0.0
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    settled_at: Optional[datetime] = None
    
    # Stellar blockchain details
    stellar_transaction_id: Optional[str] = None
    stellar_ledger: Optional[int] = None
    stellar_memo: Optional[str] = None
    
    # Counterparty information
    counterparty_id: Optional[str] = None
    counterparty_address: Optional[str] = None
    
    # Metadata
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary"""
        return {
            "transaction_id": self.transaction_id,
            "user_id": self.user_id,
            "asset_id": self.asset_id,
            "transaction_type": self.transaction_type,
            "status": self.status.value,
            "amount": self.amount,
            "quantity": self.quantity,
            "price_per_unit": self.price_per_unit,
            "currency": self.currency,
            "fees": self.fees,
            "created_at": self.created_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "settled_at": self.settled_at.isoformat() if self.settled_at else None,
            "stellar_transaction_id": self.stellar_transaction_id,
            "stellar_ledger": self.stellar_ledger,
            "stellar_memo": self.stellar_memo,
            "counterparty_id": self.counterparty_id,
            "counterparty_address": self.counterparty_address,
            "notes": self.notes,
            "metadata": self.metadata
        }


@dataclass
class Portfolio:
    """Portfolio model"""
    portfolio_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    name: str = ""
    description: str = ""
    
    # Holdings
    holdings: List[Dict[str, Any]] = field(default_factory=list)
    total_value: float = 0.0
    currency: str = "USD"
    
    # Performance metrics
    inception_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    total_return: Optional[float] = None
    annualized_return: Optional[float] = None
    volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    
    # Risk metrics
    value_at_risk_95: Optional[float] = None
    beta: Optional[float] = None
    alpha: Optional[float] = None
    
    # Allocation
    asset_allocation: Dict[str, float] = field(default_factory=dict)
    geographic_allocation: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


# User and Customer Models

@dataclass
class User:
    """User model"""
    user_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    email: str = ""
    username: Optional[str] = None
    role: UserRole = UserRole.INDIVIDUAL
    
    # Personal information
    first_name: str = ""
    last_name: str = ""
    date_of_birth: Optional[date] = None
    nationality: str = ""
    
    # Contact information
    phone: str = ""
    address: str = ""
    city: str = ""
    state_province: str = ""
    country: str = ""
    postal_code: str = ""
    
    # Account details
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    is_verified: bool = False
    
    # Financial profile
    net_worth: Optional[float] = None
    annual_income: Optional[float] = None
    investment_experience: str = ""
    risk_tolerance: str = "moderate"
    
    # Compliance
    kyc_status: str = "pending"
    aml_risk_score: Optional[float] = None
    accredited_investor: bool = False
    qualified_purchaser: bool = False
    
    # Preferences
    preferred_currency: str = "USD"
    preferred_language: str = "en"
    notification_preferences: Dict[str, bool] = field(default_factory=dict)
    
    # Stellar account
    stellar_public_key: Optional[str] = None
    stellar_account_id: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# Valuation Models

@dataclass
class ValuationResult:
    """Valuation result model"""
    valuation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    asset_id: str = ""
    valuation_date: datetime = field(default_factory=datetime.now)
    
    # Valuation details
    estimated_value: float = 0.0
    currency: str = "USD"
    valuation_method: str = ""
    confidence_score: float = 0.0
    
    # Value components
    intrinsic_value: Optional[float] = None
    market_value: Optional[float] = None
    liquidation_value: Optional[float] = None
    replacement_cost: Optional[float] = None
    
    # Comparables and methodology
    comparables_used: List[Dict[str, Any]] = field(default_factory=list)
    adjustments_made: List[Dict[str, Any]] = field(default_factory=list)
    methodology_notes: str = ""
    
    # Risk factors
    risk_factors: List[str] = field(default_factory=list)
    uncertainty_range: Optional[Dict[str, float]] = None
    
    # Market context
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    economic_indicators: Dict[str, Any] = field(default_factory=dict)
    
    # Validation
    validated_by: Optional[str] = None
    validation_date: Optional[datetime] = None
    next_valuation_date: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskAssessment:
    """Risk assessment model"""
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subject_id: str = ""  # asset_id or portfolio_id
    subject_type: str = ""  # asset or portfolio
    assessment_date: datetime = field(default_factory=datetime.now)
    
    # Overall risk metrics
    overall_risk_score: float = 0.0
    risk_level: str = "medium"  # low, medium, high, critical
    
    # Specific risk measures
    value_at_risk_95: Optional[float] = None
    value_at_risk_99: Optional[float] = None
    expected_shortfall: Optional[float] = None
    maximum_drawdown: Optional[float] = None
    volatility: Optional[float] = None
    
    # Risk factors
    market_risk: float = 0.0
    credit_risk: float = 0.0
    liquidity_risk: float = 0.0
    operational_risk: float = 0.0
    regulatory_risk: float = 0.0
    concentration_risk: float = 0.0
    
    # Stress test results
    stress_test_scenarios: Dict[str, float] = field(default_factory=dict)
    worst_case_scenario: Optional[float] = None
    
    # Recommendations
    risk_mitigation_recommendations: List[str] = field(default_factory=list)
    
    # Methodology
    methodology: str = ""
    confidence_level: float = 0.95
    time_horizon_days: int = 252
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# Compliance Models

@dataclass
class ComplianceCheck:
    """Compliance check model"""
    check_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subject_id: str = ""
    check_type: str = ""  # kyc, aml, sanctions, etc.
    status: str = "pending"  # pending, passed, failed, review_required
    
    # Check details
    performed_at: datetime = field(default_factory=datetime.now)
    performed_by: str = ""
    check_version: str = "1.0"
    
    # Results
    passed: bool = False
    score: Optional[float] = None
    findings: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    
    # Documentation
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    review_notes: str = ""
    
    # Follow-up
    requires_review: bool = False
    reviewer_id: Optional[str] = None
    review_deadline: Optional[datetime] = None
    next_check_date: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# Analytics Models

@dataclass
class MarketData:
    """Market data model"""
    data_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    asset_id: Optional[str] = None
    symbol: str = ""
    exchange: str = ""
    data_type: str = ""  # price, volume, sentiment, etc.
    
    # Data points
    timestamp: datetime = field(default_factory=datetime.now)
    value: float = 0.0
    volume: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open_price: Optional[float] = None
    close_price: Optional[float] = None
    
    # Quality metrics
    data_source: str = ""
    confidence: float = 1.0
    is_real_time: bool = False
    delay_minutes: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsReport:
    """Analytics report model"""
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    report_type: str = ""
    subject_id: str = ""
    generated_at: datetime = field(default_factory=datetime.now)
    
    # Report content
    title: str = ""
    summary: str = ""
    key_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Data and metrics
    data_points: Dict[str, Any] = field(default_factory=dict)
    charts_data: Dict[str, Any] = field(default_factory=dict)
    
    # Validation
    generated_by: str = "system"
    reviewed_by: Optional[str] = None
    approved_by: Optional[str] = None
    
    # Distribution
    recipients: List[str] = field(default_factory=list)
    is_published: bool = False
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# Factory functions for creating model instances

def create_real_estate_asset(
    name: str,
    address: str,
    property_type: str,
    owner_id: str,
    **kwargs
) -> RealEstateAsset:
    """Factory function to create real estate asset"""
    return RealEstateAsset(
        name=name,
        address=address,
        property_type=property_type,
        owner_id=owner_id,
        **kwargs
    )


def create_commodity_asset(
    name: str,
    commodity_type: str,
    quantity: float,
    unit_of_measure: str,
    owner_id: str,
    **kwargs
) -> CommodityAsset:
    """Factory function to create commodity asset"""
    return CommodityAsset(
        name=name,
        commodity_type=commodity_type,
        quantity=quantity,
        unit_of_measure=unit_of_measure,
        owner_id=owner_id,
        **kwargs
    )


def create_art_asset(
    name: str,
    artist_name: str,
    medium: str,
    owner_id: str,
    **kwargs
) -> ArtAsset:
    """Factory function to create art asset"""
    return ArtAsset(
        name=name,
        artist_name=artist_name,
        medium=medium,
        owner_id=owner_id,
        **kwargs
    )


def create_bond_asset(
    name: str,
    issuer: str,
    face_value: float,
    coupon_rate: float,
    maturity_date: date,
    owner_id: str,
    **kwargs
) -> BondAsset:
    """Factory function to create bond asset"""
    return BondAsset(
        name=name,
        issuer=issuer,
        face_value=face_value,
        coupon_rate=coupon_rate,
        maturity_date=maturity_date,
        owner_id=owner_id,
        **kwargs
    )


def create_transaction(
    user_id: str,
    asset_id: str,
    transaction_type: str,
    amount: float,
    quantity: float,
    price_per_unit: float,
    **kwargs
) -> Transaction:
    """Factory function to create transaction"""
    return Transaction(
        user_id=user_id,
        asset_id=asset_id,
        transaction_type=transaction_type,
        amount=amount,
        quantity=quantity,
        price_per_unit=price_per_unit,
        **kwargs
    )


def create_user(
    email: str,
    first_name: str,
    last_name: str,
    role: UserRole = UserRole.INDIVIDUAL,
    **kwargs
) -> User:
    """Factory function to create user"""
    return User(
        email=email,
        first_name=first_name,
        last_name=last_name,
        role=role,
        **kwargs
    )


# Export all models and enums
__all__ = [
    # Enums
    'AssetType', 'AssetStatus', 'TransactionStatus', 'UserRole', 'GeographicRegion',
    
    # Asset Models
    'BaseAsset', 'RealEstateAsset', 'CommodityAsset', 'ArtAsset', 'BondAsset',
    
    # Transaction and Portfolio Models
    'Transaction', 'Portfolio',
    
    # User Models
    'User',
    
    # Analysis Models
    'ValuationResult', 'RiskAssessment', 'ComplianceCheck',
    
    # Analytics Models
    'MarketData', 'AnalyticsReport',
    
    # Factory Functions
    'create_real_estate_asset', 'create_commodity_asset', 'create_art_asset',
    'create_bond_asset', 'create_transaction', 'create_user'
] 