"""
Database management and models for StellarVault AI Engine
"""
import asyncio
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, JSON,
    ForeignKey, UniqueConstraint, Index, create_engine, MetaData, text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

from .config import settings

# Database setup
Base = declarative_base()
metadata = MetaData()

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=True,
    echo=settings.DEBUG
)

# Create session factory
AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


class Asset(Base):
    """
    Asset model for storing asset information
    """
    __tablename__ = "assets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    asset_type = Column(String(50), nullable=False)  # real_estate, commodities, art, bonds
    description = Column(Text)
    
    # Asset metadata
    asset_metadata = Column(JSON, default=dict)
    location = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    valuations = relationship("Valuation", back_populates="asset")
    risk_assessments = relationship("RiskAssessment", back_populates="asset")
    
    # Indexes
    __table_args__ = (
        Index("idx_asset_type", "asset_type"),
        Index("idx_asset_location", "location"),
        Index("idx_asset_created_at", "created_at"),
    )


class Valuation(Base):
    """
    Valuation model for storing asset valuations
    """
    __tablename__ = "valuations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset_id = Column(UUID(as_uuid=True), ForeignKey("assets.id"), nullable=False)
    
    # Valuation data
    estimated_value = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    model_used = Column(String(100), nullable=False)
    valuation_date = Column(DateTime(timezone=True), server_default=func.now())
    
    # Valuation methodology
    methodology = Column(String(100))
    comparables = Column(JSON)
    market_conditions = Column(JSON)
    
    # AI model outputs
    model_confidence = Column(Float)
    feature_importance = Column(JSON)
    prediction_interval = Column(JSON)
    
    # Relationships
    asset = relationship("Asset", back_populates="valuations")
    
    # Indexes
    __table_args__ = (
        Index("idx_valuation_asset_id", "asset_id"),
        Index("idx_valuation_date", "valuation_date"),
        Index("idx_valuation_model", "model_used"),
    )


class RiskAssessment(Base):
    """
    Risk assessment model for storing risk analysis
    """
    __tablename__ = "risk_assessments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset_id = Column(UUID(as_uuid=True), ForeignKey("assets.id"), nullable=False)
    
    # Risk metrics
    risk_score = Column(Float, nullable=False)
    volatility = Column(Float)
    var_95 = Column(Float)  # Value at Risk 95%
    expected_shortfall = Column(Float)
    
    # Risk factors
    market_risk = Column(Float)
    liquidity_risk = Column(Float)
    credit_risk = Column(Float)
    operational_risk = Column(Float)
    
    # Risk analysis
    risk_factors = Column(JSON)
    stress_test_results = Column(JSON)
    correlation_matrix = Column(JSON)
    
    assessment_date = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    asset = relationship("Asset", back_populates="risk_assessments")
    
    # Indexes
    __table_args__ = (
        Index("idx_risk_asset_id", "asset_id"),
        Index("idx_risk_score", "risk_score"),
        Index("idx_risk_date", "assessment_date"),
    )


class MarketData(Base):
    """
    Market data model for storing market information
    """
    __tablename__ = "market_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(50), nullable=False)
    source = Column(String(100), nullable=False)
    data_type = Column(String(50), nullable=False)  # price, volume, sentiment
    
    # Market data
    value = Column(Float)
    data_metadata = Column(JSON)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    
    # Data quality
    quality_score = Column(Float, default=1.0)
    is_validated = Column(Boolean, default=False)
    
    # Indexes
    __table_args__ = (
        Index("idx_market_symbol", "symbol"),
        Index("idx_market_source", "source"),
        Index("idx_market_timestamp", "timestamp"),
        UniqueConstraint("symbol", "source", "timestamp", name="uq_market_data"),
    )


class ComplianceRecord(Base):
    """
    Compliance record model for storing regulatory compliance data
    """
    __tablename__ = "compliance_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset_id = Column(UUID(as_uuid=True), ForeignKey("assets.id"), nullable=False)
    
    # Compliance status
    status = Column(String(50), nullable=False)  # compliant, non_compliant, pending
    jurisdiction = Column(String(100), nullable=False)
    regulation_type = Column(String(100), nullable=False)
    
    # Compliance details
    requirements = Column(JSON)
    checks_performed = Column(JSON)
    violations = Column(JSON)
    remediation_steps = Column(JSON)
    
    # Timestamps
    check_date = Column(DateTime(timezone=True), server_default=func.now())
    expiry_date = Column(DateTime(timezone=True))
    
    # Relationships
    asset = relationship("Asset")
    
    # Indexes
    __table_args__ = (
        Index("idx_compliance_asset_id", "asset_id"),
        Index("idx_compliance_status", "status"),
        Index("idx_compliance_jurisdiction", "jurisdiction"),
    )


class UserPortfolio(Base):
    """
    User portfolio model for storing portfolio data
    """
    __tablename__ = "user_portfolios"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False)
    portfolio_name = Column(String(255), nullable=False)
    
    # Portfolio composition
    assets = Column(JSON)  # List of asset IDs and weights
    target_allocation = Column(JSON)
    current_allocation = Column(JSON)
    
    # Portfolio metrics
    total_value = Column(Float)
    total_return = Column(Float)
    risk_score = Column(Float)
    sharpe_ratio = Column(Float)
    
    # Rebalancing
    last_rebalance = Column(DateTime(timezone=True))
    rebalance_threshold = Column(Float, default=0.05)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index("idx_portfolio_user_id", "user_id"),
        Index("idx_portfolio_updated_at", "updated_at"),
    )


class ModelPerformance(Base):
    """
    Model performance tracking
    """
    __tablename__ = "model_performance"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    mae = Column(Float)  # Mean Absolute Error
    rmse = Column(Float)  # Root Mean Square Error
    
    # Test data
    test_data_size = Column(Integer)
    test_date = Column(DateTime(timezone=True), server_default=func.now())
    
    # Additional metrics
    metrics = Column(JSON)
    
    # Indexes
    __table_args__ = (
        Index("idx_model_name", "model_name"),
        Index("idx_model_version", "model_version"),
        Index("idx_test_date", "test_date"),
    )


# Database session dependency
async def get_db_session() -> AsyncSession:
    """
    Get database session for dependency injection
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# Database initialization
async def create_tables():
    """
    Create all database tables
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# Database utilities
async def get_asset_by_id(session: AsyncSession, asset_id: str) -> Optional[Asset]:
    """
    Get asset by ID
    """
    result = await session.get(Asset, asset_id)
    return result


async def create_asset(session: AsyncSession, asset_data: Dict[str, Any]) -> Asset:
    """
    Create new asset
    """
    asset = Asset(**asset_data)
    session.add(asset)
    await session.commit()
    await session.refresh(asset)
    return asset


async def get_latest_valuation(session: AsyncSession, asset_id: str) -> Optional[Valuation]:
    """
    Get latest valuation for an asset
    """
    from sqlalchemy import select
    
    stmt = select(Valuation).where(
        Valuation.asset_id == asset_id
    ).order_by(Valuation.valuation_date.desc()).limit(1)
    
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def create_valuation(session: AsyncSession, valuation_data: Dict[str, Any]) -> Valuation:
    """
    Create new valuation
    """
    valuation = Valuation(**valuation_data)
    session.add(valuation)
    await session.commit()
    await session.refresh(valuation)
    return valuation


async def get_portfolio_summary(session: AsyncSession, user_id: str) -> Dict[str, Any]:
    """
    Get portfolio summary for a user
    """
    from sqlalchemy import select, func
    
    stmt = select(
        func.sum(UserPortfolio.total_value).label("total_value"),
        func.avg(UserPortfolio.risk_score).label("avg_risk_score"),
        func.count(UserPortfolio.id).label("portfolio_count")
    ).where(UserPortfolio.user_id == user_id)
    
    result = await session.execute(stmt)
    row = result.first()
    
    return {
        "total_value": row.total_value or 0,
        "average_risk_score": row.avg_risk_score or 0,
        "portfolio_count": row.portfolio_count or 0
    } 


class Database:
    """
    Database management class
    """
    
    def __init__(self):
        self.engine = engine
        self.session = AsyncSessionLocal
    
    async def initialize(self):
        """
        Initialize database tables
        """
        await create_tables()
    
    async def close(self):
        """
        Close database connections
        """
        await self.engine.dispose()
    
    async def health_check(self) -> bool:
        """
        Check database health
        """
        try:
            async with self.session() as session:
                # Simple query to check connection
                await session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False


# Alias for dependency injection
async def get_db():
    """
    Database dependency for FastAPI
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close() 