"""
Configuration management for StellarVault AI Engine
"""
import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    """
    
    # Application settings
    APP_NAME: str = "StellarVault AI Engine"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=True, env="DEBUG")
    
    # Server settings
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=4, env="WORKERS")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Database settings
    DATABASE_URL: str = Field(default="sqlite+aiosqlite:///./stellarvault.db", env="DATABASE_URL")
    DB_POOL_SIZE: int = Field(default=10, env="DB_POOL_SIZE")
    DB_MAX_OVERFLOW: int = Field(default=20, env="DB_MAX_OVERFLOW")
    
    # Redis settings
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    REDIS_TTL: int = Field(default=3600, env="REDIS_TTL")  # 1 hour
    
    # Security settings
    SECRET_KEY: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    ALGORITHM: str = Field(default="HS256", env="ALGORITHM")
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:3000",
            "https://stellarvault.com",
            "https://www.stellarvault.com"
        ],
        env="ALLOWED_ORIGINS"
    )
    
    # API Keys for external services (optional for development)
    ALPHA_VANTAGE_API_KEY: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_API_KEY")
    POLYGON_API_KEY: Optional[str] = Field(default=None, env="POLYGON_API_KEY")
    NEWS_API_KEY: Optional[str] = Field(default=None, env="NEWS_API_KEY")
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    # Real Estate APIs
    ZILLOW_API_KEY: Optional[str] = Field(default=None, env="ZILLOW_API_KEY")
    REALTOR_API_KEY: Optional[str] = Field(default=None, env="REALTOR_API_KEY")
    PROPERTY_DATA_API_KEY: Optional[str] = Field(default=None, env="PROPERTY_DATA_API_KEY")
    
    # Commodities and Financial Data
    QUANDL_API_KEY: Optional[str] = Field(default=None, env="QUANDL_API_KEY")
    FRED_API_KEY: Optional[str] = Field(default=None, env="FRED_API_KEY")
    
    # Art and Collectibles
    ARTSY_API_KEY: Optional[str] = Field(default=None, env="ARTSY_API_KEY")
    ARTNET_API_KEY: Optional[str] = Field(default=None, env="ARTNET_API_KEY")
    
    # Stellar Network
    STELLAR_NETWORK: str = Field(default="testnet", env="STELLAR_NETWORK")
    STELLAR_HORIZON_URL: str = Field(
        default="https://horizon-testnet.stellar.org",
        env="STELLAR_HORIZON_URL"
    )
    STELLAR_PASSPHRASE: str = Field(
        default="Test SDF Network ; September 2015",
        env="STELLAR_PASSPHRASE"
    )
    STELLAR_SECRET_KEY: Optional[str] = Field(default=None, env="STELLAR_SECRET_KEY")
    
    # ML Model settings
    MODEL_PATH: str = Field(default="./models", env="MODEL_PATH")
    MODEL_UPDATE_INTERVAL: int = Field(default=86400, env="MODEL_UPDATE_INTERVAL")  # 24 hours
    BATCH_SIZE: int = Field(default=32, env="BATCH_SIZE")
    
    # Data collection settings
    DATA_COLLECTION_INTERVAL: int = Field(default=3600, env="DATA_COLLECTION_INTERVAL")  # 1 hour
    MAX_RETRIES: int = Field(default=3, env="MAX_RETRIES")
    REQUEST_TIMEOUT: int = Field(default=30, env="REQUEST_TIMEOUT")
    
    # Monitoring and alerting
    SENTRY_DSN: Optional[str] = Field(default=None, env="SENTRY_DSN")
    PROMETHEUS_PORT: int = Field(default=8001, env="PROMETHEUS_PORT")
    
    # Rate limiting
    RATE_LIMIT_REQUESTS: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # 1 minute
    
    # Compliance settings (optional for development)
    KYC_PROVIDER: str = Field(default="jumio", env="KYC_PROVIDER")
    KYC_API_KEY: Optional[str] = Field(default=None, env="KYC_API_KEY")
    COMPLIANCE_WEBHOOK_URL: Optional[str] = Field(default=None, env="COMPLIANCE_WEBHOOK_URL")
    
    # Valuation model parameters
    REAL_ESTATE_MODEL_THRESHOLD: float = Field(default=0.95, env="REAL_ESTATE_MODEL_THRESHOLD")
    COMMODITIES_MODEL_THRESHOLD: float = Field(default=0.90, env="COMMODITIES_MODEL_THRESHOLD")
    ART_MODEL_THRESHOLD: float = Field(default=0.85, env="ART_MODEL_THRESHOLD")
    BONDS_MODEL_THRESHOLD: float = Field(default=0.98, env="BONDS_MODEL_THRESHOLD")
    
    # Risk assessment parameters
    RISK_WINDOW_DAYS: int = Field(default=252, env="RISK_WINDOW_DAYS")  # 1 year
    MONTE_CARLO_SIMULATIONS: int = Field(default=10000, env="MONTE_CARLO_SIMULATIONS")
    CONFIDENCE_LEVEL: float = Field(default=0.95, env="CONFIDENCE_LEVEL")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings() 