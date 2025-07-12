"""
KYC/AML Compliance Service for StellarVault AI Engine

This service provides comprehensive compliance monitoring including:
- Automated KYC (Know Your Customer) verification
- AML (Anti-Money Laundering) transaction monitoring
- Risk scoring and customer due diligence
- Regulatory reporting and audit trails
- Sanctions screening and PEP checking
- Real-time transaction analysis
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from loguru import logger

from ..core.config import settings
from ..utils.data_sources import DataSourceClient


class KYCStatus(Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    REQUIRES_REVIEW = "requires_review"
    EXPIRED = "expired"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TransactionType(Enum):
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    TRANSFER = "transfer"
    TRADE = "trade"
    TOKENIZATION = "tokenization"
    REDEMPTION = "redemption"


class AlertType(Enum):
    SUSPICIOUS_TRANSACTION = "suspicious_transaction"
    VELOCITY_BREACH = "velocity_breach"
    SANCTIONS_MATCH = "sanctions_match"
    PEP_MATCH = "pep_match"
    THRESHOLD_BREACH = "threshold_breach"
    UNUSUAL_PATTERN = "unusual_pattern"
    JURISDICTION_RISK = "jurisdiction_risk"


@dataclass
class CustomerProfile:
    """Customer profile for KYC/AML"""
    customer_id: str
    first_name: str
    last_name: str
    date_of_birth: datetime
    nationality: str
    country_of_residence: str
    address: str
    phone: str
    email: str
    occupation: str
    income_source: str
    estimated_net_worth: Optional[float] = None
    business_name: Optional[str] = None
    business_type: Optional[str] = None
    is_pep: bool = False
    kyc_documents: List[str] = None
    kyc_status: KYCStatus = KYCStatus.PENDING
    risk_score: float = 0.0
    onboarding_date: datetime = None
    last_review_date: Optional[datetime] = None


@dataclass
class Transaction:
    """Transaction data for monitoring"""
    transaction_id: str
    customer_id: str
    transaction_type: TransactionType
    amount: float
    currency: str
    source_address: Optional[str] = None
    destination_address: Optional[str] = None
    source_country: Optional[str] = None
    destination_country: Optional[str] = None
    timestamp: datetime = None
    asset_type: Optional[str] = None
    counterparty_id: Optional[str] = None
    reference: Optional[str] = None


@dataclass
class ComplianceAlert:
    """Compliance alert"""
    alert_id: str
    customer_id: str
    transaction_id: Optional[str]
    alert_type: AlertType
    severity: RiskLevel
    description: str
    details: Dict[str, Any]
    created_at: datetime
    status: str = "open"  # open, investigating, closed, false_positive
    assigned_to: Optional[str] = None
    resolution_notes: Optional[str] = None


@dataclass
class ComplianceReport:
    """Compliance assessment report"""
    customer_id: str
    report_date: datetime
    kyc_status: KYCStatus
    aml_risk_score: float
    overall_risk_level: RiskLevel
    risk_factors: List[str]
    recommendations: List[str]
    alerts_summary: Dict[str, int]
    transaction_analysis: Dict[str, Any]
    next_review_date: datetime
    compliance_officer: str
    audit_trail: List[Dict[str, Any]]


class KYCAMLService:
    """
    Comprehensive KYC/AML compliance service
    """
    
    def __init__(self):
        self.data_client = DataSourceClient()
        self.sanctions_lists = set()  # In production, load from OFAC, UN, etc.
        self.pep_database = set()     # In production, load from commercial PEP database
        self.high_risk_countries = {
            "AF", "BY", "CF", "CD", "CU", "ER", "GN", "GW", "HT", "IR", 
            "IQ", "LB", "LY", "ML", "MM", "NI", "KP", "RU", "SO", "SS", 
            "SD", "SY", "UA", "VE", "YE", "ZW"
        }
        
    async def initialize(self):
        """Initialize the KYC/AML service"""
        try:
            await self._load_sanctions_lists()
            await self._load_pep_database()
            logger.info("KYC/AML service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize KYC/AML service: {e}")
            raise
    
    async def verify_customer(self, customer: CustomerProfile) -> ComplianceReport:
        """
        Perform comprehensive customer verification
        
        Args:
            customer: Customer profile data
            
        Returns:
            ComplianceReport with verification results
        """
        try:
            # Document verification
            document_score = await self._verify_documents(customer)
            
            # Identity verification
            identity_score = await self._verify_identity(customer)
            
            # Address verification
            address_score = await self._verify_address(customer)
            
            # Sanctions screening
            sanctions_result = await self._screen_sanctions(customer)
            
            # PEP screening
            pep_result = await self._screen_pep(customer)
            
            # Risk assessment
            risk_assessment = await self._assess_customer_risk(customer)
            
            # Calculate overall KYC status
            kyc_status = self._determine_kyc_status(
                document_score, identity_score, address_score,
                sanctions_result, pep_result, risk_assessment
            )
            
            # Generate recommendations
            recommendations = await self._generate_kyc_recommendations(
                customer, kyc_status, risk_assessment
            )
            
            # Create audit trail
            audit_entry = {
                "action": "customer_verification",
                "timestamp": datetime.now().isoformat(),
                "officer": "system",
                "details": {
                    "document_score": document_score,
                    "identity_score": identity_score,
                    "address_score": address_score,
                    "sanctions_match": sanctions_result['match'],
                    "pep_match": pep_result['match']
                }
            }
            
            return ComplianceReport(
                customer_id=customer.customer_id,
                report_date=datetime.now(),
                kyc_status=kyc_status,
                aml_risk_score=risk_assessment['aml_score'],
                overall_risk_level=risk_assessment['risk_level'],
                risk_factors=risk_assessment['risk_factors'],
                recommendations=recommendations,
                alerts_summary={"sanctions": 1 if sanctions_result['match'] else 0,
                              "pep": 1 if pep_result['match'] else 0},
                transaction_analysis={},
                next_review_date=self._calculate_next_review_date(kyc_status, risk_assessment['risk_level']),
                compliance_officer="system",
                audit_trail=[audit_entry]
            )
            
        except Exception as e:
            logger.error(f"Customer verification failed: {e}")
            raise
    
    async def monitor_transaction(self, transaction: Transaction) -> List[ComplianceAlert]:
        """
        Monitor transaction for AML compliance
        
        Args:
            transaction: Transaction to monitor
            
        Returns:
            List of compliance alerts if any
        """
        try:
            alerts = []
            
            # Amount-based monitoring
            amount_alerts = await self._check_amount_thresholds(transaction)
            alerts.extend(amount_alerts)
            
            # Velocity monitoring
            velocity_alerts = await self._check_transaction_velocity(transaction)
            alerts.extend(velocity_alerts)
            
            # Pattern analysis
            pattern_alerts = await self._analyze_transaction_patterns(transaction)
            alerts.extend(pattern_alerts)
            
            # Jurisdiction risk
            jurisdiction_alerts = await self._check_jurisdiction_risk(transaction)
            alerts.extend(jurisdiction_alerts)
            
            # Counterparty screening
            counterparty_alerts = await self._screen_counterparty(transaction)
            alerts.extend(counterparty_alerts)
            
            # Structuring detection
            structuring_alerts = await self._detect_structuring(transaction)
            alerts.extend(structuring_alerts)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Transaction monitoring failed: {e}")
            return []
    
    async def generate_sar_report(
        self, 
        customer_id: str, 
        suspicious_activity: str,
        transactions: List[Transaction]
    ) -> Dict[str, Any]:
        """
        Generate Suspicious Activity Report (SAR)
        
        Args:
            customer_id: Customer ID
            suspicious_activity: Description of suspicious activity
            transactions: Related transactions
            
        Returns:
            SAR report data
        """
        try:
            # Gather customer information
            customer_summary = await self._get_customer_summary(customer_id)
            
            # Analyze transaction patterns
            transaction_analysis = await self._analyze_suspicious_transactions(transactions)
            
            # Calculate total amounts
            total_amount = sum(t.amount for t in transactions)
            
            # Generate narrative
            narrative = await self._generate_sar_narrative(
                customer_summary, suspicious_activity, transaction_analysis
            )
            
            sar_report = {
                "report_id": f"SAR_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "filing_date": datetime.now().isoformat(),
                "customer_id": customer_id,
                "customer_summary": customer_summary,
                "suspicious_activity": suspicious_activity,
                "transaction_count": len(transactions),
                "total_amount": total_amount,
                "currency": transactions[0].currency if transactions else "USD",
                "date_range": {
                    "start": min(t.timestamp for t in transactions).isoformat(),
                    "end": max(t.timestamp for t in transactions).isoformat()
                },
                "transaction_analysis": transaction_analysis,
                "narrative": narrative,
                "filing_institution": "StellarVault",
                "contact_person": "Compliance Officer",
                "status": "filed"
            }
            
            return sar_report
            
        except Exception as e:
            logger.error(f"SAR report generation failed: {e}")
            raise
    
    async def _verify_documents(self, customer: CustomerProfile) -> float:
        """Verify customer documents"""
        try:
            score = 0.0
            
            if customer.kyc_documents:
                # Document presence check
                required_docs = {"id_document", "proof_of_address"}
                provided_docs = set(customer.kyc_documents)
                
                if required_docs.issubset(provided_docs):
                    score += 0.5
                
                # Document quality check (simulated)
                # In production, would use ML models for document analysis
                if len(customer.kyc_documents) >= 2:
                    score += 0.3
                
                # Additional documents
                if len(customer.kyc_documents) > 2:
                    score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Document verification failed: {e}")
            return 0.0
    
    async def _verify_identity(self, customer: CustomerProfile) -> float:
        """Verify customer identity"""
        try:
            score = 0.0
            
            # Name verification (simulated)
            if customer.first_name and customer.last_name:
                if len(customer.first_name) > 1 and len(customer.last_name) > 1:
                    score += 0.3
            
            # Date of birth verification
            if customer.date_of_birth:
                age = (datetime.now() - customer.date_of_birth).days / 365.25
                if 18 <= age <= 120:  # Reasonable age range
                    score += 0.3
            
            # Contact information verification
            if customer.email and "@" in customer.email:
                score += 0.2
            
            if customer.phone and len(customer.phone) >= 10:
                score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Identity verification failed: {e}")
            return 0.0
    
    async def _verify_address(self, customer: CustomerProfile) -> float:
        """Verify customer address"""
        try:
            score = 0.0
            
            if customer.address:
                # Basic address format check
                if len(customer.address) > 10:
                    score += 0.4
                
                # Country verification
                if customer.country_of_residence and len(customer.country_of_residence) == 2:
                    score += 0.3
                
                # Address consistency (simplified)
                # In production, would use address validation services
                score += 0.3
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Address verification failed: {e}")
            return 0.0
    
    async def _screen_sanctions(self, customer: CustomerProfile) -> Dict[str, Any]:
        """Screen customer against sanctions lists"""
        try:
            # Name matching (simplified fuzzy matching)
            full_name = f"{customer.first_name} {customer.last_name}".lower()
            
            # Check against sanctions lists
            for sanctioned_name in self.sanctions_lists:
                similarity = self._calculate_name_similarity(full_name, sanctioned_name.lower())
                if similarity > 0.85:  # High similarity threshold
                    return {
                        "match": True,
                        "matched_name": sanctioned_name,
                        "similarity": similarity,
                        "list": "OFAC"
                    }
            
            # Check country of residence
            if customer.country_of_residence in self.high_risk_countries:
                return {
                    "match": True,
                    "matched_name": None,
                    "similarity": 1.0,
                    "list": "High Risk Country"
                }
            
            return {"match": False, "matched_name": None, "similarity": 0.0, "list": None}
            
        except Exception as e:
            logger.error(f"Sanctions screening failed: {e}")
            return {"match": False, "matched_name": None, "similarity": 0.0, "list": None}
    
    async def _screen_pep(self, customer: CustomerProfile) -> Dict[str, Any]:
        """Screen customer against PEP database"""
        try:
            if customer.is_pep:
                return {
                    "match": True,
                    "pep_type": "Self-declared",
                    "position": customer.occupation,
                    "risk_level": "HIGH"
                }
            
            # Name matching against PEP database
            full_name = f"{customer.first_name} {customer.last_name}".lower()
            
            for pep_name in self.pep_database:
                similarity = self._calculate_name_similarity(full_name, pep_name.lower())
                if similarity > 0.90:  # Very high similarity for PEP
                    return {
                        "match": True,
                        "pep_type": "Database match",
                        "position": "Unknown",
                        "risk_level": "HIGH"
                    }
            
            # Occupation-based PEP check
            high_risk_occupations = {
                "politician", "government official", "judge", "military officer",
                "central banker", "diplomat", "state enterprise executive"
            }
            
            if any(occupation in customer.occupation.lower() 
                   for occupation in high_risk_occupations):
                return {
                    "match": True,
                    "pep_type": "Occupation-based",
                    "position": customer.occupation,
                    "risk_level": "MEDIUM"
                }
            
            return {"match": False, "pep_type": None, "position": None, "risk_level": "LOW"}
            
        except Exception as e:
            logger.error(f"PEP screening failed: {e}")
            return {"match": False, "pep_type": None, "position": None, "risk_level": "LOW"}
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names using Levenshtein distance"""
        try:
            def levenshtein_distance(s1, s2):
                if len(s1) < len(s2):
                    return levenshtein_distance(s2, s1)
                
                if len(s2) == 0:
                    return len(s1)
                
                previous_row = list(range(len(s2) + 1))
                for i, c1 in enumerate(s1):
                    current_row = [i + 1]
                    for j, c2 in enumerate(s2):
                        insertions = previous_row[j + 1] + 1
                        deletions = current_row[j] + 1
                        substitutions = previous_row[j] + (c1 != c2)
                        current_row.append(min(insertions, deletions, substitutions))
                    previous_row = current_row
                
                return previous_row[-1]
            
            distance = levenshtein_distance(name1, name2)
            max_len = max(len(name1), len(name2))
            
            if max_len == 0:
                return 1.0
            
            similarity = 1 - (distance / max_len)
            return similarity
            
        except Exception as e:
            logger.error(f"Name similarity calculation failed: {e}")
            return 0.0
    
    async def _assess_customer_risk(self, customer: CustomerProfile) -> Dict[str, Any]:
        """Assess overall customer risk"""
        try:
            risk_score = 0.0
            risk_factors = []
            
            # Country risk
            if customer.country_of_residence in self.high_risk_countries:
                risk_score += 30
                risk_factors.append("High-risk jurisdiction")
            
            # Occupation risk
            high_risk_occupations = {
                "money changer", "casino", "crypto exchange", "remittance",
                "pawn shop", "art dealer", "precious metals", "cash intensive"
            }
            
            if any(occupation in customer.occupation.lower() 
                   for occupation in high_risk_occupations):
                risk_score += 20
                risk_factors.append("High-risk occupation")
            
            # PEP status
            if customer.is_pep:
                risk_score += 25
                risk_factors.append("Politically Exposed Person")
            
            # Age factor
            if customer.date_of_birth:
                age = (datetime.now() - customer.date_of_birth).days / 365.25
                if age < 25 or age > 80:
                    risk_score += 10
                    risk_factors.append("Age-related risk")
            
            # Income vs net worth consistency
            if (customer.estimated_net_worth and 
                customer.estimated_net_worth > 1000000):  # High net worth
                risk_score += 15
                risk_factors.append("High net worth individual")
            
            # Determine risk level
            if risk_score < 25:
                risk_level = RiskLevel.LOW
            elif risk_score < 50:
                risk_level = RiskLevel.MEDIUM
            elif risk_score < 75:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.CRITICAL
            
            return {
                "aml_score": risk_score,
                "risk_level": risk_level,
                "risk_factors": risk_factors
            }
            
        except Exception as e:
            logger.error(f"Customer risk assessment failed: {e}")
            return {
                "aml_score": 50.0,
                "risk_level": RiskLevel.MEDIUM,
                "risk_factors": ["Assessment error"]
            }
    
    def _determine_kyc_status(
        self,
        document_score: float,
        identity_score: float,
        address_score: float,
        sanctions_result: Dict[str, Any],
        pep_result: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> KYCStatus:
        """Determine overall KYC status"""
        try:
            # Automatic rejection conditions
            if sanctions_result['match'] and sanctions_result['similarity'] > 0.90:
                return KYCStatus.REJECTED
            
            if risk_assessment['risk_level'] == RiskLevel.CRITICAL:
                return KYCStatus.REQUIRES_REVIEW
            
            # Minimum score requirements
            total_score = (document_score + identity_score + address_score) / 3
            
            if total_score < 0.6:
                return KYCStatus.REJECTED
            elif total_score < 0.8:
                return KYCStatus.REQUIRES_REVIEW
            elif pep_result['match']:
                return KYCStatus.REQUIRES_REVIEW
            elif risk_assessment['risk_level'] == RiskLevel.HIGH:
                return KYCStatus.REQUIRES_REVIEW
            else:
                return KYCStatus.VERIFIED
                
        except Exception as e:
            logger.error(f"KYC status determination failed: {e}")
            return KYCStatus.REQUIRES_REVIEW
    
    async def _check_amount_thresholds(self, transaction: Transaction) -> List[ComplianceAlert]:
        """Check transaction amount against thresholds"""
        try:
            alerts = []
            
            # Currency Transaction Report (CTR) threshold
            if transaction.amount >= 10000:  # $10,000 USD threshold
                alert = ComplianceAlert(
                    alert_id=f"CTR_{transaction.transaction_id}",
                    customer_id=transaction.customer_id,
                    transaction_id=transaction.transaction_id,
                    alert_type=AlertType.THRESHOLD_BREACH,
                    severity=RiskLevel.MEDIUM,
                    description=f"Transaction amount ${transaction.amount:,.2f} exceeds CTR threshold",
                    details={
                        "threshold": 10000,
                        "amount": transaction.amount,
                        "currency": transaction.currency
                    },
                    created_at=datetime.now()
                )
                alerts.append(alert)
            
            # Large transaction threshold
            if transaction.amount >= 50000:  # $50,000 threshold
                alert = ComplianceAlert(
                    alert_id=f"LARGE_{transaction.transaction_id}",
                    customer_id=transaction.customer_id,
                    transaction_id=transaction.transaction_id,
                    alert_type=AlertType.THRESHOLD_BREACH,
                    severity=RiskLevel.HIGH,
                    description=f"Large transaction amount ${transaction.amount:,.2f}",
                    details={
                        "threshold": 50000,
                        "amount": transaction.amount,
                        "currency": transaction.currency
                    },
                    created_at=datetime.now()
                )
                alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Amount threshold check failed: {e}")
            return []
    
    async def _check_transaction_velocity(self, transaction: Transaction) -> List[ComplianceAlert]:
        """Check transaction velocity patterns"""
        try:
            alerts = []
            
            # Simulate velocity check (in production, would query transaction history)
            # Daily velocity threshold
            daily_limit = 25000  # $25,000 daily limit
            
            # Weekly velocity threshold  
            weekly_limit = 100000  # $100,000 weekly limit
            
            # Simulated current totals (in production, calculate from DB)
            daily_total = transaction.amount  # Simplified
            weekly_total = transaction.amount  # Simplified
            
            if daily_total > daily_limit:
                alert = ComplianceAlert(
                    alert_id=f"VEL_DAILY_{transaction.customer_id}",
                    customer_id=transaction.customer_id,
                    transaction_id=transaction.transaction_id,
                    alert_type=AlertType.VELOCITY_BREACH,
                    severity=RiskLevel.HIGH,
                    description=f"Daily transaction velocity ${daily_total:,.2f} exceeds limit",
                    details={
                        "period": "daily",
                        "limit": daily_limit,
                        "total": daily_total
                    },
                    created_at=datetime.now()
                )
                alerts.append(alert)
            
            if weekly_total > weekly_limit:
                alert = ComplianceAlert(
                    alert_id=f"VEL_WEEKLY_{transaction.customer_id}",
                    customer_id=transaction.customer_id,
                    transaction_id=transaction.transaction_id,
                    alert_type=AlertType.VELOCITY_BREACH,
                    severity=RiskLevel.CRITICAL,
                    description=f"Weekly transaction velocity ${weekly_total:,.2f} exceeds limit",
                    details={
                        "period": "weekly",
                        "limit": weekly_limit,
                        "total": weekly_total
                    },
                    created_at=datetime.now()
                )
                alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Velocity check failed: {e}")
            return []
    
    async def _analyze_transaction_patterns(self, transaction: Transaction) -> List[ComplianceAlert]:
        """Analyze transaction for unusual patterns"""
        try:
            alerts = []
            
            # Round number detection (potential structuring)
            if transaction.amount % 1000 == 0 and transaction.amount >= 5000:
                alert = ComplianceAlert(
                    alert_id=f"ROUND_{transaction.transaction_id}",
                    customer_id=transaction.customer_id,
                    transaction_id=transaction.transaction_id,
                    alert_type=AlertType.UNUSUAL_PATTERN,
                    severity=RiskLevel.MEDIUM,
                    description="Round number transaction may indicate structuring",
                    details={
                        "pattern": "round_number",
                        "amount": transaction.amount
                    },
                    created_at=datetime.now()
                )
                alerts.append(alert)
            
            # Unusual time patterns (simulated)
            current_hour = datetime.now().hour
            if current_hour < 6 or current_hour > 22:  # Outside normal hours
                alert = ComplianceAlert(
                    alert_id=f"TIME_{transaction.transaction_id}",
                    customer_id=transaction.customer_id,
                    transaction_id=transaction.transaction_id,
                    alert_type=AlertType.UNUSUAL_PATTERN,
                    severity=RiskLevel.LOW,
                    description="Transaction outside normal business hours",
                    details={
                        "pattern": "unusual_time",
                        "hour": current_hour
                    },
                    created_at=datetime.now()
                )
                alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return []
    
    async def _check_jurisdiction_risk(self, transaction: Transaction) -> List[ComplianceAlert]:
        """Check jurisdiction risk for transaction"""
        try:
            alerts = []
            
            # Check source and destination countries
            risky_countries = [transaction.source_country, transaction.destination_country]
            
            for country in risky_countries:
                if country and country in self.high_risk_countries:
                    alert = ComplianceAlert(
                        alert_id=f"JURIS_{transaction.transaction_id}_{country}",
                        customer_id=transaction.customer_id,
                        transaction_id=transaction.transaction_id,
                        alert_type=AlertType.JURISDICTION_RISK,
                        severity=RiskLevel.HIGH,
                        description=f"Transaction involves high-risk jurisdiction: {country}",
                        details={
                            "country": country,
                            "risk_category": "high_risk_jurisdiction"
                        },
                        created_at=datetime.now()
                    )
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Jurisdiction risk check failed: {e}")
            return []
    
    async def _screen_counterparty(self, transaction: Transaction) -> List[ComplianceAlert]:
        """Screen transaction counterparty"""
        try:
            alerts = []
            
            if transaction.counterparty_id:
                # Simulate counterparty screening
                # In production, would check against sanctions lists and internal blacklists
                
                # Check if counterparty is on internal watch list (simulated)
                watch_list = {"WATCH001", "WATCH002", "BLOCKED001"}
                
                if transaction.counterparty_id in watch_list:
                    alert = ComplianceAlert(
                        alert_id=f"COUNTER_{transaction.transaction_id}",
                        customer_id=transaction.customer_id,
                        transaction_id=transaction.transaction_id,
                        alert_type=AlertType.SANCTIONS_MATCH,
                        severity=RiskLevel.CRITICAL,
                        description=f"Transaction with watch-listed counterparty: {transaction.counterparty_id}",
                        details={
                            "counterparty_id": transaction.counterparty_id,
                            "list_type": "internal_watch_list"
                        },
                        created_at=datetime.now()
                    )
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Counterparty screening failed: {e}")
            return []
    
    async def _detect_structuring(self, transaction: Transaction) -> List[ComplianceAlert]:
        """Detect potential structuring patterns"""
        try:
            alerts = []
            
            # Just-under threshold amounts
            structuring_thresholds = [10000, 3000, 5000]
            
            for threshold in structuring_thresholds:
                if (threshold - 500) <= transaction.amount < threshold:
                    alert = ComplianceAlert(
                        alert_id=f"STRUCT_{transaction.transaction_id}",
                        customer_id=transaction.customer_id,
                        transaction_id=transaction.transaction_id,
                        alert_type=AlertType.SUSPICIOUS_TRANSACTION,
                        severity=RiskLevel.HIGH,
                        description=f"Potential structuring: amount ${transaction.amount:,.2f} just under ${threshold:,.2f} threshold",
                        details={
                            "amount": transaction.amount,
                            "threshold": threshold,
                            "pattern": "just_under_threshold"
                        },
                        created_at=datetime.now()
                    )
                    alerts.append(alert)
                    break
            
            return alerts
            
        except Exception as e:
            logger.error(f"Structuring detection failed: {e}")
            return []
    
    async def _load_sanctions_lists(self):
        """Load sanctions lists (simulated)"""
        try:
            # In production, would load from OFAC, UN, EU, etc.
            self.sanctions_lists = {
                "Osama bin Laden", "Al-Qaeda", "Taliban", "ISIL",
                "Drug Cartel Leader", "Terror Organization"
            }
            logger.info(f"Loaded {len(self.sanctions_lists)} sanctions entries")
            
        except Exception as e:
            logger.error(f"Failed to load sanctions lists: {e}")
    
    async def _load_pep_database(self):
        """Load PEP database (simulated)"""
        try:
            # In production, would load from commercial PEP database
            self.pep_database = {
                "Vladimir Putin", "Xi Jinping", "Joe Biden", "Emmanuel Macron",
                "Prime Minister", "Central Bank Governor", "Supreme Court Justice"
            }
            logger.info(f"Loaded {len(self.pep_database)} PEP entries")
            
        except Exception as e:
            logger.error(f"Failed to load PEP database: {e}")
    
    async def _generate_kyc_recommendations(
        self,
        customer: CustomerProfile,
        kyc_status: KYCStatus,
        risk_assessment: Dict[str, Any]
    ) -> List[str]:
        """Generate KYC recommendations"""
        try:
            recommendations = []
            
            if kyc_status == KYCStatus.REQUIRES_REVIEW:
                recommendations.append("Manual review required before account activation")
                
            if risk_assessment['risk_level'] in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                recommendations.append("Enhanced due diligence procedures required")
                recommendations.append("Obtain additional documentation for income/wealth source")
                
            if customer.is_pep:
                recommendations.append("Apply enhanced monitoring for PEP customer")
                recommendations.append("Obtain senior management approval")
                
            if customer.country_of_residence in self.high_risk_countries:
                recommendations.append("Additional verification for high-risk jurisdiction")
                
            if kyc_status == KYCStatus.VERIFIED:
                recommendations.append("Customer approved for standard monitoring")
                
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return ["Standard KYC procedures apply"]
    
    def _calculate_next_review_date(self, kyc_status: KYCStatus, risk_level: RiskLevel) -> datetime:
        """Calculate next review date based on risk"""
        try:
            if risk_level == RiskLevel.CRITICAL:
                months = 3  # Quarterly review
            elif risk_level == RiskLevel.HIGH:
                months = 6  # Semi-annual review
            elif risk_level == RiskLevel.MEDIUM:
                months = 12  # Annual review
            else:
                months = 24  # Biennial review
            
            return datetime.now() + timedelta(days=months * 30)
            
        except Exception as e:
            logger.error(f"Next review date calculation failed: {e}")
            return datetime.now() + timedelta(days=365)
    
    async def _get_customer_summary(self, customer_id: str) -> Dict[str, Any]:
        """Get customer summary for reporting"""
        try:
            # In production, would query customer database
            return {
                "customer_id": customer_id,
                "name": "Customer Name",
                "account_type": "Individual",
                "onboarding_date": "2024-01-01",
                "risk_level": "MEDIUM",
                "jurisdiction": "US"
            }
            
        except Exception as e:
            logger.error(f"Customer summary failed: {e}")
            return {"customer_id": customer_id, "name": "Unknown"}
    
    async def _analyze_suspicious_transactions(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Analyze suspicious transaction patterns"""
        try:
            total_amount = sum(t.amount for t in transactions)
            transaction_count = len(transactions)
            
            # Pattern analysis
            patterns = []
            if transaction_count > 5:
                patterns.append("High frequency transactions")
            
            amounts = [t.amount for t in transactions]
            if len(set(amounts)) == 1:  # All same amount
                patterns.append("Identical transaction amounts")
            
            return {
                "total_amount": total_amount,
                "transaction_count": transaction_count,
                "patterns": patterns,
                "time_span_days": (max(t.timestamp for t in transactions) - 
                                 min(t.timestamp for t in transactions)).days
            }
            
        except Exception as e:
            logger.error(f"Suspicious transaction analysis failed: {e}")
            return {"total_amount": 0, "transaction_count": 0, "patterns": []}
    
    async def _generate_sar_narrative(
        self,
        customer_summary: Dict[str, Any],
        suspicious_activity: str,
        transaction_analysis: Dict[str, Any]
    ) -> str:
        """Generate SAR narrative"""
        try:
            narrative = f"""
SUSPICIOUS ACTIVITY REPORT

Customer Information:
- Customer ID: {customer_summary.get('customer_id', 'Unknown')}
- Name: {customer_summary.get('name', 'Unknown')}
- Account Type: {customer_summary.get('account_type', 'Unknown')}

Suspicious Activity Description:
{suspicious_activity}

Transaction Analysis:
- Total Amount: ${transaction_analysis.get('total_amount', 0):,.2f}
- Number of Transactions: {transaction_analysis.get('transaction_count', 0)}
- Time Period: {transaction_analysis.get('time_span_days', 0)} days
- Identified Patterns: {', '.join(transaction_analysis.get('patterns', []))}

This activity was flagged by our automated monitoring system for further investigation.
The customer's transaction patterns deviated significantly from their established profile.
            """.strip()
            
            return narrative
            
        except Exception as e:
            logger.error(f"SAR narrative generation failed: {e}")
            return "Suspicious activity detected requiring further investigation."


# Export the service
__all__ = [
    'KYCAMLService', 'CustomerProfile', 'Transaction', 'ComplianceAlert',
    'ComplianceReport', 'KYCStatus', 'RiskLevel', 'TransactionType', 'AlertType'
] 