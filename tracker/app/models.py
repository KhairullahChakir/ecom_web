"""
OP-ECOM Analytics Tracker - ORM Models
SQLAlchemy models matching the MariaDB schema
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, Enum, Text, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from .database import Base
import enum

class VisitorType(str, enum.Enum):
    New_Visitor = "New_Visitor"
    Returning_Visitor = "Returning_Visitor"
    Other = "Other"

class PageType(str, enum.Enum):
    Administrative = "Administrative"
    Informational = "Informational"
    ProductRelated = "ProductRelated"

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True)
    visitor_type = Column(Enum(VisitorType), default=VisitorType.New_Visitor)
    browser = Column(String(50))
    operating_system = Column(String(50))
    region = Column(Integer, default=1)
    traffic_type = Column(Integer, default=1)
    is_weekend = Column(Boolean, default=False)
    month = Column(String(10))
    special_day = Column(Float, default=0.0)
    started_at = Column(DateTime, server_default=func.now())
    ended_at = Column(DateTime, nullable=True)
    revenue = Column(Boolean, default=False)
    
    # Aggregated metrics
    administrative_count = Column(Integer, default=0)
    administrative_duration = Column(Float, default=0.0)
    informational_count = Column(Integer, default=0)
    informational_duration = Column(Float, default=0.0)
    product_related_count = Column(Integer, default=0)
    product_related_duration = Column(Float, default=0.0)
    bounce_rates = Column(Float, default=0.0)
    exit_rates = Column(Float, default=0.0)
    page_values = Column(Float, default=0.0)

class PageView(Base):
    __tablename__ = "page_views"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    page_type = Column(Enum(PageType), default=PageType.ProductRelated)
    page_url = Column(String(500))
    page_title = Column(String(255))
    duration_seconds = Column(Float, default=0.0)
    is_bounce = Column(Boolean, default=False)
    is_exit = Column(Boolean, default=False)
    page_value = Column(Float, default=0.0)
    scroll_depth = Column(Integer, default=0)
    viewed_at = Column(DateTime, server_default=func.now())

class Event(Base):
    __tablename__ = "events"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    event_category = Column(String(50))
    event_label = Column(String(255))
    event_value = Column(Float, default=0.0)
    event_data = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())


class EmailCapture(Base):
    """Stores captured emails from exit-intent popups"""
    __tablename__ = "email_captures"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    email = Column(String(255), nullable=False, index=True)
    discount_code = Column(String(20), nullable=False)
    discount_percent = Column(Integer, default=20)
    cart_value = Column(Float, default=0.0)
    used = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())
