"""
OP-ECOM Analytics Tracker - Pydantic Schemas
Request/Response validation models
"""

from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime
from enum import Enum

class VisitorTypeEnum(str, Enum):
    New_Visitor = "New_Visitor"
    Returning_Visitor = "Returning_Visitor"
    Other = "Other"

class PageTypeEnum(str, Enum):
    Administrative = "Administrative"
    Informational = "Informational"
    ProductRelated = "ProductRelated"

# --- Session Schemas ---
class SessionStartRequest(BaseModel):
    visitor_type: VisitorTypeEnum = VisitorTypeEnum.New_Visitor
    browser: Optional[str] = None
    operating_system: Optional[str] = None
    region: int = 1
    traffic_type: int = 1

class SessionStartResponse(BaseModel):
    session_id: str
    message: str = "Session started"

class SessionEndRequest(BaseModel):
    session_id: str

# --- PageView Schemas ---
class PageViewRequest(BaseModel):
    session_id: str
    page_type: PageTypeEnum = PageTypeEnum.ProductRelated
    page_url: str
    page_title: Optional[str] = None
    duration_seconds: float = 0.0
    is_bounce: bool = False
    is_exit: bool = False
    page_value: float = 0.0
    scroll_depth: int = 0

# --- Event Schemas ---
class EventRequest(BaseModel):
    session_id: str
    event_type: str
    event_category: Optional[str] = None
    event_label: Optional[str] = None
    event_value: float = 0.0
    event_data: Optional[Any] = None

# --- Purchase Schema ---
class PurchaseRequest(BaseModel):
    session_id: str
    order_value: float

class IntentCheckRequest(BaseModel):
    session_id: str
    cart_value: float = 0.0  # NEW: For personalized discount calculation

class IntentCheckResponse(BaseModel):
    probability: float
    should_intervene: bool
    abandonment_prob: float = 0.0
    purchase_prob: float = 0.0
    discount_percent: int = 20  # NEW: Personalized discount

# --- Email Capture Schemas ---
class EmailCaptureRequest(BaseModel):
    session_id: str
    email: str
    cart_value: float = 0.0

class EmailCaptureResponse(BaseModel):
    success: bool
    discount_code: str
    discount_percent: int
    message: str = "Your exclusive discount code!"

# --- Admin Schemas ---
class SessionSummary(BaseModel):
    session_id: str
    visitor_type: str
    started_at: datetime
    ended_at: Optional[datetime]
    revenue: bool
    page_count: int = 0
    
    class Config:
        from_attributes = True
