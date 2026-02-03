"""
OP-ECOM Analytics Tracker - FastAPI Router
Endpoints for tracking sessions, page views, and events
"""

import uuid
import requests
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import func

from .database import get_db
from .models import Session, PageView, Event, VisitorType, PageType
from .schemas import (
    SessionStartRequest, SessionStartResponse, SessionEndRequest,
    PageViewRequest, EventRequest, PurchaseRequest,
    IntentCheckRequest, IntentCheckResponse
)

PREDICTION_API_URL = "http://localhost:8000/predict"

router = APIRouter(prefix="/tracker", tags=["Tracker"])

def get_month_name() -> str:
    """Get current month as short name (mapped to Nov for Demo calibration)"""
    # Nov has the highest intent signal in the UCI dataset, making the demo more responsive.
    return "Nov"

def is_weekend() -> bool:
    """Check if today is weekend"""
    return datetime.now().weekday() >= 5

# --- Session Endpoints ---
@router.post("/session/start", response_model=SessionStartResponse)
async def start_session(request: SessionStartRequest, db: DBSession = Depends(get_db)):
    """Start a new tracking session"""
    session_id = str(uuid.uuid4())
    
    new_session = Session(
        session_id=session_id,
        visitor_type=VisitorType[request.visitor_type.value],
        browser=request.browser,
        operating_system=request.operating_system,
        region=request.region,
        traffic_type=request.traffic_type,
        is_weekend=is_weekend(),
        month=get_month_name()
    )
    
    db.add(new_session)
    db.commit()
    
    return SessionStartResponse(session_id=session_id)

@router.post("/session/end")
async def end_session(request: SessionEndRequest, db: DBSession = Depends(get_db)):
    """End a tracking session and calculate aggregated metrics"""
    session = db.query(Session).filter(Session.session_id == request.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get all page views for this session
    page_views = db.query(PageView).filter(PageView.session_id == request.session_id).all()
    
    # Calculate aggregated metrics
    admin_pages = [p for p in page_views if p.page_type == PageType.Administrative]
    info_pages = [p for p in page_views if p.page_type == PageType.Informational]
    product_pages = [p for p in page_views if p.page_type == PageType.ProductRelated]
    
    session.administrative_count = len(admin_pages)
    session.administrative_duration = sum(p.duration_seconds for p in admin_pages)
    session.informational_count = len(info_pages)
    session.informational_duration = sum(p.duration_seconds for p in info_pages)
    session.product_related_count = len(product_pages)
    session.product_related_duration = sum(p.duration_seconds for p in product_pages)
    
    # Calculate bounce and exit rates
    total_pages = len(page_views)
    if total_pages > 0:
        bounce_count = sum(1 for p in page_views if p.is_bounce)
        exit_count = sum(1 for p in page_views if p.is_exit)
        session.bounce_rates = bounce_count / total_pages
        session.exit_rates = exit_count / total_pages
        session.page_values = sum(p.page_value for p in page_views) / total_pages
    
    session.ended_at = datetime.now()
    db.commit()
    
    return {"message": "Session ended", "session_id": request.session_id}

# --- Page View Endpoints ---
@router.post("/pageview")
async def track_pageview(request: PageViewRequest, db: DBSession = Depends(get_db)):
    """Track a page view"""
    # Verify session exists
    session = db.query(Session).filter(Session.session_id == request.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Mark previous page as not exit
    db.query(PageView).filter(
        PageView.session_id == request.session_id,
        PageView.is_exit == True
    ).update({"is_exit": False})
    
    new_pageview = PageView(
        session_id=request.session_id,
        page_type=PageType[request.page_type.value],
        page_url=request.page_url,
        page_title=request.page_title,
        duration_seconds=request.duration_seconds,
        is_bounce=request.is_bounce,
        is_exit=True,  # Latest page is always exit until next one
        page_value=request.page_value,
        scroll_depth=request.scroll_depth
    )
    
    db.add(new_pageview)
    db.commit()
    
    return {"message": "Page view tracked", "page_id": new_pageview.id}

# --- Event Endpoints ---
@router.post("/event")
async def track_event(request: EventRequest, db: DBSession = Depends(get_db)):
    """Track a custom event"""
    session = db.query(Session).filter(Session.session_id == request.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    new_event = Event(
        session_id=request.session_id,
        event_type=request.event_type,
        event_category=request.event_category,
        event_label=request.event_label,
        event_value=request.event_value,
        event_data=request.event_data
    )
    
    db.add(new_event)
    db.commit()
    
    return {"message": "Event tracked", "event_id": new_event.id}

# --- Purchase Endpoint ---
@router.post("/purchase")
async def track_purchase(request: PurchaseRequest, db: DBSession = Depends(get_db)):
    """Mark session as converted (revenue=True)"""
    session = db.query(Session).filter(Session.session_id == request.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session.revenue = True
    
    # Also log as event
    purchase_event = Event(
        session_id=request.session_id,
        event_type="purchase",
        event_category="conversion",
        event_value=request.order_value,
        event_data={"order_value": request.order_value}
    )
    
    db.add(purchase_event)
    db.commit()
    
    return {"message": "Purchase recorded", "session_id": request.session_id}

# --- Intent Check Endpoint (Exit Intent) ---
@router.post("/check-intent", response_model=IntentCheckResponse)
async def check_intent(request: IntentCheckRequest, db: DBSession = Depends(get_db)):
    """Check purchase intent for exit intervention"""
    session = db.query(Session).filter(Session.session_id == request.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # 1. Aggregate current session stats
    page_views = db.query(PageView).filter(PageView.session_id == request.session_id).all()
    
    product_pages = [p for p in page_views if p.page_type == PageType.ProductRelated]
    admin_pages = [p for p in page_views if p.page_type == PageType.Administrative]
    info_pages = [p for p in page_views if p.page_type == PageType.Informational]
    
    total_pages = len(page_views)
    bounce_rate = 0.0
    exit_rate = 0.0
    avg_page_value = 0.0
    
    if total_pages > 0:
        bounce_count = sum(1 for p in page_views if p.is_bounce)
        # Calculate exit count dynamically (last page is potential exit)
        exit_count = sum(1 for p in page_views if p.is_exit)
        bounce_rate = bounce_count / total_pages
        exit_rate = exit_count / total_pages
        avg_page_value = sum(p.page_value for p in page_views) / total_pages

    # 2. Build feature vector for Prediction API
    features = {
        "administrative": len(admin_pages),
        "administrative_duration": sum(p.duration_seconds for p in admin_pages),
        "informational": len(info_pages),
        "informational_duration": sum(p.duration_seconds for p in info_pages),
        "product_related": len(product_pages),
        "product_related_duration": sum(p.duration_seconds for p in product_pages),
        "bounce_rates": bounce_rate,
        "exit_rates": exit_rate,
        "page_values": avg_page_value,
        "special_day": session.special_day or 0.0,
        "month": session.month,
        "operating_systems": 2, 
        "browser": 2,           
        "region": session.region,
        "traffic_type": session.traffic_type,
        "visitor_type": session.visitor_type.name,
        "weekend": session.is_weekend
    }
    
    # 3. Call Prediction API
    try:
        response = requests.post(PREDICTION_API_URL, json=features, timeout=2.0)
        result = response.json()
        probability = result.get("probability", 0.0)
    except Exception:
        probability = 0.0 

    # 4. Decision Logic (Smart Intervention)
    # Only intervene if the AI has at least 12% confidence AND the user has seen at least 1 product.
    # This prevents popups for casual "landing-page-only" visitors.
    seen_product = len(product_pages) > 0
    should_intervene = (probability > 0.12) and seen_product
    
    return IntentCheckResponse(
        probability=probability,
        should_intervene=should_intervene
    )
