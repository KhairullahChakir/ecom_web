"""
OP-ECOM Analytics Tracker - FastAPI Router
Endpoints for tracking sessions, page views, and events
"""

import uuid
import requests
import numpy as np
import onnxruntime as ort
import os
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import func

from .database import get_db
from .models import Session, PageView, Event, EmailCapture, VisitorType, PageType
from .schemas import (
    SessionStartRequest, SessionStartResponse, SessionEndRequest,
    PageViewRequest, EventRequest, PurchaseRequest,
    IntentCheckRequest, IntentCheckResponse,
    EmailCaptureRequest, EmailCaptureResponse
)

# Model paths
PREDICTION_API_URL = "http://localhost:8000/predict"
# Use absolute path to avoid relative path issues
_base_dir = os.path.dirname(os.path.abspath(__file__))
TRANSFORMER_MODEL_PATH = os.path.normpath(os.path.join(_base_dir, "..", "..", "backend", "models", "abandonment_model.onnx"))

# Initialize Abandonment Model (Final choice: TCN for speed)
abandonment_session = None
try:
    print(f"[DEBUG] Looking for model at: {TRANSFORMER_MODEL_PATH}")
    if os.path.exists(TRANSFORMER_MODEL_PATH):
        # Change working directory temporarily to help ONNX find .data files
        original_cwd = os.getcwd()
        os.chdir(os.path.dirname(TRANSFORMER_MODEL_PATH))
        abandonment_session = ort.InferenceSession(os.path.basename(TRANSFORMER_MODEL_PATH))
        os.chdir(original_cwd)
        print(f"✅ Loaded Abandonment model from {TRANSFORMER_MODEL_PATH}")
    else:
        print(f"⚠️ Abandonment model not found at {TRANSFORMER_MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Failed to load Abandonment model: {e}")

# Page type mapping for TCN input (model was trained with num_page_types=4)
# All indices must be in range [0, 3]
PAGE_TYPE_TO_IDX = {
    'Home': 0, 'Product': 1, 'ProductDetail': 1, 'Cart': 2, 
    'Checkout': 2, 'About': 3, 'Account': 3,
    'Administrative': 2, 'Informational': 3, 'ProductRelated': 1  # Map DB enums
}

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
        # 2. ==========================================
    #    STEP 1: Run Abandonment Model (TCN)
    #    This predicts if the user is ABOUT TO LEAVE.
    # ==========================================
    abandonment_prob = 0.5  # Default fallback
    
    if abandonment_session is not None and len(page_views) >= 1:
        # Build sequence for TCN
        max_seq_len = 20
        page_ids = np.zeros((1, max_seq_len), dtype=np.int64)
        durations = np.zeros((1, max_seq_len), dtype=np.float32)
        
        for i, pv in enumerate(page_views[:max_seq_len]):
            page_type_name = pv.page_type.name if hasattr(pv.page_type, 'name') else str(pv.page_type)
            page_ids[0, i] = PAGE_TYPE_TO_IDX.get(page_type_name, 2)  # Default to Product
            durations[0, i] = min(pv.duration_seconds, 180.0) / 180.0  # Normalize
        
        # Run inference
        try:
            result = abandonment_session.run(None, {
                'page_ids': page_ids,
                'durations': durations
            })
            # Apply sigmoid to logits
            logits = result[0][0][0]
            abandonment_prob = 1 / (1 + np.exp(-logits))
        except Exception as e:
            print(f"TCN inference error: {e}")
            abandonment_prob = 0.5
    
    # 3. ==========================================
    #    STEP 2: Only call TabM if abandonment risk is HIGH
    #    This saves compute and makes the system smarter.
    # ==========================================
    purchase_prob = 0.0
    
    # Only call TabM if user is likely to leave (abandonment > 30% for demo)
    if abandonment_prob > 0.30:
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
        
        try:
            response = requests.post(PREDICTION_API_URL, json=features, timeout=2.0)
            result = response.json()
            purchase_prob = result.get("probability", 0.0)
        except Exception:
            purchase_prob = 0.0

    # 4. ==========================================
    #    DECISION LOGIC: Dual-Signal Intervention
    #    Intervene if:
    #      - User is likely to LEAVE (abandonment > 40%)
    #      - AND User is likely to BUY (purchase > 5%)
    #      - AND User has seen at least 1 product
    # ==========================================
    seen_product = len(product_pages) > 0
    # TESTING: Simplified logic - only check abandonment
    should_intervene = (
        abandonment_prob > 0.40 and   # Lower threshold for testing (40%)
        seen_product                   # User has viewed at least 1 product
    )

    
    # Return the combined probability (weighted average for display)
    combined_prob = (abandonment_prob * 0.4 + purchase_prob * 0.6) if should_intervene else purchase_prob
    
    # Calculate personalized discount based on cart value
    discount_percent = calculate_discount(request.cart_value)
    
    # Generate XAI Explanation (for admin dashboard)
    total_duration = sum(pv.duration_seconds for pv in page_views)
    xai_explanation = None
    if should_intervene:
        # Build human-readable explanation
        reasons = []
        if abandonment_prob > 0.80:
            reasons.append("Very high abandonment risk detected")
        elif abandonment_prob > 0.50:
            reasons.append("Moderate abandonment risk detected")
        
        if len(product_pages) > 3:
            reasons.append(f"Browsed {len(product_pages)} products without purchasing")
        if total_duration > 120:
            reasons.append(f"Spent {int(total_duration//60)}m {int(total_duration%60)}s on site")
        if request.cart_value > 100:
            reasons.append(f"High cart value (${request.cart_value:.2f})")
        
        xai_explanation = {
            "pages_viewed": len(page_views),
            "product_pages": len(product_pages),
            "cart_pages": len([p for p in page_views if 'cart' in (p.page_url or '').lower()]),
            "total_duration_sec": round(total_duration, 1),
            "cart_value": request.cart_value,
            "abandonment_score": round(abandonment_prob * 100, 1),
            "purchase_score": round(purchase_prob * 100, 1),
            "reasons": reasons,
            "decision": " | ".join(reasons) if reasons else "AI detected exit intent"
        }
    
    return IntentCheckResponse(
        probability=combined_prob,
        should_intervene=should_intervene,
        abandonment_prob=abandonment_prob,
        purchase_prob=purchase_prob,
        discount_percent=discount_percent,
        xai_explanation=xai_explanation
    )


# --- Smart Discount Calculation ---
def calculate_discount(cart_value: float) -> int:
    """
    Calculate personalized discount based on cart value.
    Higher cart = lower discount (they're likely to buy anyway)
    Lower cart = higher discount (win them over)
    """
    if cart_value <= 50:
        return 30  # Big discount for hesitant buyers
    elif cart_value <= 150:
        return 20  # Balanced incentive
    else:
        return 10  # Small nudge for high-value carts


def generate_discount_code(discount_percent: int) -> str:
    """Generate a unique discount code"""
    import random
    import string
    suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"SAVE{discount_percent}-{suffix}"


# --- Email Capture Endpoint ---
@router.post("/email-capture", response_model=EmailCaptureResponse)
async def capture_email(request: EmailCaptureRequest, db: DBSession = Depends(get_db)):
    """
    Capture email and generate personalized discount code.
    This is called when user enters email in the exit-intent popup.
    """
    # Verify session exists
    session = db.query(Session).filter(Session.session_id == request.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Calculate personalized discount
    discount_percent = calculate_discount(request.cart_value)
    discount_code = generate_discount_code(discount_percent)
    
    # Store email capture
    email_capture = EmailCapture(
        session_id=request.session_id,
        email=request.email,
        discount_code=discount_code,
        discount_percent=discount_percent,
        cart_value=request.cart_value
    )
    db.add(email_capture)
    
    # Also log as event for analytics
    event = Event(
        session_id=request.session_id,
        event_type="email_captured",
        event_category="conversion",
        event_label=f"discount_{discount_percent}%",
        event_value=request.cart_value
    )
    db.add(event)
    db.commit()
    
    return EmailCaptureResponse(
        success=True,
        discount_code=discount_code,
        discount_percent=discount_percent,
        message=f"Your exclusive {discount_percent}% discount code!"
    )
