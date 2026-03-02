"""
OP-ECOM Analytics API Router
Provides endpoints for the analytics dashboard
"""

from fastapi import APIRouter, Depends
from sqlalchemy import func, desc, Integer, cast
from sqlalchemy.orm import Session as DBSession
from datetime import datetime, timedelta
from typing import List, Optional
from pydantic import BaseModel

from .database import get_db
from .models import Session, PageView, Event

router = APIRouter(prefix="/analytics", tags=["Analytics"])


# =====================
# Response Models
# =====================

class SummaryStats(BaseModel):
    total_sessions: int
    total_interventions: int
    total_claims: int
    total_conversions: int
    conversion_rate: float
    claim_rate: float


class TimeSeriesPoint(BaseModel):
    timestamp: str
    interventions: int
    claims: int


class PageStats(BaseModel):
    page_url: str
    page_title: str
    views: int
    abandonment_rate: float


# =====================
# API Endpoints
# =====================

@router.get("/summary", response_model=SummaryStats)
async def get_summary(db: DBSession = Depends(get_db)):
    """Get overall analytics summary"""
    
    # Total sessions
    total_sessions = db.query(func.count(Session.id)).scalar() or 0
    
    # Total interventions (exit_intent_shown events)
    total_interventions = db.query(func.count(Event.id)).filter(
        Event.event_type == "exit_intent_shown"
    ).scalar() or 0
    
    # Total claims (discount_claimed events)
    total_claims = db.query(func.count(Event.id)).filter(
        Event.event_type == "discount_claimed"
    ).scalar() or 0
    
    # Total conversions (sessions with revenue=True)
    total_conversions = db.query(func.count(Session.id)).filter(
        Session.revenue == True
    ).scalar() or 0
    
    # Calculate rates
    conversion_rate = (total_conversions / total_sessions * 100) if total_sessions > 0 else 0
    claim_rate = (total_claims / total_interventions * 100) if total_interventions > 0 else 0
    
    return SummaryStats(
        total_sessions=total_sessions,
        total_interventions=total_interventions,
        total_claims=total_claims,
        total_conversions=total_conversions,
        conversion_rate=round(conversion_rate, 1),
        claim_rate=round(claim_rate, 1)
    )


@router.get("/interventions", response_model=List[TimeSeriesPoint])
async def get_interventions_timeseries(
    hours: int = 24,
    db: DBSession = Depends(get_db)
):
    """Get intervention time series for the last N hours"""
    
    cutoff = datetime.now() - timedelta(hours=hours)
    
    # Query interventions grouped by hour
    interventions = db.query(
        func.date_format(Event.created_at, '%Y-%m-%d %H:00').label('hour'),
        func.count(Event.id).label('count')
    ).filter(
        Event.event_type == "exit_intent_shown",
        Event.created_at >= cutoff
    ).group_by('hour').order_by('hour').all()
    
    # Query claims grouped by hour
    claims = db.query(
        func.date_format(Event.created_at, '%Y-%m-%d %H:00').label('hour'),
        func.count(Event.id).label('count')
    ).filter(
        Event.event_type == "discount_claimed",
        Event.created_at >= cutoff
    ).group_by('hour').order_by('hour').all()
    
    # Merge into time series
    claims_dict = {c.hour: c.count for c in claims}
    
    result = []
    for i in interventions:
        result.append(TimeSeriesPoint(
            timestamp=i.hour,
            interventions=i.count,
            claims=claims_dict.get(i.hour, 0)
        ))
    
    return result


@router.get("/pages", response_model=List[PageStats])
async def get_page_stats(
    limit: int = 10,
    db: DBSession = Depends(get_db)
):
    """Get page-level statistics for abandonment heatmap"""
    
    # Get page views with exit rates
    pages = db.query(
        PageView.page_url,
        PageView.page_title,
        func.count(PageView.id).label('views'),
        func.sum(PageView.is_exit.cast(Integer)).label('exits')
    ).group_by(
        PageView.page_url, PageView.page_title
    ).order_by(
        desc('views')
    ).limit(limit).all()
    
    result = []
    for p in pages:
        abandonment_rate = (p.exits / p.views * 100) if p.views > 0 else 0
        result.append(PageStats(
            page_url=p.page_url or "Unknown",
            page_title=p.page_title or "Unknown",
            views=p.views,
            abandonment_rate=round(abandonment_rate, 1)
        ))
    
    return result


# =====================
# XAI Intervention Details
# =====================

class InterventionDetail(BaseModel):
    id: int
    session_id: str
    timestamp: str
    event_value: float
    xai_data: Optional[dict] = None


@router.get("/interventions/details", response_model=List[InterventionDetail])
async def get_intervention_details(
    limit: int = 20,
    db: DBSession = Depends(get_db)
):
    """Get detailed intervention list with XAI explanations"""
    
    # Get recent exit_intent_shown events with their data
    interventions = db.query(Event).filter(
        Event.event_type == "exit_intent_shown"
    ).order_by(desc(Event.created_at)).limit(limit).all()
    
    result = []
    for event in interventions:
        # Get session page view stats for XAI
        page_views = db.query(PageView).filter(
            PageView.session_id == event.session_id
        ).all()
        
        total_duration = sum(pv.duration_seconds for pv in page_views)
        product_pages = [p for p in page_views if p.page_type.name == 'ProductRelated']
        cart_pages = [p for p in page_views if 'cart' in (p.page_url or '').lower()]
        
        # Check for add_to_cart events
        cart_events = db.query(Event).filter(
            Event.session_id == event.session_id,
            Event.event_type == "add_to_cart"
        ).all()
        
        # Build XAI explanation
        xai_data = {
            "pages_viewed": len(page_views),
            "product_pages": len(product_pages),
            "cart_pages": len(cart_pages),
            "cart_items": len(cart_events),
            "total_duration": f"{int(total_duration//60)}m {int(total_duration%60)}s",
            "abandonment_score": f"{event.event_value}%",
            "reasons": []
        }
        
        # Generate reasons
        if event.event_value >= 80:
            xai_data["reasons"].append("Very high abandonment risk")
        elif event.event_value >= 50:
            xai_data["reasons"].append("Moderate abandonment risk")
        
        if len(cart_events) > 0:
            xai_data["reasons"].append(f"Added {len(cart_events)} items to cart")
        if len(product_pages) > 3:
            xai_data["reasons"].append(f"Browsed {len(product_pages)} products")
        if total_duration > 120:
            xai_data["reasons"].append(f"Extended session duration")
        if len(cart_pages) > 0:
            xai_data["reasons"].append("Visited cart page")
        
        if not xai_data["reasons"]:
            xai_data["reasons"].append("AI detected exit intent")
        
        result.append(InterventionDetail(
            id=event.id,
            session_id=event.session_id[:8] + "...",  # Truncate for display
            timestamp=event.created_at.strftime("%H:%M:%S"),
            event_value=event.event_value,
            xai_data=xai_data
        ))
    
    return result
