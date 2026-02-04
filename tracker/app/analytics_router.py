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
