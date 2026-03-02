"""
OP-ECOM Analytics Tracker - Admin Router
Endpoints for viewing sessions and exporting data
"""

import csv
import io
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import func

from .database import get_db
from .models import Session, PageView

router = APIRouter(prefix="/admin", tags=["Admin"])

@router.get("/sessions")
async def list_sessions(limit: int = 100, db: DBSession = Depends(get_db)):
    """List all tracked sessions"""
    sessions = db.query(Session).order_by(Session.started_at.desc()).limit(limit).all()
    
    result = []
    for s in sessions:
        page_count = db.query(func.count(PageView.id)).filter(PageView.session_id == s.session_id).scalar()
        result.append({
            "session_id": s.session_id,
            "visitor_type": s.visitor_type.value if s.visitor_type else "Unknown",
            "started_at": s.started_at.isoformat() if s.started_at else None,
            "ended_at": s.ended_at.isoformat() if s.ended_at else None,
            "revenue": s.revenue,
            "page_count": page_count,
            "browser": s.browser,
            "operating_system": s.operating_system
        })
    
    return {"total": len(result), "sessions": result}

@router.get("/sessions/{session_id}")
async def get_session_detail(session_id: str, db: DBSession = Depends(get_db)):
    """Get detailed info for a specific session"""
    session = db.query(Session).filter(Session.session_id == session_id).first()
    if not session:
        return {"error": "Session not found"}
    
    page_views = db.query(PageView).filter(PageView.session_id == session_id).order_by(PageView.viewed_at).all()
    
    return {
        "session": {
            "session_id": session.session_id,
            "visitor_type": session.visitor_type.value if session.visitor_type else "Unknown",
            "browser": session.browser,
            "operating_system": session.operating_system,
            "region": session.region,
            "traffic_type": session.traffic_type,
            "is_weekend": session.is_weekend,
            "month": session.month,
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "ended_at": session.ended_at.isoformat() if session.ended_at else None,
            "revenue": session.revenue,
            "metrics": {
                "administrative_count": session.administrative_count,
                "administrative_duration": session.administrative_duration,
                "informational_count": session.informational_count,
                "informational_duration": session.informational_duration,
                "product_related_count": session.product_related_count,
                "product_related_duration": session.product_related_duration,
                "bounce_rates": session.bounce_rates,
                "exit_rates": session.exit_rates,
                "page_values": session.page_values
            }
        },
        "page_views": [
            {
                "page_type": p.page_type.value if p.page_type else "Unknown",
                "page_url": p.page_url,
                "page_title": p.page_title,
                "duration_seconds": p.duration_seconds,
                "page_value": p.page_value,
                "viewed_at": p.viewed_at.isoformat() if p.viewed_at else None
            }
            for p in page_views
        ]
    }

@router.get("/export")
async def export_csv(db: DBSession = Depends(get_db)):
    """Export all completed sessions as CSV (UCI dataset format)"""
    sessions = db.query(Session).filter(Session.ended_at.isnot(None)).all()
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header matching UCI dataset
    header = [
        "Administrative", "Administrative_Duration",
        "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues",
        "SpecialDay", "Month", "OperatingSystems", "Browser",
        "Region", "TrafficType", "VisitorType", "Weekend", "Revenue"
    ]
    writer.writerow(header)
    
    # Data rows
    for s in sessions:
        row = [
            s.administrative_count,
            s.administrative_duration,
            s.informational_count,
            s.informational_duration,
            s.product_related_count,
            s.product_related_duration,
            s.bounce_rates,
            s.exit_rates,
            s.page_values,
            s.special_day,
            s.month,
            s.operating_system or "Unknown",
            s.browser or "Unknown",
            s.region,
            s.traffic_type,
            s.visitor_type.value if s.visitor_type else "Other",
            s.is_weekend,
            s.revenue
        ]
        writer.writerow(row)
    
    output.seek(0)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sessions_export.csv"}
    )

@router.get("/stats")
async def get_stats(db: DBSession = Depends(get_db)):
    """Get overall tracker statistics"""
    total_sessions = db.query(func.count(Session.id)).scalar()
    completed_sessions = db.query(func.count(Session.id)).filter(Session.ended_at.isnot(None)).scalar()
    converted_sessions = db.query(func.count(Session.id)).filter(Session.revenue == True).scalar()
    total_pageviews = db.query(func.count(PageView.id)).scalar()
    
    conversion_rate = (converted_sessions / completed_sessions * 100) if completed_sessions > 0 else 0
    
    return {
        "total_sessions": total_sessions,
        "completed_sessions": completed_sessions,
        "converted_sessions": converted_sessions,
        "conversion_rate": round(conversion_rate, 2),
        "total_pageviews": total_pageviews
    }
