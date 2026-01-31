/**
 * OP-ECOM Analytics Tracker - JavaScript Client
 * Lightweight script to track user behavior on any website
 * 
 * Usage:
 * <script src="tracker.js" data-api="http://localhost:8001"></script>
 */

(function () {
    'use strict';

    // Configuration
    const API_URL = document.currentScript?.getAttribute('data-api') || 'http://localhost:8001';
    const SESSION_KEY = 'op_ecom_session_id';

    // State
    let sessionId = null;
    let currentPageStart = Date.now();
    let currentPageUrl = window.location.href;

    // Utility functions
    function generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    function getBrowserInfo() {
        const ua = navigator.userAgent;
        let browser = 'Unknown';
        if (ua.includes('Chrome')) browser = 'Chrome';
        else if (ua.includes('Firefox')) browser = 'Firefox';
        else if (ua.includes('Safari')) browser = 'Safari';
        else if (ua.includes('Edge')) browser = 'Edge';
        else if (ua.includes('Opera')) browser = 'Opera';
        return browser;
    }

    function getOS() {
        const ua = navigator.userAgent;
        if (ua.includes('Windows')) return 'Windows';
        if (ua.includes('Mac')) return 'MacOS';
        if (ua.includes('Linux')) return 'Linux';
        if (ua.includes('Android')) return 'Android';
        if (ua.includes('iOS') || ua.includes('iPhone')) return 'iOS';
        return 'Unknown';
    }

    function getPageType() {
        const path = window.location.pathname.toLowerCase();
        if (path.includes('account') || path.includes('cart') || path.includes('checkout') || path.includes('settings')) {
            return 'Administrative';
        }
        if (path.includes('about') || path.includes('contact') || path.includes('faq') || path.includes('help')) {
            return 'Informational';
        }
        return 'ProductRelated';
    }

    async function sendToAPI(endpoint, data) {
        try {
            const response = await fetch(`${API_URL}${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            return await response.json();
        } catch (error) {
            console.warn('[OP-ECOM Tracker] API Error:', error.message);
            return null;
        }
    }

    // Session Management
    async function startSession() {
        // Check for existing session
        const existingSession = sessionStorage.getItem(SESSION_KEY);
        if (existingSession) {
            sessionId = existingSession;
            console.log('[OP-ECOM Tracker] Resuming session:', sessionId);
            return;
        }

        // Start new session
        const visitorType = localStorage.getItem('op_ecom_returning') ? 'Returning_Visitor' : 'New_Visitor';
        localStorage.setItem('op_ecom_returning', 'true');

        const result = await sendToAPI('/tracker/session/start', {
            visitor_type: visitorType,
            browser: getBrowserInfo(),
            operating_system: getOS(),
            region: 1,
            traffic_type: document.referrer ? 2 : 1
        });

        if (result?.session_id) {
            sessionId = result.session_id;
            sessionStorage.setItem(SESSION_KEY, sessionId);
            console.log('[OP-ECOM Tracker] Session started:', sessionId);
        }
    }

    async function endSession() {
        if (!sessionId) return;

        // Track final page duration
        await trackPageView();

        await sendToAPI('/tracker/session/end', {
            session_id: sessionId
        });

        sessionStorage.removeItem(SESSION_KEY);
        console.log('[OP-ECOM Tracker] Session ended:', sessionId);
    }

    // Page View Tracking
    async function trackPageView() {
        if (!sessionId) return;

        const duration = (Date.now() - currentPageStart) / 1000;

        await sendToAPI('/tracker/pageview', {
            session_id: sessionId,
            page_type: getPageType(),
            page_url: currentPageUrl,
            page_title: document.title,
            duration_seconds: duration,
            is_bounce: duration < 5,
            is_exit: false,
            page_value: 0,
            scroll_depth: Math.round((window.scrollY / (document.body.scrollHeight - window.innerHeight)) * 100) || 0
        });
    }

    // Event Tracking
    async function trackEvent(eventType, eventCategory, eventLabel, eventValue, eventData) {
        if (!sessionId) return;

        await sendToAPI('/tracker/event', {
            session_id: sessionId,
            event_type: eventType,
            event_category: eventCategory || null,
            event_label: eventLabel || null,
            event_value: eventValue || 0,
            event_data: eventData || null
        });
    }

    // Purchase Tracking
    async function trackPurchase(orderValue) {
        if (!sessionId) return;

        await sendToAPI('/tracker/purchase', {
            session_id: sessionId,
            order_value: orderValue || 0
        });

        console.log('[OP-ECOM Tracker] Purchase tracked!');
    }

    // Page Navigation Handling
    function handlePageChange() {
        trackPageView();
        currentPageStart = Date.now();
        currentPageUrl = window.location.href;
    }

    // Exit Intent & Intervention
    let exitIntentChecked = false;

    async function checkExitIntent() {
        if (!sessionId || exitIntentChecked) return;
        exitIntentChecked = true; // Check only once per session/page load to avoid spam

        console.log('[OP-ECOM Tracker] Checking exit intent...');
        const result = await sendToAPI('/tracker/check-intent', {
            session_id: sessionId
        });

        if (result && result.should_intervene) {
            showInterventionPopup(result.probability);
        }
    }

    function showInterventionPopup(probability) {
        // Create popup elements
        const overlay = document.createElement('div');
        overlay.id = 'op-ecom-overlay';
        Object.assign(overlay.style, {
            position: 'fixed', top: '0', left: '0', width: '100%', height: '100%',
            backgroundColor: 'rgba(0,0,0,0.7)', zIndex: '9999', display: 'flex',
            justifyContent: 'center', alignItems: 'center', backdropFilter: 'blur(5px)'
        });

        const popup = document.createElement('div');
        Object.assign(popup.style, {
            backgroundColor: 'white', padding: '2rem', borderRadius: '15px',
            maxWidth: '400px', textAlign: 'center', boxShadow: '0 10px 25px rgba(0,0,0,0.2)',
            fontFamily: "'Segoe UI', sans-serif"
        });

        const percent = Math.round(probability * 100);
        popup.innerHTML = `
            <div style="font-size: 3rem; margin-bottom: 1rem;">🎁</div>
            <h2 style="color: #1E4FA8; margin-bottom: 0.5rem;">Wait! Don't Go!</h2>
            <p style="color: #666; margin-bottom: 1.5rem;">
                We see you're interested! Here is a special discount just for you.
            </p>
            <div style="background: #eef2ff; padding: 10px; border-radius: 8px; margin-bottom: 1.5rem;">
                <code style="font-size: 1.5rem; color: #1E4FA8; font-weight: bold;">OP-ECOM-${percent}</code>
            </div>
            <button id="op-ecom-claim" style="
                background: #1E4FA8; color: white; border: none; padding: 12px 24px;
                font-size: 1.1rem; border-radius: 8px; cursor: pointer; width: 100%;
                transition: transform 0.2s;"
            >Claim Discount</button>
            <button id="op-ecom-close" style="
                background: transparent; border: none; color: #999; margin-top: 1rem;
                cursor: pointer; text-decoration: underline;"
            >No thanks, I'll pay full price</button>
            <div style="margin-top: 1rem; font-size: 0.8rem; color: #ccc;">
                AI Confidence: ${percent}%
            </div>
        `;

        overlay.appendChild(popup);
        document.body.appendChild(overlay);

        // Handlers
        document.getElementById('op-ecom-claim').onclick = () => {
            trackEvent('click', 'intervention', 'claim_discount', percent);
            overlay.remove();
        };
        document.getElementById('op-ecom-close').onclick = () => {
            trackEvent('click', 'intervention', 'dismiss', 0);
            overlay.remove();
        };
    }

    function setupExitIntent() {
        document.addEventListener('mouseleave', (e) => {
            if (e.clientY < 50) { // Top of screen
                checkExitIntent();
            }
        });
    }

    // Initialize
    async function init() {
        await startSession();
        setupExitIntent();

        // Track page views on navigation
        window.addEventListener('beforeunload', () => {
            trackPageView();
        });

        // Handle SPA navigation
        window.addEventListener('popstate', handlePageChange);

        // Intercept history pushState for SPA
        const originalPushState = history.pushState;
        history.pushState = function () {
            trackPageView();
            originalPushState.apply(history, arguments);
            currentPageStart = Date.now();
            currentPageUrl = window.location.href;
        };

        // End session on page close
        window.addEventListener('unload', endSession);

        console.log('[OP-ECOM Tracker] Initialized');
    }

    // Expose global API
    window.opEcomTracker = {
        trackEvent,
        trackPurchase,
        getSessionId: () => sessionId
    };

    // Start tracking
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
