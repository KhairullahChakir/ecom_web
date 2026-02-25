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
    const API_URL = document.currentScript?.getAttribute('data-api') || 'http://localhost:8002';
    const SESSION_KEY = 'op_ecom_session_id';
    const AI_POLL_INTERVAL = 3000; // Check local AI every 3 seconds
    const AI_THRESHOLD = 0.60;
    const MODEL_URL = `${API_URL}/tracker/models/tcn_real_standalone.onnx`;

    // State
    let sessionId = null;
    let currentPageStart = Date.now();
    let currentPageUrl = window.location.href;
    let pageHistory = JSON.parse(localStorage.getItem('op_ecom_history') || '[]');
    let ortSession = null;
    let exitIntentChecked = false;

    // Page type to index mapping (must match backend)
    const PAGE_TYPE_TO_IDX = {
        'Home': 0, 'Product': 1, 'ProductDetail': 1, 'Cart': 2,
        'Checkout': 2, 'About': 3, 'Account': 3,
        'Administrative': 2, 'Informational': 3, 'ProductRelated': 1
    };

    // Load ONNX Runtime Web
    async function loadONNX() {
        if (window.ort) return true;
        return new Promise((resolve) => {
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js';
            script.onload = () => {
                console.log('[OP-ECOM Tracker] ONNX Runtime loaded');
                resolve(true);
            };
            document.head.appendChild(script);
        });
    }

    async function initModel() {
        try {
            await loadONNX();
            ortSession = await ort.InferenceSession.create(MODEL_URL);
            console.log('[OP-ECOM Tracker] Edge AI Engine Initialized (Local Inference)');
        } catch (e) {
            console.error('[OP-ECOM Tracker] Failed to load local AI model:', e);
        }
    }

    function updatePageHistory() {
        const duration = (Date.now() - currentPageStart) / 1000;
        const entry = {
            type: getPageType(),
            duration: Math.min(duration, 180)
        };
        pageHistory.push(entry);
        if (pageHistory.length > 20) pageHistory.shift();
        localStorage.setItem('op_ecom_history', JSON.stringify(pageHistory));
    }

    async function runLocalInference() {
        if (!ortSession || exitIntentChecked) return;

        // Skip high intent pages
        const path = window.location.pathname.toLowerCase();
        if (path.includes('checkout') || path.includes('cart') || path.includes('success')) return;

        // Ensure we have some history
        if (pageHistory.length === 0) return;

        try {
            // Prepare inputs
            const maxSeqLen = 20;
            const pageIdsSet = new BigInt64Array(maxSeqLen).fill(0n);
            const durationsSet = new Float32Array(maxSeqLen).fill(0.0);

            // Fill with history
            pageHistory.slice(-maxSeqLen).forEach((pv, i) => {
                pageIdsSet[i] = BigInt(PAGE_TYPE_TO_IDX[pv.type] || 1);
                durationsSet[i] = pv.duration / 180.0;
            });

            // Create tensors
            const pageIdsTensor = new ort.Tensor('int64', pageIdsSet, [1, maxSeqLen]);
            const durationsTensor = new ort.Tensor('float32', durationsSet, [1, maxSeqLen]);

            // Run session
            const results = await ortSession.run({
                'page_ids': pageIdsTensor,
                'durations': durationsTensor
            });

            // Sigmoid on output
            const logits = results[Object.keys(results)[0]].data[0];
            const prob = 1 / (1 + Math.exp(-logits));

            console.log(`[OP-ECOM Tracker] Local AI Prediction: ${(prob * 100).toFixed(1)}% abandonment risk`);

            if (prob > AI_THRESHOLD) {
                console.log('[OP-ECOM Tracker] LOCAL AI DETECTED HIGH RISK - Triggering intervention!');
                exitIntentChecked = true;
                showInterventionPopup(prob);
            }
        } catch (e) {
            console.warn('[OP-ECOM Tracker] Local inference failed:', e);
        }
    }

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
        if (path === '/' || path === '' || path.includes('index.html')) return 'Home';
        if (path.includes('account') || path.includes('cart') || path.includes('checkout') || path.includes('settings')) return 'Administrative';
        if (path.includes('about') || path.includes('contact') || path.includes('faq') || path.includes('help')) return 'Informational';
        return 'ProductRelated';
    }

    function getPageValue() { return 0; }

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
        const existingSession = localStorage.getItem(SESSION_KEY);
        if (existingSession) {
            sessionId = existingSession;
            console.log('[OP-ECOM Tracker] Resuming session:', sessionId);
            return;
        }

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
            localStorage.setItem(SESSION_KEY, sessionId);
            console.log('[OP-ECOM Tracker] Session started:', sessionId);
        }
    }

    async function endSession() {
        if (!sessionId) return;
        await trackPageView();
        await sendToAPI('/tracker/session/end', { session_id: sessionId });
        localStorage.removeItem(SESSION_KEY);
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
            is_bounce: duration < 1,
            is_exit: false,
            page_value: getPageValue(),
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
        await sendToAPI('/tracker/purchase', { session_id: sessionId, order_value: orderValue || 0 });
    }

    // Page Navigation Handling
    function handlePageChange() {
        updatePageHistory();
        trackPageView();
        currentPageStart = Date.now();
        currentPageUrl = window.location.href;
    }

    async function checkExitIntent() {
        // Manual check still uses local inference
        await runLocalInference();
    }

    function showInterventionPopup(probability) {
        trackEvent('exit_intent_shown', 'intervention', 'popup_displayed', Math.round(probability * 100));
        const cart = JSON.parse(localStorage.getItem('shopDemo_cart') || '[]');
        const cartValue = cart.reduce((sum, item) => sum + ((item.price || 0) * (item.qty || 1)), 0);

        const overlay = document.createElement('div');
        overlay.id = 'op-ecom-overlay';
        Object.assign(overlay.style, {
            position: 'fixed', top: '0', left: '0', width: '100%', height: '100%',
            backgroundColor: 'rgba(0,0,0,0.6)', zIndex: '99999', display: 'flex',
            justifyContent: 'center', alignItems: 'center', backdropFilter: 'blur(8px)',
            animation: 'opFadeIn 0.3s ease'
        });

        if (!document.getElementById('op-ecom-animations')) {
            const style = document.createElement('style');
            style.id = 'op-ecom-animations';
            style.textContent = `
                @keyframes opFadeIn { from { opacity: 0; } to { opacity: 1; } }
                @keyframes opSlideUp { from { opacity: 0; transform: translateY(30px) scale(0.95); } to { opacity: 1; transform: translateY(0) scale(1); } }
                @keyframes opPulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.05); } }
            `;
            document.head.appendChild(style);
        }

        const popup = document.createElement('div');
        Object.assign(popup.style, {
            background: 'linear-gradient(135deg, #F97316 0%, #EA580C 100%)',
            padding: '2.5rem', borderRadius: '20px', maxWidth: '420px', width: '90%',
            textAlign: 'center', boxShadow: '0 25px 60px rgba(0,0,0,0.3)',
            fontFamily: "'Inter', 'Segoe UI', sans-serif", color: 'white',
            animation: 'opSlideUp 0.4s cubic-bezier(0.34,1.56,0.64,1)'
        });

        popup.innerHTML = `
            <div style="width:60px;height:60px;background:rgba(255,255,255,0.2);border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.8rem;margin:0 auto 1.2rem;">🎁</div>
            <h2 style="color:#fff;margin-bottom:0.5rem;font-size:1.6rem;font-weight:800;">Wait! Edge AI Offer</h2>
            <p style="color:rgba(255,255,255,0.85);margin-bottom:1.5rem;font-size:0.95rem;">
                Our local AI detected you might be leaving. Take 20% OFF!
            </p>
            <input type="email" id="op-ecom-email" placeholder="your@email.com" style="
                width:100%;padding:14px 16px;font-size:1rem;border:2px solid rgba(255,255,255,0.3);
                border-radius:12px;margin-bottom:1rem;box-sizing:border-box;background:rgba(255,255,255,0.15);
                color:#fff;outline:none;font-family:inherit;
            " />
            <button id="op-ecom-submit" style="
                background:#fff;color:#EA580C;border:none;padding:14px 24px;
                font-size:1.05rem;border-radius:12px;cursor:pointer;width:100%;
                font-weight:700;font-family:inherit;box-shadow:0 4px 15px rgba(0,0,0,0.1);"
            >Get My Discount</button>
            <button id="op-ecom-close" style="
                background:transparent;border:none;color:rgba(255,255,255,0.6);margin-top:1rem;
                cursor:pointer;font-size:0.85rem;font-family:inherit;"
            >No thanks, I'll pay full price</button>
        `;

        overlay.appendChild(popup);
        document.body.appendChild(overlay);
        setTimeout(() => document.getElementById('op-ecom-email')?.focus(), 400);

        document.getElementById('op-ecom-submit').onclick = async () => {
            const emailInput = document.getElementById('op-ecom-email');
            const email = emailInput.value.trim();
            if (!email || !email.includes('@')) return;

            const result = await sendToAPI('/tracker/email-capture', {
                session_id: sessionId, email: email, cart_value: cartValue
            });

            if (result && result.success) {
                popup.style.background = 'linear-gradient(135deg, #059669 0%, #10b981 100%)';
                popup.innerHTML = `
                    <div style="width:60px;height:60px;background:rgba(255,255,255,0.2);border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.8rem;margin:0 auto 1.2rem;">🎉</div>
                    <h2 style="color:#fff;margin-bottom:0.5rem;font-size:1.6rem;font-weight:800;">Success!</h2>
                    <p style="color:rgba(255,255,255,0.85);margin-bottom:1.5rem;">Use code: <strong>${result.discount_code}</strong></p>
                    <button id="op-ecom-done" style="background:#fff;color:#059669;border:none;padding:14px 24px;border-radius:12px;cursor:pointer;width:100%;font-weight:700;">Continue</button>
                `;
                document.getElementById('op-ecom-done').onclick = () => {
                    trackEvent('discount_claimed', 'intervention', 'claim_discount', result.discount_percent);
                    overlay.remove();
                };
            }
        };

        document.getElementById('op-ecom-close').onclick = () => overlay.remove();
    }

    async function init() {
        await startSession();
        await initModel();

        // Local AI Inference Loop
        setInterval(async () => {
            updatePageHistory();
            await runLocalInference();
        }, AI_POLL_INTERVAL);

        window.addEventListener('beforeunload', () => {
            updatePageHistory();
            trackPageView();
        });

        window.addEventListener('popstate', handlePageChange);
        const originalPushState = history.pushState;
        history.pushState = function () {
            updatePageHistory();
            trackPageView();
            originalPushState.apply(history, arguments);
            currentPageStart = Date.now();
            currentPageUrl = window.location.href;
        };
        window.addEventListener('pagehide', endSession);
        console.log('[OP-ECOM Tracker] Initialized with EDGE AI (Local Inference).');
    }

    window.opEcomTracker = {
        trackEvent, trackPurchase, getSessionId: () => sessionId,
        checkExitIntent: async () => { exitIntentChecked = false; await runLocalInference(); }
    };

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
