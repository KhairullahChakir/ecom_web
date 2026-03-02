/**
 * OP-ECOM Analytics Tracker - JavaScript Client
 * Edge AI (Local Inference) + Hybrid Golden Logic
 */

(function () {
    'use strict';

    // Configuration
    const API_URL = 'http://localhost:8002'; // Local Development
    const SESSION_KEY = 'op_ecom_session_id';
    const AI_POLL_INTERVAL = 3000;
    const AI_THRESHOLD = 0.40;
    const MODEL_URL = `${API_URL}/tracker/models/tcn_real_standalone.onnx`;

    // State
    let sessionId = null;
    let currentPageStart = Date.now();
    let currentPageUrl = window.location.href;
    let pageHistory = JSON.parse(localStorage.getItem('op_ecom_history') || '[]');
    let ortSession = null;
    let exitIntentChecked = false;

    // Page type to index mapping (Must align with Rigorous Training)
    // 0: Padding, 1: Browsing (Views), 2: Intent (Add to Cart), 3: Success (Transaction)
    const PAGE_TYPE_TO_IDX = {
        'Home': 1, 'Product': 1, 'ProductDetail': 1, 'Cart': 1,
        'Checkout': 1, 'About': 1, 'Account': 1,
        'Administrative': 1, 'Informational': 1, 'ProductRelated': 1,
        'AddToCartEvent': 2
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
        const type = getPageType();

        // Prevent Flooding: If same page type and very short duration, don't add
        if (pageHistory.length > 0) {
            const last = pageHistory[pageHistory.length - 1];
            if (last.type === type && duration < 0.5) return;
        }

        pageHistory.push({ type, duration: Math.min(duration, 600) });
        if (pageHistory.length > 20) pageHistory.shift();
        localStorage.setItem('op_ecom_history', JSON.stringify(pageHistory));
    }

    async function runLocalInference() {
        if (!ortSession) return;

        // Skip high intent pages
        const path = window.location.pathname.toLowerCase();
        if (path.includes('checkout') || path.includes('cart') || path.includes('success')) return;

        try {
            // DYNAMIC SEQUENCE: Past history + current dwell time
            const currentType = getPageType();
            const currentDwell = (Date.now() - currentPageStart) / 1000;

            // Build the raw sequence: past events + current page
            const rawSeq = [...pageHistory, { type: currentType, duration: currentDwell }];

            // Pattern Threshold: We need at least 10 events to establish a reliable behavior pattern.
            const MIN_EVENTS = 10;
            if (rawSeq.length < MIN_EVENTS) {
                console.log(`[OP-ECOM Tracker] AI Engine: Collecting data... (${rawSeq.length}/${MIN_EVENTS} events)`);
                return;
            }

            // Verbose logging for Transparency
            const eventsText = rawSeq.slice(-10).map(e => e.type).join(' -> ');
            console.log(`[OP-ECOM Tracker] AI analyzing last 10 events: ${eventsText}`);

            // SEQUENCE FILLING: The model expects 20 events. With fewer events,
            // zeros cause 100% risk (model learned: zeros = no activity = abandoned).
            // Solution: Fill the full 20-slot array by repeating the user's ACTUAL
            // behavior pattern, so the model sees a "typical session" from this user.
            const maxSeqLen = 20;
            const pageIdsSet = new BigInt64Array(maxSeqLen);
            const durationsSet = new Float32Array(maxSeqLen);

            for (let i = 0; i < maxSeqLen; i++) {
                const srcIdx = i < rawSeq.length ? i : (i % rawSeq.length);
                const pv = rawSeq[srcIdx];

                pageIdsSet[i] = BigInt(PAGE_TYPE_TO_IDX[pv.type] || 1);

                // Last slot always gets the "ongoing" marker (0.05)
                if (i === maxSeqLen - 1) {
                    durationsSet[i] = 0.05;
                } else {
                    durationsSet[i] = Math.min(pv.duration / 600.0, 1.0);
                }
            }

            const results = await ortSession.run({
                'page_ids': new ort.Tensor('int64', pageIdsSet, [1, maxSeqLen]),
                'durations': new ort.Tensor('float32', durationsSet, [1, maxSeqLen])
            });

            const logits = results[Object.keys(results)[0]].data[0];
            const prob = 1 / (1 + Math.exp(-logits));

            // Log risk score (Model 2: TCN)
            console.log(`[OP-ECOM Tracker] Model 2 (TCN) Prediction: ${(prob * 100).toFixed(1)}% abandonment risk`);

            // Hybrid Check (The Golden Logic)
            if (prob > AI_THRESHOLD) {
                // COOLDOWN: If we just checked this second, don't spam
                const lastCheck = parseInt(sessionStorage.getItem('op_ecom_last_check') || '0');
                if (Date.now() - lastCheck < 5000) return; // 5s cooldown for demo

                console.log(`[OP-ECOM Tracker] Local AI detected high risk (${(prob * 100).toFixed(1)}%). Verifying buyer value...`);
                sessionStorage.setItem('op_ecom_last_check', Date.now().toString());

                const cart = JSON.parse(localStorage.getItem('shopDemo_cart') || '[]');
                const cartValue = cart.reduce((sum, item) => sum + ((item.price || 0) * (item.qty || 1)), 0);

                const result = await sendToAPI('/tracker/check-intent', {
                    session_id: sessionId, cart_value: cartValue,
                    local_abandonment_score: prob
                });

                if (result && result.should_intervene) {
                    console.log(`[OP-ECOM Tracker] GOLDEN LOGIC VERIFIED: Model1(TabM)=${(result.purchase_prob * 100).toFixed(1)}% purchase | Model2(TCN)=${(prob * 100).toFixed(1)}% risk => Intervention!`);
                    exitIntentChecked = true;
                    showInterventionPopup(prob);
                } else {
                    // Logic: Only log when threshold crossed, otherwise stays silent to prevent console noise
                    console.log(`[OP-ECOM Tracker] Business Logic: Intervention declined (Purchase Prob: ${(result?.purchase_prob * 100).toFixed(1)}%)`);
                }
            }
        } catch (e) {
            console.warn('[OP-ECOM Tracker] Local inference failed:', e);
        }
    }

    // Utility Functions
    function getBrowserInfo() {
        const ua = navigator.userAgent;
        if (ua.includes('Chrome')) return 'Chrome';
        if (ua.includes('Firefox')) return 'Firefox';
        if (ua.includes('Safari')) return 'Safari';
        return 'Other';
    }

    function getOS() {
        const ua = navigator.userAgent;
        if (ua.includes('Windows')) return 'Windows';
        if (ua.includes('Mac')) return 'MacOS';
        return 'Other';
    }

    function getPageType() {
        const path = window.location.pathname.toLowerCase();
        if (path === '/' || path === '' || path.includes('index.html')) return 'Home';
        if (path.includes('about') || path.includes('contact')) return 'Informational';
        if (path.includes('cart') || path.includes('checkout') || path.includes('account')) return 'Administrative';
        return 'ProductRelated';
    }

    async function sendToAPI(endpoint, data) {
        try {
            const response = await fetch(`${API_URL}${endpoint}`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            return await response.json();
        } catch (error) { return null; }
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
            visitor_type: visitorType, browser: getBrowserInfo(),
            operating_system: getOS(), region: 1, traffic_type: 1
        });

        if (result?.session_id) {
            sessionId = result.session_id;
            localStorage.setItem(SESSION_KEY, sessionId);
            pageHistory = []; // Wipe history on new session
            localStorage.setItem('op_ecom_history', '[]');
            console.log('[OP-ECOM Tracker] New session started. History cleared.');
        }
    }

    async function endSession() {
        if (!sessionId) return;
        await trackPageView();
        await sendToAPI('/tracker/session/end', { session_id: sessionId });
        localStorage.removeItem(SESSION_KEY);
        localStorage.removeItem('op_ecom_history');
    }

    async function trackPageView() {
        if (!sessionId) return;
        const duration = (Date.now() - currentPageStart) / 1000;
        // Map frontend page types to backend PageTypeEnum values
        const pageType = getPageType();
        const BACKEND_PAGE_TYPE = {
            'Home': 'ProductRelated', 'Product': 'ProductRelated',
            'ProductDetail': 'ProductRelated', 'ProductRelated': 'ProductRelated',
            'About': 'Informational', 'Informational': 'Informational',
            'Cart': 'Administrative', 'Checkout': 'Administrative',
            'Account': 'Administrative', 'Administrative': 'Administrative'
        };
        await sendToAPI('/tracker/pageview', {
            session_id: sessionId, page_type: BACKEND_PAGE_TYPE[pageType] || 'ProductRelated',
            page_url: window.location.href, page_title: document.title,
            duration_seconds: duration, is_bounce: duration < 2,
            is_exit: false, page_value: 0, scroll_depth: 0
        });
    }

    async function trackEvent(eventType, eventCategory, eventLabel, eventValue, eventData) {
        if (!sessionId) return;

        // Add significant events to TCN history
        if (eventType === 'add_to_cart') {
            pageHistory.push({ type: 'AddToCartEvent', duration: 0.5 });
            localStorage.setItem('op_ecom_history', JSON.stringify(pageHistory));
        }

        await sendToAPI('/tracker/event', {
            session_id: sessionId, event_type: eventType,
            event_category: eventCategory, event_label: eventLabel,
            event_value: eventValue, event_data: eventData
        });
    }

    async function trackPurchase(orderValue) {
        if (!sessionId) return;
        await sendToAPI('/tracker/purchase', { session_id: sessionId, order_value: orderValue });
        endSession();
    }

    // UI Intervention
    function showInterventionPopup(prob) {
        if (document.getElementById('op-ecom-overlay')) return;

        const overlay = document.createElement('div');
        overlay.id = 'op-ecom-overlay';
        overlay.style = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.6);backdrop-filter:blur(5px);z-index:9999;display:flex;align-items:center;justify-content:center;font-family:inherit;animate:fadeIn 0.3s;';

        const cartValue = localStorage.getItem('shopDemo_cart_total') || '0.00';

        overlay.innerHTML = `
            <div id="op-ecom-popup" style="background:#fff;padding:2.5rem;border-radius:24px;width:90%;max-width:420px;text-align:center;box-shadow:0 25px 50px -12px rgba(0,0,0,0.25);position:relative;">
                <div style="position:absolute;top:1rem;right:1rem;cursor:pointer;font-size:1.2rem;color:#94a3b8;" id="op-ecom-close">✕</div>
                <div style="width:70px;height:70px;background:#fff7ed;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:2rem;margin:0 auto 1.5rem;box-shadow:inset 0 2px 4px 0 rgba(0,0,0,0.06);">🎁</div>
                <h2 style="color:#1e293b;margin-bottom:0.75rem;font-size:1.75rem;font-weight:800;letter-spacing:-0.025em;">Wait! Don't Go...</h2>
                <p style="color:#64748b;margin-bottom:2rem;line-height:1.6;font-size:1.05rem;">We noticed you have items in your cart. Get <strong>20% OFF</strong> your order if you finish now!</p>
                <div style="background:#f8fafc;padding:1.25rem;border-radius:16px;margin-bottom:2rem;border:1px solid #f1f5f9;">
                    <input type="email" id="op-ecom-email" placeholder="Enter your email" style="width:100%;padding:12px;border:1px solid #e2e8f0;border-radius:12px;margin-bottom:1rem;outline:none;">
                    <button id="op-ecom-claim" style="background:linear-gradient(to right, #f97316, #fb923c);color:#fff;border:none;padding:14px;border-radius:12px;cursor:pointer;width:100%;font-weight:700;font-size:1rem;box-shadow:0 10px 15px -3px rgba(249, 115, 22, 0.3);">Claim My Discount</button>
                </div>
            </div>
        `;

        document.body.appendChild(overlay);

        document.getElementById('op-ecom-claim').onclick = async () => {
            const email = document.getElementById('op-ecom-email').value;
            if (!email || !email.includes('@')) return;
            const res = await sendToAPI('/tracker/email-capture', { session_id: sessionId, email, cart_value: parseFloat(cartValue) });
            if (res?.success) overlay.remove();
        };
        document.getElementById('op-ecom-close').onclick = () => overlay.remove();
    }

    async function init() {
        // CLEANUP: If history is flooded with identical high-risk signals, clear it
        if (pageHistory.length > 5) {
            const firstType = pageHistory[0].type;
            if (pageHistory.every(h => h.type === firstType && h.duration < 1)) {
                pageHistory = [];
                localStorage.setItem('op_ecom_history', '[]');
            }
        }

        await startSession();
        await initModel();

        // Record the current page load immediately so the AI sees it
        updatePageHistory();

        setInterval(runLocalInference, AI_POLL_INTERVAL);

        // Save page history BEFORE navigation so the next page has the full sequence
        window.addEventListener('beforeunload', () => {
            updatePageHistory();
            trackPageView();
        });
        window.addEventListener('pagehide', endSession);
    }

    // Expose tracker API to the window scope so external code
    // (e.g., demo site's addToCart button) can report events
    window.opEcomTracker = {
        trackEvent: trackEvent,
        trackPurchase: trackPurchase
    };

    init();
})();
