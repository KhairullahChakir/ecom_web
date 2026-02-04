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
    const AI_POLL_INTERVAL = 5000; // Check AI prediction every 5 seconds
    const AI_THRESHOLD = 0.60; // Trigger popup when abandonment probability > 60% (balanced)

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
        // Specifically detect home page
        if (path === '/' || path === '' || path.includes('index.html')) {
            return 'Home';
        }
        if (path.includes('account') || path.includes('cart') || path.includes('checkout') || path.includes('settings')) {
            return 'Administrative';
        }
        if (path.includes('about') || path.includes('contact') || path.includes('faq') || path.includes('help')) {
            return 'Informational';
        }
        return 'ProductRelated';
    }

    function getPageValue() {
        // 1. Check if we are on the Cart page (High Intent)
        if (window.location.pathname.includes('cart.html')) {
            return 20.0; // High value for being in the cart
        }

        // 2. Check for product detail page container
        const detailContainer = document.getElementById('product-detail');
        if (detailContainer && detailContainer.getAttribute('data-page-value')) {
            const val = parseFloat(detailContainer.getAttribute('data-page-value'));
            console.log(`[OP-ECOM Tracker] Found Detail PageValue: ${val}`);
            return val;
        }

        // 2. Try to find the first product card in view or any card with value
        const cards = document.querySelectorAll('[data-page-value]');
        if (cards.length > 0) {
            // If many cards (like grid), taking an average or the first one
            // In UCI, it's the value of the page. Grid page usually has 0 value in UCI,
            // but for demo we can assign it the avg of products on it.
            let sum = 0;
            cards.forEach(c => sum += parseFloat(c.getAttribute('data-page-value') || 0));
            const avg = sum / cards.length;
            console.log(`[OP-ECOM Tracker] Found Grid Avg PageValue: ${avg}`);
            return avg;
        }
        return 0;
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
        // Check for existing session (localStorage for cross-tab persistence)
        const existingSession = localStorage.getItem(SESSION_KEY);
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
            localStorage.setItem(SESSION_KEY, sessionId);
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

        localStorage.removeItem(SESSION_KEY);
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

        await sendToAPI('/tracker/purchase', {
            session_id: sessionId,
            order_value: orderValue || 0
        });

        console.log('[OP-ECOM Tracker] Purchase tracked! Session:', sessionId, 'Value:', orderValue);
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

        // NEW: Avoid triggering if the user has been on the page for less than 0.5 seconds
        // This prevents immediate popups on refresh/landing
        const dwellTime = (Date.now() - currentPageStart) / 1000;
        if (dwellTime < 0.5) {
            console.log(`[OP-ECOM Tracker] Exit intent ignored: session too short (${dwellTime.toFixed(1)}s)`);
            return;
        }

        exitIntentChecked = true; // Check only once per session/page load to avoid spam

        console.log('[OP-ECOM Tracker] Checking exit intent...');
        const result = await sendToAPI('/tracker/check-intent', {
            session_id: sessionId
        });

        console.log('[OP-ECOM Tracker] Intent Result:', result);

        if (result && result.should_intervene) {
            showInterventionPopup(result.probability);
        } else {
            console.log('[OP-ECOM Tracker] Probability too low for intervention:', result?.probability);
        }
    }

    function showInterventionPopup(probability, discountPercent) {
        // Track that popup was shown (event_type must match analytics query)
        trackEvent('exit_intent_shown', 'intervention', 'popup_displayed', Math.round(probability * 100));

        // Get cart value for personalized discount
        const cart = JSON.parse(localStorage.getItem('cart') || '[]');
        const cartValue = cart.reduce((sum, item) => sum + (item.price || 0), 0);

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

        // Step 1: Email capture form
        popup.innerHTML = `
            <div style="font-size: 3rem; margin-bottom: 1rem;">🎁</div>
            <h2 style="color: #1E4FA8; margin-bottom: 0.5rem;">Wait! Don't Go!</h2>
            <p style="color: #666; margin-bottom: 1.5rem;">
                Enter your email to unlock an <strong>exclusive discount</strong> just for you!
            </p>
            <input type="email" id="op-ecom-email" placeholder="your@email.com" style="
                width: 100%; padding: 12px; font-size: 1rem; border: 2px solid #ddd;
                border-radius: 8px; margin-bottom: 1rem; box-sizing: border-box;
            " />
            <button id="op-ecom-submit" style="
                background: #1E4FA8; color: white; border: none; padding: 12px 24px;
                font-size: 1.1rem; border-radius: 8px; cursor: pointer; width: 100%;
                transition: transform 0.2s;"
            >Get My Discount</button>
            <button id="op-ecom-close" style="
                background: transparent; border: none; color: #999; margin-top: 1rem;
                cursor: pointer; text-decoration: underline;"
            >No thanks, I'll pay full price</button>
        `;

        overlay.appendChild(popup);
        document.body.appendChild(overlay);

        // Handle email submission
        document.getElementById('op-ecom-submit').onclick = async () => {
            const emailInput = document.getElementById('op-ecom-email');
            const email = emailInput.value.trim();

            if (!email || !email.includes('@')) {
                emailInput.style.borderColor = 'red';
                return;
            }

            // Call email-capture API
            const result = await sendToAPI('/tracker/email-capture', {
                session_id: sessionId,
                email: email,
                cart_value: cartValue
            });

            if (result && result.success) {
                // Step 2: Show discount code
                popup.innerHTML = `
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🎉</div>
                    <h2 style="color: #10b981; margin-bottom: 0.5rem;">Your Exclusive Discount!</h2>
                    <p style="color: #666; margin-bottom: 1.5rem;">
                        Here's your <strong>${result.discount_percent}% OFF</strong> code:
                    </p>
                    <div style="background: #eef2ff; padding: 15px; border-radius: 8px; margin-bottom: 1.5rem;">
                        <code style="font-size: 1.5rem; color: #1E4FA8; font-weight: bold; user-select: all;">
                            ${result.discount_code}
                        </code>
                    </div>
                    <p style="color: #999; font-size: 0.85rem; margin-bottom: 1.5rem;">
                        Valid for 24 hours • We've also sent it to ${email}
                    </p>
                    <button id="op-ecom-done" style="
                        background: #10b981; color: white; border: none; padding: 12px 24px;
                        font-size: 1.1rem; border-radius: 8px; cursor: pointer; width: 100%;"
                    >Continue Shopping</button>
                `;

                document.getElementById('op-ecom-done').onclick = () => {
                    trackEvent('discount_claimed', 'intervention', 'claim_discount', result.discount_percent);
                    overlay.remove();
                };
            }
        };

        document.getElementById('op-ecom-close').onclick = () => {
            trackEvent('click', 'intervention', 'dismiss', 0);
            overlay.remove();
        };
    }

    // NEW: Proactive AI-based exit detection (replaces mouse-based detection)
    let aiPollInterval = null;

    function setupProactiveAI() {
        // Start polling the AI every few seconds
        aiPollInterval = setInterval(async () => {
            if (!sessionId || exitIntentChecked) return;

            // Make sure user has been on page for at least 3 seconds
            const dwellTime = (Date.now() - currentPageStart) / 1000;
            if (dwellTime < 3) return;

            console.log('[OP-ECOM Tracker] AI Polling: Checking abandonment risk...');

            const result = await sendToAPI('/tracker/check-intent', {
                session_id: sessionId
            });

            if (result && result.abandonment_prob) {
                console.log(`[OP-ECOM Tracker] AI Prediction: ${(result.abandonment_prob * 100).toFixed(1)}% abandonment risk`);

                // If AI detects high abandonment risk, trigger intervention
                if (result.abandonment_prob > AI_THRESHOLD && result.should_intervene) {
                    console.log('[OP-ECOM Tracker] AI DETECTED HIGH RISK - Triggering intervention!');
                    exitIntentChecked = true;
                    clearInterval(aiPollInterval); // Stop polling after intervention
                    showInterventionPopup(result.probability);
                }
            }
        }, AI_POLL_INTERVAL);

        console.log(`[OP-ECOM Tracker] Proactive AI monitoring started (polling every ${AI_POLL_INTERVAL / 1000}s)`);
    }

    // Initialize
    async function init() {
        await startSession();

        // Start proactive AI-based detection (pure AI, no mouse events)
        setupProactiveAI();

        // Track page views on navigation
        window.addEventListener('beforeunload', () => {
            trackPageView();
            if (aiPollInterval) clearInterval(aiPollInterval);
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

        // End session on page close (modern)
        window.addEventListener('pagehide', endSession);

        console.log('[OP-ECOM Tracker] Initialized with PROACTIVE AI detection. AI will predict abandonment automatically.');
    }

    // Expose global API
    window.opEcomTracker = {
        trackEvent,
        trackPurchase,
        getSessionId: () => sessionId,
        checkExitIntent: () => {
            exitIntentChecked = false; // Reset lock for manual test
            checkExitIntent();
        }
    };

    // Start tracking
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
