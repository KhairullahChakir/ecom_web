"use client";

import { useState } from "react";

interface PredictResponse {
  label: string;
  probability: number;
  latency_ms: number;
}

// Sample sessions from the UCI dataset
const sampleSessions = [
  {
    name: "High Intent Buyer",
    data: {
      administrative: 3,
      administrative_duration: 80.5,
      informational: 1,
      informational_duration: 35.0,
      product_related: 45,
      product_related_duration: 2500.0,
      bounce_rates: 0.01,
      exit_rates: 0.02,
      page_values: 25.5,
      special_day: 0.0,
      month: "Nov",
      operating_systems: 2,
      browser: 2,
      region: 1,
      traffic_type: 2,
      visitor_type: "Returning_Visitor",
      weekend: false,
    },
  },
  {
    name: "Low Intent Browser",
    data: {
      administrative: 0,
      administrative_duration: 0.0,
      informational: 0,
      informational_duration: 0.0,
      product_related: 5,
      product_related_duration: 120.0,
      bounce_rates: 0.15,
      exit_rates: 0.2,
      page_values: 0.0,
      special_day: 0.0,
      month: "Feb",
      operating_systems: 1,
      browser: 1,
      region: 3,
      traffic_type: 1,
      visitor_type: "New_Visitor",
      weekend: true,
    },
  },
  {
    name: "Special Day Shopper",
    data: {
      administrative: 2,
      administrative_duration: 45.0,
      informational: 2,
      informational_duration: 60.0,
      product_related: 30,
      product_related_duration: 1800.0,
      bounce_rates: 0.02,
      exit_rates: 0.03,
      page_values: 10.0,
      special_day: 0.8,
      month: "May",
      operating_systems: 3,
      browser: 2,
      region: 2,
      traffic_type: 3,
      visitor_type: "Returning_Visitor",
      weekend: false,
    },
  },
];

export default function Home() {
  const [selectedSession, setSelectedSession] = useState(0);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(sampleSessions[selectedSession].data),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data: PredictResponse = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to connect to API");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="gradient-bg text-white py-6 px-4">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-2">🛒 OP-ECOM</h1>
          <p className="text-blue-100 text-lg">
            Online Shoppers Purchase Prediction API
          </p>
        </div>
      </header>

      {/* Hero Section */}
      <section className="gradient-bg text-white pb-20 pt-10 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl font-bold mb-4">
            Predict Purchase Intent in Real-Time
          </h2>
          <p className="text-xl text-blue-100 max-w-2xl mx-auto">
            Powered by TabM deep learning model optimized for &lt;10ms CPU inference.
            Analyze user sessions and predict conversion probability instantly.
          </p>
        </div>
      </section>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 -mt-10">
        {/* Prediction Card */}
        <div className="glass-card rounded-2xl p-8 mb-8">
          <h3 className="text-2xl font-bold mb-6 text-[var(--lajawardi-dark)]">
            🎯 Try a Prediction
          </h3>

          {/* Session Selector */}
          <div className="mb-6">
            <label className="block text-sm font-semibold mb-3 text-[var(--foreground)]">
              Select a Sample Session:
            </label>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {sampleSessions.map((session, index) => (
                <button
                  key={index}
                  onClick={() => setSelectedSession(index)}
                  className={`p-4 rounded-xl text-left transition-all ${selectedSession === index
                      ? "bg-[var(--lajawardi-primary)] text-white shadow-lg"
                      : "bg-[var(--lajawardi-light)] hover:bg-[var(--lajawardi-primary)] hover:text-white"
                    }`}
                >
                  <div className="font-semibold">{session.name}</div>
                  <div className="text-sm opacity-80 mt-1">
                    {session.data.product_related} pages • {session.data.visitor_type.replace("_", " ")}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Session Details */}
          <div className="mb-6 p-4 bg-[var(--lajawardi-light)] rounded-xl">
            <h4 className="font-semibold mb-3 text-[var(--lajawardi-dark)]">
              Session Features:
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <div>
                <span className="text-[var(--lajawardi-dark)] font-medium">Product Pages:</span>{" "}
                {sampleSessions[selectedSession].data.product_related}
              </div>
              <div>
                <span className="text-[var(--lajawardi-dark)] font-medium">Duration:</span>{" "}
                {Math.round(sampleSessions[selectedSession].data.product_related_duration)}s
              </div>
              <div>
                <span className="text-[var(--lajawardi-dark)] font-medium">Bounce Rate:</span>{" "}
                {(sampleSessions[selectedSession].data.bounce_rates * 100).toFixed(1)}%
              </div>
              <div>
                <span className="text-[var(--lajawardi-dark)] font-medium">Page Value:</span>{" "}
                ${sampleSessions[selectedSession].data.page_values.toFixed(2)}
              </div>
              <div>
                <span className="text-[var(--lajawardi-dark)] font-medium">Month:</span>{" "}
                {sampleSessions[selectedSession].data.month}
              </div>
              <div>
                <span className="text-[var(--lajawardi-dark)] font-medium">Visitor:</span>{" "}
                {sampleSessions[selectedSession].data.visitor_type.replace("_", " ")}
              </div>
              <div>
                <span className="text-[var(--lajawardi-dark)] font-medium">Weekend:</span>{" "}
                {sampleSessions[selectedSession].data.weekend ? "Yes" : "No"}
              </div>
              <div>
                <span className="text-[var(--lajawardi-dark)] font-medium">Special Day:</span>{" "}
                {sampleSessions[selectedSession].data.special_day}
              </div>
            </div>
          </div>

          {/* Predict Button */}
          <button
            onClick={handlePredict}
            disabled={loading}
            className="btn-primary w-full text-lg"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <span className="animate-pulse">⏳</span> Predicting...
              </span>
            ) : (
              "🚀 Predict Purchase Intent"
            )}
          </button>

          {/* Error Message */}
          {error && (
            <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-xl text-red-700">
              <strong>Error:</strong> {error}
              <p className="text-sm mt-1 text-red-600">
                Make sure the backend is running: <code>uvicorn app.main:app --reload</code>
              </p>
            </div>
          )}

          {/* Result Card */}
          {result && (
            <div className="mt-8 animate-slide-up">
              <div className="glass-card rounded-xl p-6 border-2 border-[var(--lajawardi-primary)]">
                <h4 className="text-lg font-semibold mb-4 text-[var(--lajawardi-dark)]">
                  📊 Prediction Result
                </h4>
                <div className="flex flex-col md:flex-row items-center justify-between gap-6">
                  {/* Label */}
                  <div className="text-center">
                    <div className="text-sm text-gray-500 mb-2">Will Purchase?</div>
                    <span className={result.label === "YES" ? "badge-yes" : "badge-no"}>
                      {result.label}
                    </span>
                  </div>

                  {/* Probability */}
                  <div className="text-center">
                    <div className="text-sm text-gray-500 mb-2">Probability</div>
                    <div className="text-4xl font-bold text-[var(--lajawardi-primary)]">
                      {(result.probability * 100).toFixed(1)}%
                    </div>
                  </div>

                  {/* Latency */}
                  <div className="text-center">
                    <div className="text-sm text-gray-500 mb-2">Latency</div>
                    <div className="text-2xl font-bold text-[var(--lajawardi-dark)]">
                      {result.latency_ms.toFixed(2)} ms
                    </div>
                    <div className="text-xs text-green-600 mt-1">
                      ⚡ {result.latency_ms < 10 ? "Under target!" : "Optimizing..."}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* API Info Card */}
        <div className="glass-card rounded-2xl p-8 mb-8">
          <h3 className="text-2xl font-bold mb-6 text-[var(--lajawardi-dark)]">
            📡 API Endpoints
          </h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="p-4 bg-[var(--lajawardi-light)] rounded-xl">
              <div className="flex items-center gap-2 mb-2">
                <span className="px-2 py-1 bg-green-500 text-white text-xs font-bold rounded">GET</span>
                <code className="font-mono text-[var(--lajawardi-dark)]">/health</code>
              </div>
              <p className="text-sm text-gray-600">Health check endpoint</p>
            </div>
            <div className="p-4 bg-[var(--lajawardi-light)] rounded-xl">
              <div className="flex items-center gap-2 mb-2">
                <span className="px-2 py-1 bg-blue-500 text-white text-xs font-bold rounded">POST</span>
                <code className="font-mono text-[var(--lajawardi-dark)]">/predict</code>
              </div>
              <p className="text-sm text-gray-600">Get prediction with label, probability, latency</p>
            </div>
          </div>
        </div>

        {/* Features Card */}
        <div className="glass-card rounded-2xl p-8 mb-8">
          <h3 className="text-2xl font-bold mb-6 text-[var(--lajawardi-dark)]">
            ✨ Features
          </h3>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="text-4xl mb-3">⚡</div>
              <h4 className="font-bold mb-2">Fast Inference</h4>
              <p className="text-sm text-gray-600">ONNX Runtime optimized for &lt;10ms CPU inference</p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-3">🧠</div>
              <h4 className="font-bold mb-2">TabM Model</h4>
              <p className="text-sm text-gray-600">State-of-the-art deep learning for tabular data</p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-3">📊</div>
              <h4 className="font-bold mb-2">Real-time Metrics</h4>
              <p className="text-sm text-gray-600">Latency tracking with mean, p95, p99 stats</p>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="gradient-bg text-white py-8 mt-12">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <p className="text-blue-100">
            OP-ECOM • Online Shoppers Purchase Prediction API
          </p>
          <p className="text-sm text-blue-200 mt-2">
            UCI Dataset: 12,330 sessions • 18 features • Target: Revenue
          </p>
        </div>
      </footer>
    </div>
  );
}
