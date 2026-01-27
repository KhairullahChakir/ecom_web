"use client";

import React, { useState, useEffect } from "react";
import Head from "next/head";

// --- Types ---

interface PredictionRequest {
  administrative: number;
  administrative_duration: number;
  informational: number;
  informational_duration: number;
  product_related: number;
  product_related_duration: number;
  bounce_rates: number;
  exit_rates: number;
  page_values: number;
  special_day: number;
  month: string;
  operating_systems: number;
  browser: number;
  region: number;
  traffic_type: number;
  visitor_type: string;
  weekend: boolean;
}

interface PredictionResponse {
  label: string;
  probability: number;
  latency_ms: number;
}

// --- Sample Data ---

const SAMPLES = [
  {
    name: "High Intent (Returning)",
    data: {
      administrative: 10,
      administrative_duration: 150,
      informational: 2,
      informational_duration: 40,
      product_related: 45,
      product_related_duration: 1800,
      bounce_rates: 0,
      exit_rates: 0.01,
      page_values: 65.5,
      special_day: 0,
      month: "Nov",
      operating_systems: 2,
      browser: 2,
      region: 1,
      traffic_type: 2,
      visitor_type: "Returning_Visitor",
      weekend: true,
    }
  },
  {
    name: "Casual Browser",
    data: {
      administrative: 2,
      administrative_duration: 20,
      informational: 0,
      informational_duration: 0,
      product_related: 15,
      product_related_duration: 300,
      bounce_rates: 0.05,
      exit_rates: 0.08,
      page_values: 0,
      special_day: 0,
      month: "May",
      operating_systems: 1,
      browser: 1,
      region: 3,
      traffic_type: 1,
      visitor_type: "New_Visitor",
      weekend: false,
    }
  },
  {
    name: "High Bounce Rate",
    data: {
      administrative: 0,
      administrative_duration: 0,
      informational: 0,
      informational_duration: 0,
      product_related: 1,
      product_related_duration: 0,
      bounce_rates: 0.2,
      exit_rates: 0.2,
      page_values: 0,
      special_day: 0,
      month: "Feb",
      operating_systems: 3,
      browser: 2,
      region: 1,
      traffic_type: 3,
      visitor_type: "Returning_Visitor",
      weekend: false,
    }
  }
];

// --- Components ---

const InputGroup = ({ label, children }: { label: string; children: React.ReactNode }) => (
  <div className="flex flex-col gap-1">
    <label className="text-sm font-semibold text-slate-500 uppercase tracking-wider">{label}</label>
    {children}
  </div>
);

const Gauge = ({ value }: { value: number }) => {
  const percentage = value * 100;
  const color = value > 0.5 ? "var(--success)" : "var(--danger)";

  return (
    <div className="relative w-48 h-48 flex items-center justify-center">
      <svg className="w-full h-full transform -rotate-90">
        <circle
          cx="96"
          cy="96"
          r="80"
          fill="transparent"
          stroke="var(--lajawardi-light)"
          strokeWidth="12"
          className="opacity-20"
        />
        <circle
          cx="96"
          cy="96"
          r="80"
          fill="transparent"
          stroke={color}
          strokeWidth="12"
          strokeDasharray={502.6}
          strokeDashoffset={502.6 - (502.6 * value)}
          strokeLinecap="round"
          className="transition-all duration-1000 ease-out"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-4xl font-bold" style={{ color }}>{Math.round(percentage)}%</span>
        <span className="text-xs text-slate-400 font-medium">PROBABILITY</span>
      </div>
    </div>
  );
};

export default function Home() {
  const [formData, setFormData] = useState<PredictionRequest>(SAMPLES[0].data);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    let finalValue: any = value;

    if (type === "number") {
      finalValue = parseFloat(value);
    } else if (type === "checkbox") {
      finalValue = (e.target as HTMLInputElement).checked;
    }

    setFormData(prev => ({ ...prev, [name]: finalValue }));
  };

  const loadSample = (sample: any) => {
    setFormData(sample.data);
    setResult(null);
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      if (!response.ok) throw new Error("API Connection Failed");

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError("Could not reach backend API. Make sure FastAPI is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen pb-20">
      {/* Header Section */}
      <header className="gradient-bg py-12 px-6 mb-8 text-white text-center shadow-xl">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-5xl font-extrabold mb-4 tracking-tight drop-shadow-md">
            OP-ECOM <span className="text-lajawardi-light">Predictor</span>
          </h1>
          <p className="text-xl opacity-90 font-light max-w-2xl mx-auto">
            High-performance deep learning (TabM) for predicting online shopper conversion in milliseconds.
          </p>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-6 grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Form */}
        <div className="lg:col-span-7">
          <section className="glass-card rounded-2xl p-8 mb-8 border border-white/20">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold flex items-center gap-2">
                <span className="w-8 h-8 rounded-lg gradient-bg flex items-center justify-center text-white text-sm">1</span>
                Session Configuration
              </h2>
              <div className="flex gap-2">
                {SAMPLES.map((s, i) => (
                  <button
                    key={i}
                    onClick={() => loadSample(s)}
                    className="text-xs px-3 py-1.5 rounded-full border border-lajawardi-primary/30 text-lajawardi-primary hover:bg-lajawardi-primary hover:text-white transition-all font-medium"
                  >
                    {s.name}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <InputGroup label="Administrative Pages">
                <input type="number" name="administrative" value={formData.administrative} onChange={handleInputChange} className="input-field" />
              </InputGroup>
              <InputGroup label="Admin Duration (sec)">
                <input type="number" name="administrative_duration" value={formData.administrative_duration} onChange={handleInputChange} className="input-field" />
              </InputGroup>

              <InputGroup label="Informational Pages">
                <input type="number" name="informational" value={formData.informational} onChange={handleInputChange} className="input-field" />
              </InputGroup>
              <InputGroup label="Info Duration (sec)">
                <input type="number" name="informational_duration" value={formData.informational_duration} onChange={handleInputChange} className="input-field" />
              </InputGroup>

              <InputGroup label="Product Related Pages">
                <input type="number" name="product_related" value={formData.product_related} onChange={handleInputChange} className="input-field" />
              </InputGroup>
              <InputGroup label="Product Duration (sec)">
                <input type="number" name="product_related_duration" value={formData.product_related_duration} onChange={handleInputChange} className="input-field" />
              </InputGroup>

              <InputGroup label="Bounce Rates">
                <input type="number" step="0.01" name="bounce_rates" value={formData.bounce_rates} onChange={handleInputChange} className="input-field" />
              </InputGroup>
              <InputGroup label="Exit Rates">
                <input type="number" step="0.01" name="exit_rates" value={formData.exit_rates} onChange={handleInputChange} className="input-field" />
              </InputGroup>

              <InputGroup label="Page Values">
                <input type="number" step="0.1" name="page_values" value={formData.page_values} onChange={handleInputChange} className="input-field" />
              </InputGroup>
              <InputGroup label="Special Day Proximity">
                <input type="number" step="0.1" name="special_day" value={formData.special_day} onChange={handleInputChange} className="input-field" />
              </InputGroup>

              <InputGroup label="Month">
                <select name="month" value={formData.month} onChange={handleInputChange} className="input-field">
                  {["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].map(m => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </InputGroup>
              <InputGroup label="Visitor Type">
                <select name="visitor_type" value={formData.visitor_type} onChange={handleInputChange} className="input-field">
                  <option value="Returning_Visitor">Returning Visitor</option>
                  <option value="New_Visitor">New Visitor</option>
                  <option value="Other">Other</option>
                </select>
              </InputGroup>

              <InputGroup label="Weekend">
                <div className="flex items-center gap-2 mt-2">
                  <input
                    type="checkbox"
                    name="weekend"
                    checked={formData.weekend}
                    onChange={handleInputChange}
                    className="w-5 h-5 accent-lajawardi-primary"
                  />
                  <span className="text-sm font-medium">Session on weekend?</span>
                </div>
              </InputGroup>
            </div>

            <button
              onClick={handlePredict}
              disabled={loading}
              className="btn-primary w-full mt-10 text-lg flex items-center justify-center gap-3"
            >
              {loading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/50 border-t-white rounded-full animate-spin" />
                  Calculating...
                </>
              ) : (
                <>Run TabM Inference 🚀</>
              )}
            </button>

            {error && (
              <p className="mt-4 text-danger text-sm text-center font-medium bg-red-50 p-3 rounded-lg border border-red-100 italic">
                {error}
              </p>
            )}
          </section>
        </div>

        {/* Right Column: Prediction Results */}
        <div className="lg:col-span-5 h-full">
          <section className="sticky top-8 flex flex-col gap-6">
            <h2 className="text-2xl font-bold flex items-center gap-2 text-slate-800">
              <span className="w-8 h-8 rounded-lg gradient-bg flex items-center justify-center text-white text-sm">2</span>
              Inference Insights
            </h2>

            {result ? (
              <div className="glass-card rounded-2xl p-8 border border-white/20 animate-slide-up flex flex-col items-center text-center">
                <div className="mb-8">
                  <Gauge value={result.probability} />
                </div>

                <h3 className="text-slate-400 text-sm font-bold uppercase tracking-widest mb-2">Prediction</h3>
                <div className={`badge-${result.label.toLowerCase()} mb-6`}>
                  {result.label === "YES" ? "PURCHASE INTENT DETECTED" : "NO CONVERSION LIKELY"}
                </div>

                <div className="grid grid-cols-2 gap-4 w-full">
                  <div className="bg-lajawardi-light/30 rounded-xl p-4 border border-lajawardi-primary/5">
                    <p className="text-xs text-slate-500 font-bold uppercase mb-1">Latency</p>
                    <p className="text-2xl font-black text-lajawardi-primary">{result.latency_ms} ms</p>
                  </div>
                  <div className="bg-slate-100 rounded-xl p-4 border border-slate-200">
                    <p className="text-xs text-slate-500 font-bold uppercase mb-1">Architecture</p>
                    <p className="text-xl font-bold text-slate-700">TabM-ONNX</p>
                  </div>
                </div>

                <div className="w-full mt-8 p-6 bg-slate-50 rounded-2xl border border-slate-200 text-left">
                  <h4 className="text-sm font-bold text-slate-600 mb-4 flex items-center gap-2">
                    <svg className="w-4 h-4 text-lajawardi-primary" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M10 12a2 2 0 100-4 2 2 0 000 4z" />
                      <path fillRule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd" />
                    </svg>
                    Top Influencing Factors
                  </h4>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-slate-500">Page Values</span>
                      <div className="h-2 w-32 bg-slate-200 rounded-full overflow-hidden">
                        <div className="h-full bg-lajawardi-primary w-[90%] rounded-full"></div>
                      </div>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-slate-500">Exit Rates</span>
                      <div className="h-2 w-32 bg-slate-200 rounded-full overflow-hidden">
                        <div className="h-full bg-lajawardi-primary w-[65%] rounded-full"></div>
                      </div>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-slate-500">ProductRelated</span>
                      <div className="h-2 w-32 bg-slate-200 rounded-full overflow-hidden">
                        <div className="h-full bg-lajawardi-primary w-[40%] rounded-full"></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="glass-card rounded-2xl p-12 border border-dashed border-slate-300 flex flex-col items-center justify-center text-center opacity-60">
                <div className="w-16 h-16 rounded-full bg-slate-100 flex items-center justify-center mb-4">
                  <svg className="w-8 h-8 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                </div>
                <p className="text-slate-500 font-medium italic">
                  Awaiting configuration... <br />
                  Set features and run prediction to see results.
                </p>
              </div>
            )}

            <div className="p-6 bg-lajawardi-dark/5 border border-lajawardi-primary/10 rounded-2xl">
              <h4 className="text-xs font-bold text-lajawardi-primary uppercase tracking-widest mb-2">Model Info</h4>
              <p className="text-sm text-slate-600 leading-relaxed font-medium">
                Our <span className="text-lajawardi-primary font-bold">TabM</span> deep learning model achieves 89.8% AUC-ROC, optimized for low-latency CPU production environments.
              </p>
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}
