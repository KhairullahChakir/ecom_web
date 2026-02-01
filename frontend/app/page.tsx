'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Activity, ShoppingCart, Zap } from 'lucide-react';
import PurchaseForm from './components/PurchaseForm';
import PredictionGauge from './components/PredictionGauge';

export default function Home() {
  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  return (
    <main className="min-h-screen p-8 md:p-12 text-gray-100 flex flex-col items-center">
      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-6xl flex justify-between items-center mb-12"
      >
        <div className="flex items-center gap-3">
          <div className="p-3 rounded-xl bg-blue-600/20 border border-blue-500/30 neon-border">
            <Zap className="w-8 h-8 text-blue-400" />
          </div>
          <div>
            <h1 className="text-3xl font-bold tracking-tight">OP-ECOM <span className="text-blue-400">Intelligence</span></h1>
            <p className="text-gray-400 text-sm">Real-time Shopper Intent Analysis</p>
          </div>
        </div>

        <div className="hidden md:flex gap-6 text-sm font-medium text-gray-400">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-green-400" />
            <span>Model: TabM (K=1)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span>System Online</span>
          </div>
        </div>
      </motion.header>

      {/* Content Grid */}
      <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-12 gap-8">

        {/* Left: Input Form */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="lg:col-span-7"
        >
          <div className="glass-panel rounded-3xl p-8 h-full">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold flex items-center gap-2">
                <ShoppingCart className="w-5 h-5 text-purple-400" />
                User Behavior Data
              </h2>
              <span className="text-xs px-3 py-1 rounded-full bg-white/5 border border-white/10">Active Session</span>
            </div>

            <PurchaseForm setPrediction={setPrediction} setLoading={setLoading} />
          </div>
        </motion.div>

        {/* Right: Visualization */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="lg:col-span-5 flex flex-col gap-6"
        >
          {/* Main Gauge */}
          <div className="glass-panel rounded-3xl p-8 flex-1 min-h-[400px] flex flex-col items-center justify-center relative overflow-hidden">
            <div className="absolute inset-0 bg-blue-500/5 blur-3xl" />
            <PredictionGauge prediction={prediction} loading={loading} />
          </div>

          {/* Stats Card */}
          <div className="glass-panel rounded-2xl p-6 flex flex-row justify-between items-center">
            <div>
              <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">Inference Latency</p>
              <div className="text-2xl font-mono font-bold text-green-400">
                {prediction?.inference_latency_ms ? `${prediction.inference_latency_ms.toFixed(4)}` : '0.0000'} <span className="text-sm text-gray-500">ms</span>
              </div>
            </div>
            <div className="h-10 w-[1px] bg-white/10" />
            <div>
              <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">Total Request</p>
              <div className="text-2xl font-mono font-bold">
                {prediction?.total_latency_ms ? `${prediction.total_latency_ms.toFixed(2)}` : '0.00'} <span className="text-sm text-gray-500">ms</span>
              </div>
            </div>
          </div>
        </motion.div>

      </div>
    </main>
  );
}
