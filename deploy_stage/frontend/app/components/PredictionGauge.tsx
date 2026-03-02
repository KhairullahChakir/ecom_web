'use client';

import { motion } from 'framer-motion';
import { Loader2, CheckCircle, XCircle, AlertCircle } from 'lucide-react';

interface Props {
    prediction: any;
    loading: boolean;
}

export default function PredictionGauge({ prediction, loading }: Props) {

    if (loading) {
        return (
            <div className="flex flex-col items-center justify-center text-blue-400">
                <Loader2 className="w-16 h-16 animate-spin mb-4" />
                <p className="text-sm font-mono tracking-widest animate-pulse">PROCESSING</p>
            </div>
        );
    }

    if (!prediction) {
        return (
            <div className="text-center text-gray-500">
                <div className="w-40 h-40 rounded-full border-4 border-white/5 flex items-center justify-center mx-auto mb-6">
                    <span className="text-3xl font-thin text-white/10">--%</span>
                </div>
                <p className="text-sm">Enter data to generate prediction</p>
            </div>
        );
    }

    // Handle Error State
    if (prediction.error) {
        return (
            <div className="flex flex-col items-center justify-center text-red-400">
                <AlertCircle className="w-16 h-16 mb-4" />
                <p className="text-lg font-bold">Prediction Failed</p>
                <p className="text-xs text-red-400/60 mt-2">Check console for details</p>
            </div>
        );
    }

    const isBuy = prediction.label === 'YES';
    let prob = prediction.probability * 100;

    // Safety check for NaN
    if (isNaN(prob)) prob = 0;

    return (
        <div className="flex flex-col items-center relative z-10 w-full">
            {/* Gauge Circle */}
            <div className="relative w-64 h-64 flex items-center justify-center mb-8">
                {/* Background Track */}
                <svg className="absolute inset-0 w-full h-full transform -rotate-90">
                    <circle cx="128" cy="128" r="110" stroke="currentColor" strokeWidth="12" fill="transparent" className="text-white/5" />
                    {/* Progress Arc */}
                    <motion.circle
                        initial={{ strokeDasharray: "0 1000" }}
                        animate={{ strokeDasharray: `${prob * 6.9} 1000` }}
                        transition={{ duration: 1.5, ease: "easeOut" }}
                        cx="128" cy="128" r="110"
                        stroke="currentColor"
                        strokeWidth="12"
                        fill="transparent"
                        strokeLinecap="round"
                        className={isBuy ? "text-green-500 shadow-[0_0_20px_rgba(34,197,94,0.5)]" : "text-red-500"}
                    />
                </svg>

                {/* Center Content */}
                <div className="flex flex-col items-center">
                    <motion.span
                        initial={{ scale: 0.5, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        className="text-6xl font-bold tracking-tighter"
                    >
                        {prob.toFixed(1)}%
                    </motion.span>
                    <span className="text-xs text-gray-400 uppercase tracking-widest mt-2">Probability</span>
                </div>
            </div>

            {/* Result Card */}
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                className={`w-full max-w-sm rounded-xl border p-6 flex items-center gap-4 ${isBuy
                    ? 'bg-green-500/10 border-green-500/30'
                    : 'bg-red-500/10 border-red-500/30'
                    }`}
            >
                <div className={`p-3 rounded-full ${isBuy ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                    {isBuy ? <CheckCircle className="w-8 h-8" /> : <XCircle className="w-8 h-8" />}
                </div>
                <div>
                    <h3 className={`text-xl font-bold ${isBuy ? 'text-green-400' : 'text-red-400'}`}>
                        {isBuy ? "Purchase Likely" : "No Purchase"}
                    </h3>
                    <p className="text-sm text-gray-300 opacity-80">
                        {isBuy
                            ? "High confidence customer shows strong intent."
                            : "Customer shows low engagement signals."}
                    </p>
                </div>
            </motion.div>
        </div>
    );
}
