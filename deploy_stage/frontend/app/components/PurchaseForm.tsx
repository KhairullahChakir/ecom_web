'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Loader2 } from 'lucide-react';

interface Props {
    setPrediction: (data: any) => void;
    setLoading: (loading: boolean) => void;
}

export default function PurchaseForm({ setPrediction, setLoading }: Props) {
    const [formData, setFormData] = useState({
        administrative: 0,
        administrative_duration: 0,
        informational: 0,
        informational_duration: 0,
        product_related: 0,
        product_related_duration: 0,
        bounce_rates: 0,
        exit_rates: 0,
        page_values: 0,
        special_day: 0,
        month: "Nov",
        operating_systems: 2,
        browser: 2,
        region: 1,
        traffic_type: 2,
        visitor_type: "Returning_Visitor",
        weekend: false
    });

    const handleChange = (e: any) => {
        const { name, value, type, checked } = e.target;
        let newValue: any = value;

        if (type === 'number') {
            const parsed = parseFloat(value);
            // Default to 0 if NaN to avoid React/API errors
            newValue = isNaN(parsed) ? 0 : parsed;
        } else if (type === 'checkbox') {
            newValue = checked;
        }

        setFormData(prev => ({
            ...prev,
            [name]: newValue
        }));
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setPrediction(null);

        try {
            const res = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData),
            });

            if (!res.ok) {
                throw new Error(`API Error: ${res.statusText}`);
            }

            const data = await res.json();

            // Artificial delay for UI effect (optional, because API is too fast!)
            setTimeout(() => {
                setPrediction(data);
                setLoading(false);
            }, 400);

        } catch (error) {
            console.error(error);
            setLoading(false);
            setPrediction({ error: true });
        }
    };

    const inputClass = "w-full bg-black/20 border border-white/10 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/50 transition-all";
    const labelClass = "block text-xs font-medium text-gray-400 mb-1.5 uppercase tracking-wide";

    return (
        <form onSubmit={handleSubmit} className="space-y-6">

            {/* Section 1: Page Activity */}
            <div className="space-y-4">
                <h3 className="text-sm font-semibold text-blue-400 border-b border-blue-500/10 pb-2">Page Activity</h3>
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <label className={labelClass}>Admin Pages</label>
                        <input type="number" name="administrative" value={formData.administrative} onChange={handleChange} className={inputClass} />
                    </div>
                    <div>
                        <label className={labelClass}>Duration (s)</label>
                        <input type="number" name="administrative_duration" value={formData.administrative_duration} onChange={handleChange} className={inputClass} />
                    </div>
                    <div>
                        <label className={labelClass}>Info Pages</label>
                        <input type="number" name="informational" value={formData.informational} onChange={handleChange} className={inputClass} />
                    </div>
                    <div>
                        <label className={labelClass}>Duration (s)</label>
                        <input type="number" name="informational_duration" value={formData.informational_duration} onChange={handleChange} className={inputClass} />
                    </div>
                    <div>
                        <label className={labelClass}>Product Pages</label>
                        <input type="number" name="product_related" value={formData.product_related} onChange={handleChange} className={inputClass} />
                    </div>
                    <div>
                        <label className={labelClass}>Duration (s)</label>
                        <input type="number" name="product_related_duration" value={formData.product_related_duration} onChange={handleChange} className={inputClass} />
                    </div>
                </div>
            </div>

            {/* Section 2: Metrics */}
            <div className="space-y-4">
                <h3 className="text-sm font-semibold text-blue-400 border-b border-blue-500/10 pb-2">User Metrics</h3>
                <div className="grid grid-cols-3 gap-4">
                    <div>
                        <label className={labelClass}>Bounce Rate</label>
                        <input type="number" step="0.01" name="bounce_rates" value={formData.bounce_rates} onChange={handleChange} className={inputClass} />
                    </div>
                    <div>
                        <label className={labelClass}>Exit Rate</label>
                        <input type="number" step="0.01" name="exit_rates" value={formData.exit_rates} onChange={handleChange} className={inputClass} />
                    </div>
                    <div>
                        <label className={labelClass}>Page Values</label>
                        <input type="number" step="0.1" name="page_values" value={formData.page_values} onChange={handleChange} className={inputClass} />
                    </div>
                </div>
            </div>

            {/* Section 3: Context */}
            <div className="space-y-4">
                <h3 className="text-sm font-semibold text-blue-400 border-b border-blue-500/10 pb-2">Context</h3>
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <label className={labelClass}>Month</label>
                        <select name="month" value={formData.month} onChange={handleChange} className={inputClass}>
                            {['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'].map(m => (
                                <option key={m} value={m} className="bg-gray-900">{m}</option>
                            ))}
                        </select>
                    </div>
                    <div>
                        <label className={labelClass}>Visitor Type</label>
                        <select name="visitor_type" value={formData.visitor_type} onChange={handleChange} className={inputClass}>
                            <option value="Returning_Visitor" className="bg-gray-900">Returning</option>
                            <option value="New_Visitor" className="bg-gray-900">New</option>
                            <option value="Other" className="bg-gray-900">Other</option>
                        </select>
                    </div>
                </div>
                <div className="flex items-center gap-2 pt-2">
                    <input type="checkbox" name="weekend" checked={formData.weekend} onChange={handleChange}
                        className="w-4 h-4 rounded border-gray-600 text-blue-600 focus:ring-blue-500 bg-gray-900"
                    />
                    <label className="text-sm text-gray-300">Is Weekend?</label>
                </div>
            </div>

            <button disabled={false} // Placeholder for loading
                type="submit"
                className="w-full mt-6 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 text-white font-medium py-3 rounded-lg shadow-lg shadow-blue-500/20 transition-all transform active:scale-95 flex justify-center items-center gap-2"
            >
                <span>Analyze Purchase Intent</span>
            </button>
        </form>
    );
}
