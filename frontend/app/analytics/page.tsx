'use client';

import { useEffect, useState } from 'react';

// Types
interface SummaryStats {
    total_sessions: number;
    total_interventions: number;
    total_claims: number;
    total_conversions: number;
    conversion_rate: number;
    claim_rate: number;
}

interface TimeSeriesPoint {
    timestamp: string;
    interventions: number;
    claims: number;
}

interface PageStats {
    page_url: string;
    page_title: string;
    views: number;
    abandonment_rate: number;
}

// API Base URL
const API_BASE = 'http://localhost:8001';

// KPI Card Component
function KPICard({ title, value, subtitle, color }: { title: string; value: string | number; subtitle?: string; color: string }) {
    return (
        <div className={`bg-gradient-to-br ${color} rounded-2xl p-6 shadow-lg text-white`}>
            <h3 className="text-sm font-medium opacity-80 uppercase tracking-wide">{title}</h3>
            <p className="text-4xl font-bold mt-2">{value}</p>
            {subtitle && <p className="text-sm opacity-70 mt-1">{subtitle}</p>}
        </div>
    );
}

// Simple Bar Component
function BarChart({ data }: { data: PageStats[] }) {
    const maxViews = Math.max(...data.map(d => d.views), 1);

    return (
        <div className="space-y-3">
            {data.map((page, i) => (
                <div key={i} className="flex items-center gap-4">
                    <div className="w-32 text-sm text-gray-600 truncate" title={page.page_title}>
                        {page.page_title || 'Unknown'}
                    </div>
                    <div className="flex-1 h-8 bg-gray-200 rounded-full overflow-hidden">
                        <div
                            className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-end pr-2"
                            style={{ width: `${(page.views / maxViews) * 100}%` }}
                        >
                            <span className="text-xs text-white font-medium">{page.views}</span>
                        </div>
                    </div>
                    <div className="w-16 text-sm text-gray-500 text-right">
                        {page.abandonment_rate}%
                    </div>
                </div>
            ))}
        </div>
    );
}

export default function AnalyticsPage() {
    const [summary, setSummary] = useState<SummaryStats | null>(null);
    const [timeSeries, setTimeSeries] = useState<TimeSeriesPoint[]>([]);
    const [pages, setPages] = useState<PageStats[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        async function fetchData() {
            try {
                setLoading(true);

                // Fetch all data in parallel
                const [summaryRes, timeSeriesRes, pagesRes] = await Promise.all([
                    fetch(`${API_BASE}/analytics/summary`),
                    fetch(`${API_BASE}/analytics/interventions?hours=24`),
                    fetch(`${API_BASE}/analytics/pages?limit=10`)
                ]);

                if (!summaryRes.ok || !timeSeriesRes.ok || !pagesRes.ok) {
                    throw new Error('Failed to fetch analytics data');
                }

                const [summaryData, timeSeriesData, pagesData] = await Promise.all([
                    summaryRes.json(),
                    timeSeriesRes.json(),
                    pagesRes.json()
                ]);

                setSummary(summaryData);
                setTimeSeries(timeSeriesData);
                setPages(pagesData);
                setError(null);
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Unknown error');
            } finally {
                setLoading(false);
            }
        }

        fetchData();

        // Auto-refresh every 30 seconds
        const interval = setInterval(fetchData, 30000);
        return () => clearInterval(interval);
    }, []);

    if (loading && !summary) {
        return (
            <div className="min-h-screen bg-gray-50 flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
                    <p className="mt-4 text-gray-600">Loading analytics...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="min-h-screen bg-gray-50 flex items-center justify-center">
                <div className="text-center text-red-600">
                    <p className="text-xl font-semibold">Error loading analytics</p>
                    <p className="mt-2">{error}</p>
                    <p className="mt-4 text-sm text-gray-500">Make sure the Tracker API is running on port 8001</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gray-50 py-8 px-4">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-gray-900">📊 AI Analytics Dashboard</h1>
                    <p className="text-gray-600 mt-1">Real-time insights into AI intervention performance</p>
                </div>

                {/* KPI Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                    <KPICard
                        title="Total Sessions"
                        value={summary?.total_sessions.toLocaleString() || 0}
                        subtitle="All-time visitors"
                        color="from-blue-500 to-blue-600"
                    />
                    <KPICard
                        title="Interventions Shown"
                        value={summary?.total_interventions.toLocaleString() || 0}
                        subtitle="AI popups triggered"
                        color="from-purple-500 to-purple-600"
                    />
                    <KPICard
                        title="Discounts Claimed"
                        value={summary?.total_claims.toLocaleString() || 0}
                        subtitle={`${summary?.claim_rate || 0}% claim rate`}
                        color="from-green-500 to-green-600"
                    />
                    <KPICard
                        title="Conversions"
                        value={summary?.total_conversions.toLocaleString() || 0}
                        subtitle={`${summary?.conversion_rate || 0}% conversion rate`}
                        color="from-orange-500 to-orange-600"
                    />
                </div>

                {/* Charts Section */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Time Series */}
                    <div className="bg-white rounded-2xl shadow-lg p-6">
                        <h2 className="text-xl font-semibold text-gray-800 mb-4">📈 Interventions (Last 24h)</h2>
                        {timeSeries.length > 0 ? (
                            <div className="space-y-2">
                                {timeSeries.slice(-8).map((point, i) => (
                                    <div key={i} className="flex items-center gap-2 text-sm">
                                        <span className="w-28 text-gray-500">{point.timestamp.split(' ')[1] || point.timestamp}</span>
                                        <div className="flex-1 h-6 bg-gray-100 rounded-full overflow-hidden flex">
                                            <div
                                                className="h-full bg-purple-500"
                                                style={{ width: `${Math.min((point.interventions / (Math.max(...timeSeries.map(t => t.interventions)) || 1)) * 100, 100)}%` }}
                                            />
                                        </div>
                                        <span className="w-8 text-right text-purple-600 font-medium">{point.interventions}</span>
                                        <span className="w-8 text-right text-green-600 font-medium">{point.claims}</span>
                                    </div>
                                ))}
                                <div className="flex justify-end gap-4 text-xs text-gray-500 mt-2">
                                    <span className="flex items-center gap-1"><span className="w-3 h-3 bg-purple-500 rounded"></span> Interventions</span>
                                    <span className="flex items-center gap-1"><span className="w-3 h-3 bg-green-500 rounded"></span> Claims</span>
                                </div>
                            </div>
                        ) : (
                            <p className="text-gray-500 text-center py-8">No intervention data yet</p>
                        )}
                    </div>

                    {/* Top Pages */}
                    <div className="bg-white rounded-2xl shadow-lg p-6">
                        <h2 className="text-xl font-semibold text-gray-800 mb-4">🔥 Top Pages by Views</h2>
                        {pages.length > 0 ? (
                            <BarChart data={pages} />
                        ) : (
                            <p className="text-gray-500 text-center py-8">No page view data yet</p>
                        )}
                        <p className="text-xs text-gray-400 mt-4 text-right">% = Exit Rate</p>
                    </div>
                </div>

                {/* Footer */}
                <div className="mt-8 text-center text-sm text-gray-500">
                    <p>🔄 Auto-refreshes every 30 seconds | Last updated: {new Date().toLocaleTimeString()}</p>
                </div>
            </div>
        </div>
    );
}
