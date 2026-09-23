"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { DashboardSummary } from "@/types/claim";
import { getDashboardSummary, formatINR } from "@/lib/api";
import RouteDistributionChart from "@/components/charts/RouteDistributionChart";
import SeverityBarChart from "@/components/charts/SeverityBarChart";
import LocationPartChart from "@/components/charts/LocationPartChart";
import Spinner from "@/components/ui/Spinner";
import Alert from "@/components/ui/Alert";
import Button from "@/components/ui/Button";

export default function ReviewerDashboardPage() {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSummary = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getDashboardSummary();
      setSummary(data);
    } catch (err: unknown) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to load dashboard summary from backend."
      );
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSummary();
  }, [fetchSummary]);

  if (isLoading) {
    return (
      <div className="flex h-96 items-center justify-center">
        <Spinner size="lg" />
      </div>
    );
  }

  if (error || !summary) {
    return (
      <div className="mx-auto max-w-4xl px-4 py-12 space-y-4">
        <Alert variant="danger" title="Dashboard Connection Error">
          {error || "Unable to retrieve dashboard analytics."} Ensure the FastAPI backend is running on port 8000.
        </Alert>
        <Button variant="outline" size="sm" onClick={fetchSummary}>
          Retry Connection
        </Button>
      </div>
    );
  }

  const fastTrackRate =
    summary.total_claims > 0
      ? ((summary.fast_track_count / summary.total_claims) * 100).toFixed(1)
      : "0.0";

  const flaggedCount = summary.fraud_review_count + summary.damage_review_count;

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-8">
      {/* Page Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="flex items-center gap-2 text-xs font-medium text-slate-500 mb-1">
            <Link href="/" className="hover:text-indigo-600">
              ClaimVision
            </Link>
            <span>/</span>
            <span className="text-slate-900">Analytics</span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">
            Executive Claims Analytics & Triage KPIs
          </h1>
          <p className="mt-1 text-xs text-slate-500">
            Live operational intelligence across fraud screening, damage severity, damaged vehicle parts, and adjuster overrides.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Link href="/reviewer/queue">
            <Button variant="outline" size="sm">
              📋 Go to Review Queue
            </Button>
          </Link>
          <Button variant="secondary" size="sm" onClick={fetchSummary} isLoading={isLoading}>
            🔄 Refresh
          </Button>
        </div>
      </div>

      {/* KPI Summary Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <span className="text-xs font-medium text-slate-500">Total Claims</span>
          <div className="mt-1 text-2xl font-bold text-slate-900">
            {summary.total_claims}
          </div>
          <p className="mt-1 text-[11px] text-slate-400">All filed claims</p>
        </div>

        <div className="rounded-xl border border-emerald-200 bg-emerald-50/50 p-5 shadow-sm">
          <span className="text-xs font-medium text-emerald-800">Straight-Through Rate</span>
          <div className="mt-1 text-2xl font-bold text-emerald-700">
            {fastTrackRate}%
          </div>
          <p className="mt-1 text-[11px] text-emerald-600">
            {summary.fast_track_count} fast-track settlements
          </p>
        </div>

        <div className="rounded-xl border border-rose-200 bg-rose-50/50 p-5 shadow-sm">
          <span className="text-xs font-medium text-rose-800">Flagged for Review</span>
          <div className="mt-1 text-2xl font-bold text-rose-700">
            {flaggedCount}
          </div>
          <p className="mt-1 text-[11px] text-rose-600">
            {summary.fraud_review_count} fraud • {summary.damage_review_count} damage
          </p>
        </div>

        <div className="rounded-xl border border-indigo-200 bg-indigo-50/50 p-5 shadow-sm">
          <span className="text-xs font-medium text-indigo-800">Average Claim Cost</span>
          <div className="mt-1 text-xl font-bold text-indigo-900">
            {formatINR(summary.average_cost.min)} – {formatINR(summary.average_cost.max)}
          </div>
          <p className="mt-1 text-[11px] text-indigo-600">Based on assessed claims</p>
        </div>

        <div className="rounded-xl border border-purple-200 bg-purple-50/50 p-5 shadow-sm">
          <span className="text-xs font-medium text-purple-800">Adjuster Override Rate</span>
          <div className="mt-1 text-2xl font-bold text-purple-700">
            {(summary.override_rate * 100).toFixed(1)}%
          </div>
          <p className="mt-1 text-[11px] text-purple-600">
            Human corrections applied
          </p>
        </div>
      </div>

      {/* Visual Analytics Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Route Distribution */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">
              Triage Route Distribution
            </h3>
            <p className="text-xs text-slate-500">
              Breakdown of automated decision routing
            </p>
          </div>
          <RouteDistributionChart statusCounts={summary.status_counts} />
        </div>

        {/* Severity Breakdown */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">
              Damage Severity Breakdown
            </h3>
            <p className="text-xs text-slate-500">
              Classified by MobileNetV2 / CNN
            </p>
          </div>
          <SeverityBarChart severityDistribution={summary.severity_distribution} />
        </div>

        {/* Damaged Parts Distribution */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">
              Vehicle Part Location
            </h3>
            <p className="text-xs text-slate-500">
              Dominant damaged part (Location CNN)
            </p>
          </div>
          <LocationPartChart locationDistribution={summary.location_distribution} />
        </div>
      </div>

      {/* Operational Intelligence Card */}
      <div className="rounded-xl border border-slate-200 bg-slate-50 p-6">
        <h3 className="text-sm font-bold text-slate-900 mb-2">
          💡 ClaimVision Explainability & Governance Note
        </h3>
        <p className="text-xs text-slate-600 leading-relaxed">
          ClaimVision AI is an explainable decision-support system. High-risk fraud signals and severe damage assessments never trigger automated settlement; they are strictly routed to the Reviewer Queue for SIU and human adjuster confirmation. Adjuster corrections are logged alongside AI findings in an immutable audit timeline, ensuring full traceability and continuous evaluation data.
        </p>
      </div>
    </div>
  );
}
