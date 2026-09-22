"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Claim } from "@/types/claim";
import { listClaims } from "@/lib/api";
import ReviewQueueTable from "@/components/reviewer/ReviewQueueTable";
import Spinner from "@/components/ui/Spinner";
import Alert from "@/components/ui/Alert";
import Button from "@/components/ui/Button";

export default function ReviewerQueuePage() {
  const [claims, setClaims] = useState<Claim[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchClaims = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await listClaims();
      setClaims(data);
    } catch (err: unknown) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to load claims queue from backend."
      );
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchClaims();
  }, []);

  const fraudAlerts = claims.filter(
    (c) =>
      c.status === "FRAUD_REVIEW" || c.assessment?.route === "FRAUD_REVIEW"
  ).length;

  const damageAlerts = claims.filter(
    (c) =>
      c.status === "MANUAL_DAMAGE_REVIEW" ||
      c.assessment?.route === "MANUAL_DAMAGE_REVIEW"
  ).length;

  const completedReviews = claims.filter(
    (c) => c.review !== null && c.review.decision !== "PENDING"
  ).length;

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-6">
      {/* Page Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="flex items-center gap-2 text-xs font-medium text-slate-500 mb-1">
            <Link href="/" className="hover:text-indigo-600">
              ClaimVision
            </Link>
            <span>/</span>
            <span className="text-slate-900">Reviewer Workspace</span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">
            Claims Triage & Review Queue
          </h1>
          <p className="mt-1 text-xs text-slate-500">
            Human-in-the-loop dashboard for inspecting flagged suspicious images, severe damage, and manual adjustments.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Link href="/reviewer/dashboard">
            <Button variant="outline" size="sm">
              📊 View Analytics Dashboard
            </Button>
          </Link>
        </div>
      </div>

      {/* KPI Highlights */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <span className="text-xs text-slate-500 font-medium">Pending Triage</span>
          <div className="mt-1 text-2xl font-bold text-slate-900">
            {fraudAlerts + damageAlerts}
          </div>
        </div>

        <div className="rounded-xl border border-rose-200 bg-rose-50/50 p-4 shadow-sm">
          <span className="text-xs text-rose-700 font-medium">Fraud Review Alerts</span>
          <div className="mt-1 text-2xl font-bold text-rose-800">{fraudAlerts}</div>
        </div>

        <div className="rounded-xl border border-amber-200 bg-amber-50/50 p-4 shadow-sm">
          <span className="text-xs text-amber-800 font-medium">Damage Overrides</span>
          <div className="mt-1 text-2xl font-bold text-amber-900">{damageAlerts}</div>
        </div>

        <div className="rounded-xl border border-teal-200 bg-teal-50/50 p-4 shadow-sm">
          <span className="text-xs text-teal-800 font-medium">Decisions Recorded</span>
          <div className="mt-1 text-2xl font-bold text-teal-900">{completedReviews}</div>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="danger" title="Backend Connection Error">
          {error}. Ensure the FastAPI server is running on port 8000.
        </Alert>
      )}

      {/* Main Table */}
      {isLoading ? (
        <div className="flex h-64 items-center justify-center rounded-xl border border-slate-200 bg-white shadow-sm">
          <Spinner size="lg" />
        </div>
      ) : (
        <ReviewQueueTable
          claims={claims}
          isLoading={isLoading}
          onRefresh={fetchClaims}
        />
      )}
    </div>
  );
}
