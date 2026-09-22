"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { Claim } from "@/types/claim";
import { getClaim, getImageUrl, getRouteInfo, getStatusBadge, formatDate } from "@/lib/api";
import DamageOverlayViewer from "@/components/DamageOverlayViewer";
import ModelInsightsPanel from "@/components/reviewer/ModelInsightsPanel";
import CostTable from "@/components/reviewer/CostTable";
import CorrectionForm from "@/components/reviewer/CorrectionForm";
import Spinner from "@/components/ui/Spinner";
import Alert from "@/components/ui/Alert";
import Button from "@/components/ui/Button";

export default function ReviewerClaimWorkspacePage() {
  const params = useParams<{ id: string }>();
  const claimId = params.id;

  const [claim, setClaim] = useState<Claim | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchClaimData = useCallback(async () => {
    if (!claimId) return;
    setIsLoading(true);
    setError(null);
    try {
      const data = await getClaim(claimId);
      setClaim(data);
    } catch (err: unknown) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to load claim from backend."
      );
    } finally {
      setIsLoading(false);
    }
  }, [claimId]);

  useEffect(() => {
    fetchClaimData();
  }, [fetchClaimData]);

  if (isLoading) {
    return (
      <div className="flex h-96 items-center justify-center">
        <Spinner size="lg" />
      </div>
    );
  }

  if (error || !claim) {
    return (
      <div className="mx-auto max-w-4xl px-4 py-12 space-y-4">
        <Alert variant="danger" title="Error Loading Claim">
          {error || "Claim not found."}
        </Alert>
        <Link href="/reviewer/queue">
          <Button variant="outline" size="sm">
            ← Back to Review Queue
          </Button>
        </Link>
      </div>
    );
  }

  const assessment = claim.assessment;
  const primaryImage = claim.images.length > 0 ? claim.images[0] : null;
  const imagePathToDisplay = primaryImage?.local_path || assessment?.image_path || "";
  const imageUrl = imagePathToDisplay ? getImageUrl(imagePathToDisplay) : "";
  const routeInfo = assessment ? getRouteInfo(assessment.route) : null;
  const statusInfo = getStatusBadge(claim.status);

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-6">
      {/* Top Navigation & Breadcrumbs */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-b border-slate-200 pb-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-medium text-slate-500 mb-1">
            <Link href="/reviewer/queue" className="hover:text-indigo-600">
              Review Queue
            </Link>
            <span>/</span>
            <span className="font-mono text-slate-700">{claim.policy_number}</span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 flex items-center gap-3">
            <span>Claim #{claim.policy_number}</span>
            {routeInfo && (
              <span
                className={`inline-flex items-center rounded-full px-3 py-0.5 text-xs font-semibold ${routeInfo.bg} ${routeInfo.color} border ${routeInfo.border}`}
              >
                {routeInfo.label}
              </span>
            )}
            <span
              className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${statusInfo.bg} ${statusInfo.color}`}
            >
              {statusInfo.label}
            </span>
          </h1>
          <p className="mt-1 text-xs text-slate-500">
            {claim.vehicle_year} {claim.vehicle_make} {claim.vehicle_model} ({claim.vehicle_segment} segment) • Filed on {formatDate(claim.created_at)}
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Link href={`/claims/${claim.id}/timeline`}>
            <Button variant="outline" size="sm">
              🕒 Customer Audit Timeline
            </Button>
          </Link>
          <Link href="/reviewer/queue">
            <Button variant="secondary" size="sm">
              ← Back to Queue
            </Button>
          </Link>
        </div>
      </div>

      {/* Main Review Workspace Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Visual Evidence (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-4">
            <h3 className="text-sm font-semibold text-slate-900">
              Photographic Evidence & YOLO Detections
            </h3>

            {imagePathToDisplay ? (
              <DamageOverlayViewer
                imagePath={imagePathToDisplay}
                detections={assessment?.detections || []}
              />
            ) : (
              <div className="flex h-64 items-center justify-center rounded-lg bg-slate-100 text-xs text-slate-400">
                No photographic evidence uploaded for this claim.
              </div>
            )}
          </div>

          {/* Incident Details Card */}
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-3">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Claim & Incident Details
            </h3>
            <div className="space-y-2 text-xs">
              <div>
                <span className="text-slate-500">Vehicle:</span>{" "}
                <span className="font-medium text-slate-800">
                  {claim.vehicle_year} {claim.vehicle_make} {claim.vehicle_model}
                </span>
              </div>
              <div>
                <span className="text-slate-500">Vehicle Segment:</span>{" "}
                <span className="font-medium capitalize text-slate-800">
                  {claim.vehicle_segment}
                </span>
              </div>
              <div>
                <span className="text-slate-500">Incident Date:</span>{" "}
                <span className="font-medium text-slate-800">
                  {formatDate(claim.incident_date || claim.created_at)}
                </span>
              </div>
              <div>
                <span className="text-slate-500 block mb-1">Description:</span>
                <p className="rounded-lg bg-slate-50 p-2.5 text-slate-700 leading-relaxed border border-slate-100">
                  {claim.incident_description || "No description provided."}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: AI Signals & Costing (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          {/* AI Model Insights */}
          <ModelInsightsPanel assessment={assessment} />

          {/* Itemized Cost Breakdown */}
          <CostTable cost={assessment?.cost || null} />
        </div>
      </div>

      {/* Full-Width Adjuster Manual Override Form */}
      <div className="pt-2">
        <CorrectionForm
          claim={claim}
          onUpdated={(updated) => setClaim(updated)}
        />
      </div>
    </div>
  );
}
