"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { AssessmentResult, Claim } from "@/types/claim";
import { formatDate, getClaim, getClaimAssessment, getStatusBadge } from "@/lib/api";
import AssessmentCard from "@/components/AssessmentCard";
import DamageOverlayViewer from "@/components/DamageOverlayViewer";
import Badge from "@/components/ui/Badge";
import Button from "@/components/ui/Button";
import Card from "@/components/ui/Card";

export default function ClaimResultPage() {
  const params = useParams();
  const claimId = params.id as string;

  const [claim, setClaim] = useState<Claim | null>(null);
  const [assessment, setAssessment] = useState<AssessmentResult | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      setErrorMsg(null);
      try {
        const c = await getClaim(claimId);
        setClaim(c);
        if (c.assessment) {
          setAssessment(c.assessment);
        } else {
          const a = await getClaimAssessment(claimId);
          setAssessment(a);
        }
      } catch (err: unknown) {
        setErrorMsg(
          err instanceof Error ? err.message : "Failed to load assessment results."
        );
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [claimId]);

  if (isLoading) {
    return (
      <main className="mx-auto max-w-7xl px-4 py-16 text-center">
        <div className="flex flex-col items-center gap-3">
          <div className="h-10 w-10 animate-spin rounded-full border-4 border-indigo-600 border-t-transparent" />
          <p className="text-sm text-slate-500">Retrieving assessment report...</p>
        </div>
      </main>
    );
  }

  if (errorMsg || !claim || !assessment) {
    return (
      <main className="mx-auto max-w-3xl px-4 py-16">
        <Card className="text-center py-8">
          <h2 className="text-lg font-bold text-slate-900">Assessment Not Found</h2>
          <p className="mt-1 text-xs text-slate-500 max-w-md mx-auto">
            {errorMsg || "This claim has not been assessed yet. Please trigger assessment first."}
          </p>
          <div className="mt-6 flex justify-center gap-3">
            <Link href={`/claims/${claimId}/processing`}>
              <Button size="sm">Run Assessment</Button>
            </Link>
            <Link href="/claims">
              <Button variant="secondary" size="sm">Back to Claims</Button>
            </Link>
          </div>
        </Card>
      </main>
    );
  }

  const isNotAVehicle =
    assessment.genai_gate?.is_vehicle === false ||
    assessment.reason_codes?.includes("not_a_vehicle");

  if (isNotAVehicle) {
    const detectedObj = assessment.genai_gate?.detected_object || "document/non-vehicle";
    const reasoning =
      assessment.genai_gate?.reasoning ||
      "The uploaded evidence does not contain a motor vehicle. Assessment has been halted.";
    const modelName = assessment.genai_gate?.model_name || "GenAI Intake Gate";

    return (
      <main className="mx-auto max-w-xl px-4 py-16 text-center">
        <Card className="border-rose-200 bg-rose-50/60 py-12 px-6 shadow-sm">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-rose-100 text-rose-600 text-3xl font-bold">
            ✕
          </div>

          <h1 className="text-3xl font-black text-rose-950">
            Not a car
          </h1>

          <p className="mt-3 text-sm text-slate-600">
            Please upload a photo of a car.
          </p>

          <div className="mt-8 flex justify-center gap-3">
            <Link href="/claims/new">
              <Button size="md" variant="primary">
                Upload Car Photo
              </Button>
            </Link>
            <Link href="/claims">
              <Button size="md" variant="outline">
                Back
              </Button>
            </Link>
          </div>
        </Card>
      </main>
    );
  }

  const statusBadge = getStatusBadge(claim.status);
  const primaryImagePath = claim.images[0]?.local_path || assessment.image_path || "";

  const isFraud =
    assessment.fraud?.flag === 1 ||
    assessment.route === "FRAUD_REVIEW" ||
    (assessment.fraud && assessment.fraud.probability > 0.50);

  return (
    <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-8">
      {/* Top Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold text-slate-900">
              Claim Assessment Report
            </h1>
            <Badge color={statusBadge.color} bg={statusBadge.bg}>
              {statusBadge.label}
            </Badge>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Policy: <span className="font-mono font-semibold">{claim.policy_number}</span> •
            Vehicle: <span className="font-semibold">{claim.vehicle_year} {claim.vehicle_make} {claim.vehicle_model}</span> •
            Filed: {formatDate(claim.created_at)}
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Link href={`/claims/${claim.id}/timeline`}>
            <Button variant="outline" size="sm">
              Audit Timeline ⏱️
            </Button>
          </Link>
          <Link href={`/reviewer/claims/${claim.id}`}>
            <Button variant="secondary" size="sm">
              Adjuster Workspace 👤
            </Button>
          </Link>
          <Link href="/claims">
            <Button variant="outline" size="sm">
              ← Claims List
            </Button>
          </Link>
        </div>
      </div>

      {/* Visual Damage Overlay Viewer - Only shown if genuine (NOT fraud) */}
      {!isFraud && primaryImagePath && (
        <Card>
          <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-3">
            Damage Detection
          </h2>
          <DamageOverlayViewer
            imagePath={primaryImagePath}
            detections={assessment.detections || []}
          />
        </Card>
      )}

      {/* Assessment Metrics Card */}
      <AssessmentCard assessment={assessment} />
    </main>
  );
}
