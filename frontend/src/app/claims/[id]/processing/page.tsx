"use client";

import React, { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { assessClaim, getAssessmentStatus } from "@/lib/api";
import Button from "@/components/ui/Button";
import Card from "@/components/ui/Card";

export default function ClaimProcessingPage() {
  const params = useParams();
  const router = useRouter();
  const claimId = params.id as string;

  const [status, setStatus] = useState<string>("ASSESSING_FRAUD");
  const [activeStep, setActiveStep] = useState(1);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    let intervalId: NodeJS.Timeout;
    let isCancelled = false;

    const runAssessment = async () => {
      try {
        // Trigger assessment
        await assessClaim(claimId);

        // Poll status every 1.5s
        intervalId = setInterval(async () => {
          if (isCancelled) return;
          try {
            const data = await getAssessmentStatus(claimId);
            setStatus(data.status);

            // Animate steps based on status
            if (data.status === "ASSESSING_FRAUD") {
              setActiveStep(2);
            } else if (data.status === "ASSESSING_DAMAGE") {
              setActiveStep(3);
            } else if (data.is_complete) {
              setActiveStep(4);
              clearInterval(intervalId);
              // Brief delay for visual satisfaction before navigating
              setTimeout(() => {
                router.push(`/claims/${claimId}/result`);
              }, 1200);
            }
          } catch {
            // Keep polling
          }
        }, 1500);
      } catch (err: unknown) {
        setErrorMsg(
          err instanceof Error ? err.message : "Assessment failed. Please try again."
        );
      }
    };

    runAssessment();

    return () => {
      isCancelled = true;
      if (intervalId) clearInterval(intervalId);
    };
  }, [claimId, router]);

  const stages = [
    { title: "OpenCV Evidence Integrity", desc: "Checking blur score, resolution, and pHash duplicates" },
    { title: "MobileNetV2 Fraud Risk Scoring", desc: "Evaluating image authenticity and suspicious patterns" },
    { title: "Severity & Part Location Classification", desc: "Predicting damage tier (minor/moderate/severe) & damaged part" },
    { title: "YOLOv8 Localization & Repair Costing", desc: "Generating damage bounding boxes and itemized cost table" },
  ];

  return (
    <main className="mx-auto max-w-2xl px-4 py-16 sm:px-6 lg:px-8">
      <Card className="text-center py-10">
        <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-indigo-50 text-indigo-600">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-indigo-600 border-t-transparent" />
        </div>

        <h1 className="text-2xl font-bold text-slate-900">
          Analyzing Claim Evidence...
        </h1>
        <p className="mt-2 text-xs text-slate-500 max-w-md mx-auto">
          Claim <span className="font-mono font-semibold">{claimId.slice(0, 8)}...</span> is
          passing through our multi-stage AI assessment pipeline.
        </p>

        {errorMsg ? (
          <div className="mt-8 rounded-xl bg-rose-50 p-4 text-xs text-rose-700 border border-rose-200">
            <p className="font-semibold">⚠️ Assessment Interrupted</p>
            <p className="mt-1">{errorMsg}</p>
            <div className="mt-4">
              <Button size="sm" onClick={() => window.location.reload()}>
                Retry Assessment
              </Button>
            </div>
          </div>
        ) : (
          <div className="mt-10 text-left space-y-4 max-w-lg mx-auto">
            {stages.map((stage, idx) => {
              const stepNum = idx + 1;
              const isDone = activeStep > stepNum;
              const isCurrent = activeStep === stepNum;

              return (
                <div
                  key={idx}
                  className={`flex items-start gap-3.5 rounded-xl p-3 transition-all ${
                    isCurrent
                      ? "border border-indigo-200 bg-indigo-50/50 shadow-xs"
                      : isDone
                      ? "border border-slate-100 bg-slate-50/50 opacity-80"
                      : "border border-transparent opacity-40"
                  }`}
                >
                  <div className="mt-0.5">
                    {isDone ? (
                      <span className="flex h-5 w-5 items-center justify-center rounded-full bg-emerald-600 text-white text-xs">
                        ✓
                      </span>
                    ) : isCurrent ? (
                      <div className="h-5 w-5 animate-spin rounded-full border-2 border-indigo-600 border-t-transparent" />
                    ) : (
                      <span className="flex h-5 w-5 items-center justify-center rounded-full border border-slate-300 text-slate-400 text-xs">
                        {stepNum}
                      </span>
                    )}
                  </div>
                  <div>
                    <h3 className={`text-xs font-bold ${isCurrent ? "text-indigo-900" : "text-slate-800"}`}>
                      {stage.title}
                    </h3>
                    <p className="text-[11px] text-slate-500 mt-0.5">{stage.desc}</p>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </Card>
    </main>
  );
}
