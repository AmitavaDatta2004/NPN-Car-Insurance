"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { assessClaim, getAssessmentStatus } from "@/lib/api";
import { GenAIGateSummary } from "@/types/claim";
import Badge from "@/components/ui/Badge";
import Button from "@/components/ui/Button";
import Card from "@/components/ui/Card";

export default function ClaimProcessingPage() {
  const params = useParams();
  const router = useRouter();
  const claimId = (params?.id as string) || "";

  const [status, setStatus] = useState<string>("ASSESSING_FRAUD");
  const [activeStep, setActiveStep] = useState(1);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [notVehicleGate, setNotVehicleGate] = useState<GenAIGateSummary | null>(null);

  const stages = [
    { title: "1. Image Quality Verification", desc: "Checking image clarity, brightness, and resolution" },
    { title: "2. Vehicle Intake Verification", desc: "Confirming vehicle presence in the uploaded photograph" },
    { title: "3. Fraud Risk Screening", desc: "Checking image authenticity and tamper detection" },
    { title: "4. Damage Severity Assessment", desc: "Analyzing severity of collision impact" },
    { title: "5. Damaged Part Identification", desc: "Identifying affected vehicle components" },
    { title: "6. Damage Localization", desc: "Mapping damage regions on vehicle" },
    { title: "7. Repair Cost Estimation", desc: "Calculating estimated repair and replacement costs in INR (₹)" },
  ];

  useEffect(() => {
    if (!claimId) return;

    let intervalId: NodeJS.Timeout;
    let stepTimer: NodeJS.Timeout;
    let isCancelled = false;

    // Smooth visual progression across stages while assessment runs
    stepTimer = setInterval(() => {
      setActiveStep((prev) => (prev < 6 ? prev + 1 : prev));
    }, 1200);

    const runAssessment = async () => {
      try {
        // Trigger assessment
        const assessment = await assessClaim(claimId);

        // Immediate Halt: If not a vehicle, stop right here!
        if (assessment.genai_gate && !assessment.genai_gate.is_vehicle) {
          clearInterval(stepTimer);
          setNotVehicleGate(assessment.genai_gate);
          return;
        }

        // Poll status every 1.5s
        intervalId = setInterval(async () => {
          if (isCancelled) return;
          try {
            const data = await getAssessmentStatus(claimId);
            setStatus(data.status);

            if (data.is_complete) {
              clearInterval(stepTimer);
              setActiveStep(7);
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
        clearInterval(stepTimer);
        setErrorMsg(
          err instanceof Error ? err.message : "Assessment failed. Please try again."
        );
      }
    };

    runAssessment();

    return () => {
      isCancelled = true;
      if (intervalId) clearInterval(intervalId);
      if (stepTimer) clearInterval(stepTimer);
    };
  }, [claimId, router]);

  if (notVehicleGate) {
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
          Claim <span className="font-mono font-semibold">{claimId ? `${claimId.slice(0, 8)}...` : "Loading..."}</span> is
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
