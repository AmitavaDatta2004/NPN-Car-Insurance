"use client";

import { useState } from "react";
import { Claim, ReviewRecord } from "@/types/claim";
import { saveReviewCorrection, submitReviewDecision, formatDate } from "@/lib/api";
import Button from "@/components/ui/Button";

interface CorrectionFormProps {
  claim: Claim;
  onUpdated: (updatedClaim: Claim) => void;
}

const VEHICLE_PARTS = [
  { id: "front_bumper", label: "Front Bumper" },
  { id: "rear_bumper", label: "Rear Bumper" },
  { id: "hood", label: "Hood" },
  { id: "door", label: "Door" },
  { id: "headlamp", label: "Headlamp" },
];

export default function CorrectionForm({ claim, onUpdated }: CorrectionFormProps) {
  const existingReview: ReviewRecord | null = claim.review;

  // Initialize overrides from existing review or AI prediction
  const initialSeverity =
    (existingReview?.overrides?.severity as string) ||
    claim.assessment?.severity?.predicted_class ||
    "moderate";

  const initialParts: string[] =
    (existingReview?.overrides?.parts as string[]) ||
    (claim.assessment?.location?.predicted_part
      ? [claim.assessment.location.predicted_part]
      : []);

  const [reviewerId, setReviewerId] = useState(
    existingReview?.reviewer_id || "adjuster_1"
  );
  const [severity, setSeverity] = useState<string>(initialSeverity);
  const [selectedParts, setSelectedParts] = useState<string[]>(initialParts);
  const [notes, setNotes] = useState<string>(existingReview?.notes || "");
  const [isSaving, setIsSaving] = useState(false);
  const [isDeciding, setIsDeciding] = useState<"APPROVED" | "REJECTED" | null>(null);
  const [feedback, setFeedback] = useState<{ type: "success" | "error"; message: string } | null>(
    null
  );

  const togglePart = (partId: string) => {
    setSelectedParts((prev) =>
      prev.includes(partId) ? prev.filter((p) => p !== partId) : [...prev, partId]
    );
  };

  const handleSaveCorrection = async () => {
    setIsSaving(true);
    setFeedback(null);
    try {
      const overrides: Record<string, unknown> = {
        severity,
        parts: selectedParts,
      };

      const updated = await saveReviewCorrection(claim.id, {
        reviewer_id: reviewerId,
        notes,
        overrides,
      });

      onUpdated(updated);
      setFeedback({
        type: "success",
        message: "Adjuster notes and overrides saved successfully.",
      });
    } catch (err: unknown) {
      setFeedback({
        type: "error",
        message: err instanceof Error ? err.message : "Failed to save review correction.",
      });
    } finally {
      setIsSaving(false);
    }
  };

  const handleDecision = async (decision: "APPROVED" | "REJECTED") => {
    if (!notes.trim() && decision === "REJECTED") {
      setFeedback({
        type: "error",
        message: "Please provide a reason in the notes field before rejecting the claim.",
      });
      return;
    }

    setIsDeciding(decision);
    setFeedback(null);
    try {
      const overrides: Record<string, unknown> = {
        severity,
        parts: selectedParts,
      };

      const updated = await submitReviewDecision(claim.id, {
        reviewer_id: reviewerId,
        decision,
        notes,
        overrides,
      });

      onUpdated(updated);
      setFeedback({
        type: "success",
        message: `Claim successfully marked as ${decision}.`,
      });
    } catch (err: unknown) {
      setFeedback({
        type: "error",
        message: err instanceof Error ? err.message : `Failed to submit ${decision} decision.`,
      });
    } finally {
      setIsDeciding(null);
    }
  };

  const isCompleted = claim.status === "COMPLETED";

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-slate-100 pb-4">
        <div>
          <h3 className="text-base font-bold text-slate-900">
            Adjuster Review & Manual Override Workspace
          </h3>
          <p className="text-xs text-slate-500">
            Override AI findings where necessary. Corrections do not overwrite the original model outputs.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-500 font-medium">Reviewer ID:</label>
          <input
            type="text"
            value={reviewerId}
            onChange={(e) => setReviewerId(e.target.value)}
            disabled={isCompleted}
            className="w-32 rounded border border-slate-200 bg-slate-50 px-2 py-1 text-xs font-mono text-slate-800 disabled:opacity-60"
          />
        </div>
      </div>

      {existingReview && existingReview.decision !== "PENDING" && (
        <div
          className={`rounded-lg p-4 text-xs ${
            existingReview.decision === "APPROVED"
              ? "bg-emerald-50 border border-emerald-200 text-emerald-900"
              : "bg-rose-50 border border-rose-200 text-rose-900"
          }`}
        >
          <div className="flex items-center justify-between font-semibold">
            <span>
              Final Decision: {existingReview.decision} by {existingReview.reviewer_id}
            </span>
            <span className="font-normal text-[11px]">
              {formatDate(existingReview.timestamp)}
            </span>
          </div>
          {existingReview.notes && (
            <p className="mt-1 text-slate-700 italic font-sans">
              &ldquo;{existingReview.notes}&rdquo;
            </p>
          )}
        </div>
      )}

      {feedback && (
        <div
          className={`rounded-lg p-3 text-xs ${
            feedback.type === "success"
              ? "bg-emerald-50 text-emerald-800 border border-emerald-200"
              : "bg-rose-50 text-rose-800 border border-rose-200"
          }`}
        >
          {feedback.message}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Severity Override */}
        <div className="space-y-3">
          <label className="block text-xs font-semibold uppercase tracking-wider text-slate-700">
            Severity Tier Override
          </label>
          <p className="text-[11px] text-slate-500">
            Original AI prediction:{" "}
            <span className="font-semibold capitalize text-indigo-700">
              {claim.assessment?.severity?.predicted_class || "N/A"}
            </span>
          </p>
          <div className="grid grid-cols-3 gap-2">
            {(["minor", "moderate", "severe"] as const).map((tier) => (
              <button
                key={tier}
                type="button"
                disabled={isCompleted}
                onClick={() => setSeverity(tier)}
                className={`rounded-lg border p-2.5 text-center text-xs font-semibold capitalize transition-all ${
                  severity === tier
                    ? "border-indigo-600 bg-indigo-50 text-indigo-700 shadow-sm"
                    : "border-slate-200 bg-white text-slate-600 hover:border-slate-300"
                } disabled:opacity-50`}
              >
                {tier}
              </button>
            ))}
          </div>
        </div>

        {/* Damaged Parts Override */}
        <div className="space-y-3">
          <label className="block text-xs font-semibold uppercase tracking-wider text-slate-700">
            Damaged Parts Selection
          </label>
          <p className="text-[11px] text-slate-500">
            Dominant part detected:{" "}
            <span className="font-semibold capitalize text-purple-700">
              {claim.assessment?.location?.predicted_part?.replace(/_/g, " ") || "N/A"}
            </span>
          </p>
          <div className="flex flex-wrap gap-2">
            {VEHICLE_PARTS.map((part) => {
              const isSelected = selectedParts.includes(part.id);
              return (
                <button
                  key={part.id}
                  type="button"
                  disabled={isCompleted}
                  onClick={() => togglePart(part.id)}
                  className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-all ${
                    isSelected
                      ? "border-purple-600 bg-purple-50 text-purple-700 shadow-sm"
                      : "border-slate-200 bg-white text-slate-600 hover:border-slate-300"
                  } disabled:opacity-50`}
                >
                  {isSelected ? "✓ " : "+ "}
                  {part.label}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Adjuster Notes */}
      <div className="space-y-2">
        <label className="block text-xs font-semibold uppercase tracking-wider text-slate-700">
          Adjuster Assessment Notes & Justification
        </label>
        <textarea
          rows={3}
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          disabled={isCompleted}
          placeholder="Record notes on evidence clarity, SIU fraud verification, or repair cost override reasons..."
          className="w-full rounded-lg border border-slate-300 p-3 text-xs text-slate-900 placeholder-slate-400 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:opacity-60"
        />
      </div>

      {/* Action Buttons */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-100 pt-4">
        <Button
          variant="outline"
          size="sm"
          onClick={handleSaveCorrection}
          isLoading={isSaving}
          disabled={isCompleted}
        >
          Save Draft Notes & Overrides
        </Button>

        <div className="flex items-center gap-2">
          <Button
            variant="danger"
            size="sm"
            onClick={() => handleDecision("REJECTED")}
            isLoading={isDeciding === "REJECTED"}
            disabled={isCompleted}
          >
            Reject Claim
          </Button>
          <Button
            variant="primary"
            size="sm"
            onClick={() => handleDecision("APPROVED")}
            isLoading={isDeciding === "APPROVED"}
            disabled={isCompleted}
          >
            Approve & Authorize Settlement
          </Button>
        </div>
      </div>
    </div>
  );
}
