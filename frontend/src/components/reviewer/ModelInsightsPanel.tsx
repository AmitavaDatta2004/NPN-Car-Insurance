"use client";

import { AssessmentResult } from "@/types/claim";
import Badge from "@/components/ui/Badge";

interface ModelInsightsPanelProps {
  assessment: AssessmentResult | null;
}

export default function ModelInsightsPanel({ assessment }: ModelInsightsPanelProps) {
  if (!assessment) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-6 text-center text-slate-400">
        <p className="text-sm">No assessment data available for this claim.</p>
      </div>
    );
  }

  const { fraud, severity, location, quality, model_versions, reason_codes } = assessment;

  return (
    <div className="space-y-4">
      {/* Reason Codes & Decision Summary */}
      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">
          AI Routing & Decision Basis
        </h3>
        <div className="flex flex-wrap gap-1.5">
          {reason_codes.map((code) => (
            <span
              key={code}
              className="inline-flex items-center rounded-md bg-slate-100 px-2.5 py-1 text-xs font-mono text-slate-700 border border-slate-200"
            >
              {code}
            </span>
          ))}
        </div>
      </div>

      {/* Grid of Model Signals */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Fraud Risk Signal */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-slate-900 flex items-center gap-1.5">
              <span>🛡️ Fraud & Authenticity</span>
              {model_versions?.fraud && (
                <span className="font-mono text-[10px] text-slate-400 font-normal">
                  ({model_versions.fraud})
                </span>
              )}
            </h4>
            {fraud && (
              <Badge
                variant={
                  fraud.risk_level === "low"
                    ? "success"
                    : fraud.risk_level === "medium"
                    ? "warning"
                    : "danger"
                }
              >
                {fraud.risk_level.toUpperCase()} RISK
              </Badge>
            )}
          </div>

          {fraud ? (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-600">Suspicious Probability</span>
                <span className="font-semibold text-slate-900">
                  {(fraud.probability * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
                <div
                  className={`h-full rounded-full transition-all ${
                    fraud.probability >= 0.65
                      ? "bg-rose-500"
                      : fraud.probability >= 0.35
                      ? "bg-amber-500"
                      : "bg-emerald-500"
                  }`}
                  style={{ width: `${Math.min(100, Math.max(5, fraud.probability * 100))}%` }}
                />
              </div>
              {fraud.warnings && fraud.warnings.length > 0 && (
                <div className="mt-2 space-y-1">
                  {fraud.warnings.map((w, idx) => (
                    <p key={idx} className="text-[11px] text-rose-600">
                      ⚠️ {w}
                    </p>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <p className="text-xs text-slate-400">Fraud check skipped or not executed.</p>
          )}
        </div>

        {/* Severity Classification */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-slate-900 flex items-center gap-1.5">
              <span>💥 Damage Severity</span>
              {model_versions?.severity && (
                <span className="font-mono text-[10px] text-slate-400 font-normal">
                  ({model_versions.severity})
                </span>
              )}
            </h4>
            {severity && (
              <Badge
                variant={
                  severity.predicted_class === "minor"
                    ? "success"
                    : severity.predicted_class === "moderate"
                    ? "warning"
                    : "danger"
                }
              >
                {severity.predicted_class.toUpperCase()}
              </Badge>
            )}
          </div>

          {severity ? (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-600">Model Confidence</span>
                <span className="font-semibold text-slate-900">
                  {(severity.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="space-y-1.5 pt-1">
                {Object.entries(severity.probabilities || {}).map(([cls, prob]) => (
                  <div key={cls} className="space-y-0.5">
                    <div className="flex justify-between text-[11px]">
                      <span className="capitalize text-slate-500">{cls}</span>
                      <span className="font-medium text-slate-700">
                        {(prob * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
                      <div
                        className="h-full rounded-full bg-indigo-500"
                        style={{ width: `${prob * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-400">Severity assessment not available.</p>
          )}
        </div>

        {/* Location CNN Classification */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-slate-900 flex items-center gap-1.5">
              <span>📍 Damaged Part Location</span>
              {location?.model_version && (
                <span className="font-mono text-[10px] text-slate-400 font-normal">
                  ({location.model_version})
                </span>
              )}
            </h4>
            {location && (
              <span className="inline-flex items-center rounded-md bg-indigo-50 px-2 py-0.5 text-xs font-semibold text-indigo-700 border border-indigo-200">
                {location.predicted_part.replace(/_/g, " ").toUpperCase()}
              </span>
            )}
          </div>

          {location ? (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-600">Dominant Part Confidence</span>
                <span className="font-semibold text-slate-900">
                  {(location.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="space-y-1.5 pt-1">
                {location.top3 && location.top3.length > 0 ? (
                  location.top3.map(([part, prob]) => (
                    <div key={part} className="space-y-0.5">
                      <div className="flex justify-between text-[11px]">
                        <span className="capitalize text-slate-500">
                          {part.replace(/_/g, " ")}
                        </span>
                        <span className="font-medium text-slate-700">
                          {(prob * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
                        <div
                          className="h-full rounded-full bg-purple-500"
                          style={{ width: `${prob * 100}%` }}
                        />
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-[11px] text-slate-400">No top probabilities recorded.</p>
                )}
              </div>
              {location.warning && (
                <p className="text-[11px] text-amber-600">⚠️ {location.warning}</p>
              )}
            </div>
          ) : (
            <p className="text-xs text-slate-400">Location CNN classification not available.</p>
          )}
        </div>

        {/* Evidence Quality (OpenCV) */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-slate-900">
              <span>📷 OpenCV Evidence Quality</span>
            </h4>
            {quality && (
              <Badge variant={quality.acceptable ? "success" : "danger"}>
                {quality.acceptable ? "PASSED" : "FAILED"}
              </Badge>
            )}
          </div>

          {quality ? (
            <div className="grid grid-cols-3 gap-2 pt-1 text-center">
              <div className="rounded-lg bg-slate-50 p-2 border border-slate-100">
                <div className="text-[10px] text-slate-500">Blur Var</div>
                <div className="text-xs font-semibold text-slate-800">
                  {quality.blur_score.toFixed(1)}
                </div>
              </div>
              <div className="rounded-lg bg-slate-50 p-2 border border-slate-100">
                <div className="text-[10px] text-slate-500">Brightness</div>
                <div className="text-xs font-semibold text-slate-800">
                  {quality.brightness.toFixed(0)}
                </div>
              </div>
              <div className="rounded-lg bg-slate-50 p-2 border border-slate-100">
                <div className="text-[10px] text-slate-500">Contrast</div>
                <div className="text-xs font-semibold text-slate-800">
                  {quality.contrast.toFixed(1)}
                </div>
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-400">Quality checks not available.</p>
          )}

          {quality && quality.warnings && quality.warnings.length > 0 && (
            <div className="space-y-1">
              {quality.warnings.map((w, idx) => (
                <p key={idx} className="text-[11px] text-orange-600">
                  ⚠️ {w}
                </p>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
