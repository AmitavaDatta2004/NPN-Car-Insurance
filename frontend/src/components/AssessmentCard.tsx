"use client";

import React from "react";
import { AssessmentResult } from "@/types/claim";
import { formatINR, getRouteInfo } from "@/lib/api";
import Badge from "./ui/Badge";
import Card from "./ui/Card";

interface AssessmentCardProps {
  assessment: AssessmentResult;
}

export default function AssessmentCard({ assessment }: AssessmentCardProps) {
  const routeInfo = getRouteInfo(assessment.route);
  const fraudProb = assessment.fraud?.probability ?? 0.0;
  const fraudPercent = (fraudProb * 100).toFixed(1);

  return (
    <div className="space-y-6">
      {/* Route Decision Banner */}
      <div
        className={`rounded-2xl border p-6 transition-all shadow-sm ${routeInfo.bg} ${routeInfo.border}`}
      >
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
              Triage Recommendation
            </span>
            <h2 className={`text-2xl font-black tracking-tight ${routeInfo.color}`}>
              {routeInfo.label}
            </h2>
          </div>
          <div className="flex items-center gap-2">
            <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 shadow-xs border border-slate-200">
              ⚡ {assessment.inference_ms.toFixed(0)} ms
            </span>
          </div>
        </div>
        <p className="mt-2 text-sm text-slate-700 leading-relaxed max-w-3xl">
          {routeInfo.description}
        </p>

        {assessment.reason_codes && assessment.reason_codes.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-2">
            {assessment.reason_codes.map((code) => (
              <span
                key={code}
                className="inline-flex items-center rounded-md bg-white/80 px-2.5 py-1 text-xs font-medium text-slate-800 border border-slate-200/60"
              >
                • {code.replace(/_/g, " ")}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* 4 Pillars Grid: Fraud, Severity, Location, Cost */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {/* 1. Fraud Risk */}
        <Card className="flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase text-slate-500">
                Fraud Risk
              </span>
              <Badge
                color={
                  fraudProb >= 0.65
                    ? "text-rose-800"
                    : fraudProb >= 0.3
                    ? "text-amber-800"
                    : "text-emerald-800"
                }
                bg={
                  fraudProb >= 0.65
                    ? "bg-rose-100"
                    : fraudProb >= 0.3
                    ? "bg-amber-100"
                    : "bg-emerald-100"
                }
              >
                {assessment.fraud?.risk_level?.toUpperCase() || "LOW"}
              </Badge>
            </div>
            <div className="mt-3">
              <span className="text-3xl font-bold text-slate-900">
                {fraudPercent}%
              </span>
              <span className="ml-1 text-xs text-slate-500">suspicion score</span>
            </div>
            {/* Progress Bar */}
            <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-slate-100">
              <div
                className={`h-full rounded-full transition-all ${
                  fraudProb >= 0.65
                    ? "bg-rose-500"
                    : fraudProb >= 0.3
                    ? "bg-amber-500"
                    : "bg-emerald-500"
                }`}
                style={{ width: `${Math.min(100, Math.max(5, fraudProb * 100))}%` }}
              />
            </div>
          </div>
          <p className="mt-3 text-[11px] text-slate-400">
            Model: {assessment.model_versions?.fraud || "FRD-MNV2-001"}
          </p>
        </Card>

        {/* 2. Severity Tier */}
        <Card className="flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase text-slate-500">
                Damage Severity
              </span>
              <Badge
                color={
                  assessment.severity?.predicted_class === "severe"
                    ? "text-rose-800"
                    : assessment.severity?.predicted_class === "moderate"
                    ? "text-amber-800"
                    : "text-blue-800"
                }
                bg={
                  assessment.severity?.predicted_class === "severe"
                    ? "bg-rose-100"
                    : assessment.severity?.predicted_class === "moderate"
                    ? "bg-amber-100"
                    : "bg-blue-100"
                }
              >
                {assessment.severity?.predicted_class?.toUpperCase() || "MODERATE"}
              </Badge>
            </div>
            <div className="mt-3">
              <span className="text-3xl font-bold text-slate-900 capitalize">
                {assessment.severity?.predicted_class || "—"}
              </span>
              <p className="text-xs text-slate-500 mt-0.5">
                Confidence:{" "}
                {assessment.severity?.confidence
                  ? `${(assessment.severity.confidence * 100).toFixed(0)}%`
                  : "—"}
              </p>
            </div>
          </div>
          <p className="mt-3 text-[11px] text-slate-400">
            Model: {assessment.model_versions?.severity || "SEV-MNV2-001"}
          </p>
        </Card>

        {/* 3. Damaged Part Location */}
        <Card className="flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase text-slate-500">
                Primary Damaged Part
              </span>
              <Badge color="text-indigo-800" bg="bg-indigo-100">
                CNN Location
              </Badge>
            </div>
            <div className="mt-3">
              <span className="text-2xl font-bold text-slate-900 capitalize">
                {assessment.location?.predicted_part?.replace(/_/g, " ") || "Damage"}
              </span>
              <p className="text-xs text-slate-500 mt-0.5">
                Confidence:{" "}
                {assessment.location?.confidence
                  ? `${(assessment.location.confidence * 100).toFixed(0)}%`
                  : "—"}
              </p>
            </div>
          </div>
          <p className="mt-3 text-[11px] text-slate-400">
            Model: {assessment.model_versions?.location || "LOC-MNV2-001"}
          </p>
        </Card>

        {/* 4. Cost Range */}
        <Card className="flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase text-slate-500">
                Estimated Repair Cost
              </span>
              <Badge color="text-teal-800" bg="bg-teal-100">
                INR (₹)
              </Badge>
            </div>
            <div className="mt-3">
              <span className="text-2xl font-bold text-slate-900">
                {assessment.cost ? formatINR(assessment.cost.min_cost) : "₹0"}
              </span>
              <span className="text-xs text-slate-500 mx-1">to</span>
              <span className="text-2xl font-bold text-slate-900">
                {assessment.cost ? formatINR(assessment.cost.max_cost) : "₹0"}
              </span>
              <p className="text-xs text-slate-500 mt-0.5">
                Vehicle: {assessment.cost?.vehicle_segment || "compact"}
              </p>
            </div>
          </div>
          <p className="mt-3 text-[11px] text-slate-400">
            Rule Table: {assessment.model_versions?.costing || "COST-RULES-001"}
          </p>
        </Card>
      </div>

      {/* Itemized Cost Breakdown Table */}
      {assessment.cost && assessment.cost.breakdown.length > 0 && (
        <Card>
          <h3 className="mb-3 text-sm font-bold text-slate-900 uppercase tracking-wide">
            Itemized Repair & Replacement Estimate
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="pb-2 font-semibold">Damaged Part</th>
                  <th className="pb-2 font-semibold">Severity Tier</th>
                  <th className="pb-2 font-semibold">Action</th>
                  <th className="pb-2 font-semibold text-right">Min Cost (₹)</th>
                  <th className="pb-2 font-semibold text-right">Max Cost (₹)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-slate-700">
                {assessment.cost.breakdown.map((item, idx) => (
                  <tr key={idx} className="hover:bg-slate-50">
                    <td className="py-2.5 font-medium capitalize">
                      {item.part.replace(/_/g, " ")}
                    </td>
                    <td className="py-2.5 capitalize">{item.severity}</td>
                    <td className="py-2.5">
                      <span
                        className={`rounded px-1.5 py-0.5 font-semibold text-[10px] ${
                          item.action === "replace"
                            ? "bg-rose-100 text-rose-700"
                            : "bg-blue-100 text-blue-700"
                        }`}
                      >
                        {item.action.toUpperCase()}
                      </span>
                    </td>
                    <td className="py-2.5 text-right font-medium">
                      {formatINR(item.min_cost)}
                    </td>
                    <td className="py-2.5 text-right font-medium">
                      {formatINR(item.max_cost)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Disclaimer per AGENTS.md §13 */}
      <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-xs text-slate-500 leading-relaxed">
        <p className="font-semibold text-slate-700 mb-1">
          ⚖️ Prototype Assessment Disclaimer
        </p>
        <p>
          This evaluation is an AI-assisted triage recommendation generated by ClaimVision AI for local demonstration.
          It does not represent a legally binding insurer claim approval or denial. Physical surveyor inspection remains
          required where indicated.
        </p>
      </div>
    </div>
  );
}
