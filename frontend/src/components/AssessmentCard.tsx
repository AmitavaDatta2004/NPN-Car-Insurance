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
  const fraudFlag = assessment.fraud?.flag ?? (assessment.fraud && assessment.fraud.probability > 0.50 ? 1 : 0);
  const isFraud = fraudFlag === 1;
  const genai = assessment.genai_gate;

  if (genai && !genai.is_vehicle) {
    return (
      <div className="rounded-2xl border border-rose-200 bg-rose-50/70 p-8 text-center shadow-xs">
        <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-2xl bg-rose-100 text-rose-600 text-2xl font-bold">
          ✕
        </div>
        <h3 className="text-xl font-bold text-rose-900">
          Not a car
        </h3>
        <p className="mt-2 text-sm text-slate-600">
          Please upload a photo of a car.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Route Decision Banner - Simple, Non-Technical */}
      <div
        className={`rounded-2xl border p-6 transition-all shadow-sm ${routeInfo.bg} ${routeInfo.border}`}
      >
        <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
          Status
        </span>
        <h2 className={`text-2xl font-black tracking-tight mt-0.5 ${routeInfo.color}`}>
          {isFraud ? "Fraud Detected" : routeInfo.label}
        </h2>
        <p className="mt-1.5 text-sm text-slate-700 leading-relaxed max-w-2xl">
          {isFraud
            ? "This claim has been flagged for review."
            : routeInfo.description}
        </p>
      </div>

      {/* 4 Clean, Simple Result Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {/* 1. Authenticity: Fraud or No fraud */}
        <Card className="flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase text-slate-500">
              Authenticity
            </span>
            <Badge
              color={isFraud ? "text-rose-800" : "text-emerald-800"}
              bg={isFraud ? "bg-rose-100 border border-rose-300" : "bg-emerald-100 border border-emerald-300"}
            >
              {isFraud ? "FRAUD" : "NO FRAUD"}
            </Badge>
          </div>
          <div className="mt-3">
            <span className={`text-2xl font-black ${isFraud ? "text-rose-700" : "text-emerald-700"}`}>
              {isFraud ? "Fraud" : "No fraud"}
            </span>
            <p className="text-xs text-slate-500 mt-1">
              {isFraud ? "Flagged for review" : "Image verified"}
            </p>
          </div>
        </Card>

        {/* 2. Damage Level */}
        <Card className="flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase text-slate-500">
              Damage
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
                  ? "bg-rose-100 border border-rose-300"
                  : assessment.severity?.predicted_class === "moderate"
                  ? "bg-amber-100 border border-amber-300"
                  : "bg-blue-100 border border-blue-300"
              }
            >
              {assessment.severity?.predicted_class?.toUpperCase() || "MODERATE"}
            </Badge>
          </div>
          <div className="mt-3">
            <span className="text-2xl font-black text-slate-900 capitalize">
              {assessment.severity?.predicted_class || "Moderate"}
            </span>
            <p className="text-xs text-slate-500 mt-1">
              Collision severity tier
            </p>
          </div>
        </Card>

        {/* 3. Damaged Part */}
        <Card className="flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase text-slate-500">
              Damaged Part
            </span>
            <Badge color="text-indigo-800" bg="bg-indigo-100">
              Part
            </Badge>
          </div>
          <div className="mt-3">
            <span className="text-2xl font-bold text-slate-900 capitalize">
              {assessment.location?.predicted_part?.replace(/_/g, " ") || "Body Damage"}
            </span>
            <p className="text-xs text-slate-500 mt-1">
              Primary impact area
            </p>
          </div>
        </Card>

        {/* 4. Estimated Cost */}
        <Card className="flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase text-slate-500">
              Estimated Cost
            </span>
            <Badge color="text-teal-800" bg="bg-teal-100">
              INR (₹)
            </Badge>
          </div>
          <div className="mt-3">
            <div className="flex items-baseline gap-1">
              <span className="text-2xl font-bold text-slate-900">
                {assessment.cost ? formatINR(assessment.cost.min_cost) : "₹0"}
              </span>
              <span className="text-xs text-slate-500">to</span>
              <span className="text-2xl font-bold text-slate-900">
                {assessment.cost ? formatINR(assessment.cost.max_cost) : "₹0"}
              </span>
            </div>
            <p className="text-xs text-slate-500 mt-1 capitalize">
              Vehicle: {assessment.cost?.vehicle_segment || "compact"}
            </p>
          </div>
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
