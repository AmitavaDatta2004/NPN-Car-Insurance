"use client";

import { useState } from "react";
import Link from "next/link";
import { Claim } from "@/types/claim";
import { formatDate, formatINR, getRouteInfo, getStatusBadge } from "@/lib/api";
import Badge from "@/components/ui/Badge";
import Button from "@/components/ui/Button";

interface ReviewQueueTableProps {
  claims: Claim[];
  isLoading?: boolean;
  onRefresh?: () => void;
}

export default function ReviewQueueTable({
  claims,
  isLoading = false,
  onRefresh,
}: ReviewQueueTableProps) {
  const [filter, setFilter] = useState<"ALL_FLAGGED" | "FRAUD_REVIEW" | "MANUAL_DAMAGE_REVIEW" | "ALL">("ALL_FLAGGED");
  const [searchTerm, setSearchTerm] = useState("");

  const filteredClaims = claims.filter((claim) => {
    // Route / status filter
    if (filter === "ALL_FLAGGED") {
      const isFlagged =
        claim.status === "FRAUD_REVIEW" ||
        claim.status === "MANUAL_DAMAGE_REVIEW" ||
        claim.assessment?.route === "FRAUD_REVIEW" ||
        claim.assessment?.route === "MANUAL_DAMAGE_REVIEW";
      if (!isFlagged) return false;
    } else if (filter === "FRAUD_REVIEW") {
      const isFraud =
        claim.status === "FRAUD_REVIEW" || claim.assessment?.route === "FRAUD_REVIEW";
      if (!isFraud) return false;
    } else if (filter === "MANUAL_DAMAGE_REVIEW") {
      const isDamage =
        claim.status === "MANUAL_DAMAGE_REVIEW" ||
        claim.assessment?.route === "MANUAL_DAMAGE_REVIEW";
      if (!isDamage) return false;
    }

    // Search filter
    if (searchTerm.trim()) {
      const term = searchTerm.toLowerCase();
      const matchId = claim.id.toLowerCase().includes(term);
      const matchPolicy = claim.policy_number.toLowerCase().includes(term);
      const matchVehicle = `${claim.vehicle_year} ${claim.vehicle_make} ${claim.vehicle_model}`
        .toLowerCase()
        .includes(term);
      return matchId || matchPolicy || matchVehicle;
    }

    return true;
  });

  return (
    <div className="space-y-4">
      {/* Search & Filter Controls */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => setFilter("ALL_FLAGGED")}
            className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-colors ${
              filter === "ALL_FLAGGED"
                ? "bg-slate-900 text-white shadow-sm"
                : "bg-slate-100 text-slate-700 hover:bg-slate-200"
            }`}
          >
            Flagged for Review (
            {
              claims.filter(
                (c) =>
                  c.status === "FRAUD_REVIEW" ||
                  c.status === "MANUAL_DAMAGE_REVIEW" ||
                  c.assessment?.route === "FRAUD_REVIEW" ||
                  c.assessment?.route === "MANUAL_DAMAGE_REVIEW"
              ).length
            }
            )
          </button>
          <button
            onClick={() => setFilter("FRAUD_REVIEW")}
            className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-colors ${
              filter === "FRAUD_REVIEW"
                ? "bg-rose-700 text-white shadow-sm"
                : "bg-rose-50 text-rose-700 hover:bg-rose-100"
            }`}
          >
            Fraud Alerts (
            {
              claims.filter(
                (c) =>
                  c.status === "FRAUD_REVIEW" ||
                  c.assessment?.route === "FRAUD_REVIEW"
              ).length
            }
            )
          </button>
          <button
            onClick={() => setFilter("MANUAL_DAMAGE_REVIEW")}
            className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-colors ${
              filter === "MANUAL_DAMAGE_REVIEW"
                ? "bg-amber-700 text-white shadow-sm"
                : "bg-amber-50 text-amber-800 hover:bg-amber-100"
            }`}
          >
            Damage Overrides (
            {
              claims.filter(
                (c) =>
                  c.status === "MANUAL_DAMAGE_REVIEW" ||
                  c.assessment?.route === "MANUAL_DAMAGE_REVIEW"
              ).length
            }
            )
          </button>
          <button
            onClick={() => setFilter("ALL")}
            className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-colors ${
              filter === "ALL"
                ? "bg-indigo-700 text-white shadow-sm"
                : "bg-slate-100 text-slate-700 hover:bg-slate-200"
            }`}
          >
            All Claims ({claims.length})
          </button>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="text"
            placeholder="Search claim, policy, vehicle..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full sm:w-64 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs text-slate-900 placeholder-slate-400 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          />
          {onRefresh && (
            <Button
              variant="outline"
              size="sm"
              onClick={onRefresh}
              isLoading={isLoading}
              title="Refresh queue"
            >
              Refresh
            </Button>
          )}
        </div>
      </div>

      {/* Table */}
      <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-slate-200 bg-slate-50 font-semibold text-slate-600">
              <tr>
                <th className="px-4 py-3">Claim & Policy</th>
                <th className="px-4 py-3">Vehicle Details</th>
                <th className="px-4 py-3">Incident Date</th>
                <th className="px-4 py-3">Triage Route</th>
                <th className="px-4 py-3">AI Finding</th>
                <th className="px-4 py-3">Review Status</th>
                <th className="px-4 py-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredClaims.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-4 py-12 text-center text-slate-400">
                    <p className="text-sm font-medium">No claims match the selected criteria</p>
                    <p className="mt-1 text-xs">
                      {searchTerm ? "Try broadening your search term." : "The review queue is currently clear."}
                    </p>
                  </td>
                </tr>
              ) : (
                filteredClaims.map((claim) => {
                  const assessment = claim.assessment;
                  const routeInfo = assessment ? getRouteInfo(assessment.route) : null;
                  const statusInfo = getStatusBadge(claim.status);
                  const isReviewed = claim.review !== null && claim.review.decision !== "PENDING";

                  return (
                    <tr key={claim.id} className="hover:bg-slate-50/80 transition-colors">
                      <td className="px-4 py-3">
                        <div className="font-semibold text-slate-900">
                          {claim.policy_number}
                        </div>
                        <div className="font-mono text-[11px] text-slate-400 truncate max-w-[140px]">
                          {claim.id}
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <div className="font-medium text-slate-800">
                          {claim.vehicle_year} {claim.vehicle_make} {claim.vehicle_model}
                        </div>
                        <div className="text-[11px] text-slate-500 capitalize">
                          {claim.vehicle_segment} segment
                        </div>
                      </td>
                      <td className="px-4 py-3 text-slate-600">
                        {formatDate(claim.incident_date || claim.created_at)}
                      </td>
                      <td className="px-4 py-3">
                        {routeInfo ? (
                          <span
                            className={`inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium ${routeInfo.bg} ${routeInfo.color}`}
                          >
                            {routeInfo.label}
                          </span>
                        ) : (
                          <Badge variant="secondary">{claim.status}</Badge>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        {assessment ? (
                          <div className="space-y-0.5">
                            {assessment.fraud && assessment.fraud.risk_level === "high" && (
                              <div className="font-medium text-rose-700">
                                Fraud Risk: {(assessment.fraud.probability * 100).toFixed(0)}%
                              </div>
                            )}
                            {assessment.severity && (
                              <div className="text-slate-700 capitalize">
                                Severity: <span className="font-semibold">{assessment.severity.predicted_class}</span>
                              </div>
                            )}
                            {assessment.cost && (
                              <div className="text-slate-500 text-[11px]">
                                {formatINR(assessment.cost.min_cost)} – {formatINR(assessment.cost.max_cost)}
                              </div>
                            )}
                          </div>
                        ) : (
                          <span className="text-slate-400">Pending assessment</span>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        {isReviewed ? (
                          <span className="inline-flex items-center gap-1 rounded-full bg-teal-50 px-2 py-0.5 text-[11px] font-semibold text-teal-800">
                            ✓ {claim.review?.decision}
                          </span>
                        ) : (
                          <span
                            className={`inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium ${statusInfo.bg} ${statusInfo.color}`}
                          >
                            {statusInfo.label}
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <Link href={`/reviewer/claims/${claim.id}`}>
                          <Button size="sm" variant={isReviewed ? "outline" : "primary"}>
                            {isReviewed ? "View Audit" : "Review"}
                          </Button>
                        </Link>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
