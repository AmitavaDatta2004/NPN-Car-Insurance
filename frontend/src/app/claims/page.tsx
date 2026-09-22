"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { Claim } from "@/types/claim";
import { formatDate, getRouteInfo, getStatusBadge, listClaims } from "@/lib/api";
import Badge from "@/components/ui/Badge";
import Button from "@/components/ui/Button";
import Card from "@/components/ui/Card";

export default function ClaimsListPage() {
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
        err instanceof Error ? err.message : "Failed to load claims. Is the backend server running?"
      );
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchClaims();
  }, []);

  return (
    <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Active Claims</h1>
          <p className="text-sm text-slate-500">
            Real-time list of in-memory insurance claims submitted to ClaimVision AI.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline" size="sm" onClick={fetchClaims} isLoading={isLoading}>
            ↻ Refresh
          </Button>
          <Link href="/claims/new">
            <Button size="sm">+ New Claim</Button>
          </Link>
        </div>
      </div>

      {error && (
        <div className="mb-6 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
          ⚠️ {error}
        </div>
      )}

      {isLoading ? (
        <div className="flex h-64 items-center justify-center">
          <div className="flex flex-col items-center gap-2">
            <div className="h-8 w-8 animate-spin rounded-full border-4 border-indigo-600 border-t-transparent" />
            <p className="text-xs text-slate-500">Loading claims from backend...</p>
          </div>
        </div>
      ) : claims.length === 0 ? (
        <Card className="text-center py-12">
          <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-slate-100 text-slate-400">
            📋
          </div>
          <h3 className="text-base font-semibold text-slate-900">No Claims Found</h3>
          <p className="mt-1 text-xs text-slate-500 max-w-md mx-auto">
            You haven&apos;t filed any claims yet. Create a new claim to experience the automated AI assessment pipeline.
          </p>
          <div className="mt-6">
            <Link href="/claims/new">
              <Button>File a New Claim</Button>
            </Link>
          </div>
        </Card>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {claims.map((claim) => {
            const statusBadge = getStatusBadge(claim.status);
            const routeBadge = claim.assessment ? getRouteInfo(claim.assessment.route) : null;

            return (
              <Card key={claim.id} className="flex flex-col justify-between hover:shadow-md transition-shadow">
                <div>
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-xs font-bold text-slate-600">
                      {claim.policy_number}
                    </span>
                    <Badge color={statusBadge.color} bg={statusBadge.bg}>
                      {statusBadge.label}
                    </Badge>
                  </div>

                  <h3 className="mt-2 text-base font-bold text-slate-900">
                    {claim.vehicle_year} {claim.vehicle_make} {claim.vehicle_model}
                  </h3>
                  <p className="text-xs text-slate-500 capitalize">
                    Segment: {claim.vehicle_segment} • Photos: {claim.images.length}
                  </p>

                  {claim.incident_description && (
                    <p className="mt-2 text-xs text-slate-600 line-clamp-2">
                      {claim.incident_description}
                    </p>
                  )}

                  {routeBadge && (
                    <div className="mt-3">
                      <span className={`inline-block rounded-md border px-2 py-0.5 text-xs font-semibold ${routeBadge.color} ${routeBadge.bg} ${routeBadge.border}`}>
                        {routeBadge.label}
                      </span>
                    </div>
                  )}
                </div>

                <div className="mt-5 border-t border-slate-100 pt-3 flex items-center justify-between text-xs">
                  <span className="text-[11px] text-slate-400">
                    {formatDate(claim.created_at)}
                  </span>
                  <div className="flex items-center gap-2">
                    <Link
                      href={`/claims/${claim.id}/timeline`}
                      className="text-slate-500 hover:text-slate-800"
                    >
                      Timeline
                    </Link>
                    {claim.assessment ? (
                      <Link
                        href={`/claims/${claim.id}/result`}
                        className="font-semibold text-indigo-600 hover:text-indigo-800"
                      >
                        View Result →
                      </Link>
                    ) : (
                      <Link
                        href={`/claims/${claim.id}/processing`}
                        className="font-semibold text-indigo-600 hover:text-indigo-800"
                      >
                        Assess →
                      </Link>
                    )}
                  </div>
                </div>
              </Card>
            );
          })}
        </div>
      )}
    </main>
  );
}
