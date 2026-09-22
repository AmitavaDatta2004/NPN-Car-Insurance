"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { Claim, TimelineEvent } from "@/types/claim";
import { formatDate, getClaim, getClaimTimeline, getStatusBadge } from "@/lib/api";
import Badge from "@/components/ui/Badge";
import Button from "@/components/ui/Button";
import Card from "@/components/ui/Card";

export default function ClaimTimelinePage() {
  const params = useParams();
  const claimId = params.id as string;

  const [claim, setClaim] = useState<Claim | null>(null);
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      setErrorMsg(null);
      try {
        const [c, evs] = await Promise.all([
          getClaim(claimId),
          getClaimTimeline(claimId),
        ]);
        setClaim(c);
        setEvents(evs);
      } catch (err: unknown) {
        setErrorMsg(
          err instanceof Error ? err.message : "Failed to load claim timeline."
        );
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [claimId]);

  return (
    <main className="mx-auto max-w-4xl px-4 py-8 sm:px-6 lg:px-8 space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-200 pb-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Claim Audit Timeline</h1>
          {claim && (
            <p className="text-xs text-slate-500 mt-1">
              Policy: <span className="font-mono font-semibold">{claim.policy_number}</span> •
              Vehicle: <span className="font-semibold">{claim.vehicle_year} {claim.vehicle_make} {claim.vehicle_model}</span>
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {claim?.assessment && (
            <Link href={`/claims/${claimId}/result`}>
              <Button size="sm">View Assessment →</Button>
            </Link>
          )}
          <Link href="/claims">
            <Button variant="outline" size="sm">Back to Claims</Button>
          </Link>
        </div>
      </div>

      {errorMsg && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-xs text-rose-700">
          ⚠️ {errorMsg}
        </div>
      )}

      {isLoading ? (
        <div className="flex h-64 items-center justify-center">
          <div className="flex flex-col items-center gap-2">
            <div className="h-8 w-8 animate-spin rounded-full border-4 border-indigo-600 border-t-transparent" />
            <p className="text-xs text-slate-500">Loading timeline events...</p>
          </div>
        </div>
      ) : events.length === 0 ? (
        <Card className="text-center py-8">
          <p className="text-xs text-slate-500">No timeline events recorded yet.</p>
        </Card>
      ) : (
        <Card>
          <div className="relative border-l-2 border-slate-200 ml-4 pl-6 space-y-6 py-2">
            {events.map((ev, idx) => {
              const toBadge = getStatusBadge(ev.to_status);

              return (
                <div key={ev.id || idx} className="relative">
                  {/* Dot */}
                  <div className="absolute -left-[31px] top-1.5 h-3.5 w-3.5 rounded-full border-2 border-white bg-indigo-600 shadow-xs" />

                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <Badge color={toBadge.color} bg={toBadge.bg}>
                        {toBadge.label}
                      </Badge>
                      {ev.from_status && (
                        <span className="text-[11px] text-slate-400">
                          (from {ev.from_status})
                        </span>
                      )}
                    </div>
                    <span className="text-[11px] text-slate-400 font-mono">
                      {formatDate(ev.timestamp)}
                    </span>
                  </div>

                  <p className="mt-1.5 text-xs text-slate-700 font-medium">
                    {ev.reason}
                  </p>

                  <p className="mt-0.5 text-[11px] text-slate-400">
                    Actor: <span className="font-semibold text-slate-600">{ev.actor}</span>
                  </p>
                </div>
              );
            })}
          </div>
        </Card>
      )}
    </main>
  );
}
