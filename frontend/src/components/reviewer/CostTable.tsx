"use client";

import { CostSummary } from "@/types/claim";
import { formatINR } from "@/lib/api";

interface CostTableProps {
  cost: CostSummary | null;
}

export default function CostTable({ cost }: CostTableProps) {
  if (!cost) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-5 text-center text-slate-400">
        <p className="text-xs">No cost estimation available (e.g., claim routed to fraud review or quality rejected).</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-4">
      <div className="flex items-center justify-between border-b border-slate-100 pb-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-900">
            Itemized Repair & Replacement Estimate
          </h3>
          <p className="text-xs text-slate-500">
            Vehicle segment: <span className="capitalize font-medium text-slate-700">{cost.vehicle_segment}</span>
          </p>
        </div>
        <div className="text-right">
          <span className="text-xs text-slate-500">Estimated Range</span>
          <div className="text-base font-bold text-slate-900">
            {formatINR(cost.min_cost)} – {formatINR(cost.max_cost)}
          </div>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="border-b border-slate-100 bg-slate-50/50 font-semibold text-slate-600">
            <tr>
              <th className="px-3 py-2">Damaged Part</th>
              <th className="px-3 py-2">Assessed Action</th>
              <th className="px-3 py-2">Severity Tier</th>
              <th className="px-3 py-2 text-right">Cost Range ({cost.currency})</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {cost.breakdown.length === 0 ? (
              <tr>
                <td colSpan={4} className="px-3 py-4 text-center text-slate-400">
                  No individual line items estimated.
                </td>
              </tr>
            ) : (
              cost.breakdown.map((item, idx) => (
                <tr key={idx} className="hover:bg-slate-50/60">
                  <td className="px-3 py-2 font-medium capitalize text-slate-800">
                    {item.part.replace(/_/g, " ")}
                  </td>
                  <td className="px-3 py-2">
                    <span
                      className={`inline-flex rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase ${
                        item.action === "replace"
                          ? "bg-rose-50 text-rose-700"
                          : "bg-blue-50 text-blue-700"
                      }`}
                    >
                      {item.action}
                    </span>
                  </td>
                  <td className="px-3 py-2 capitalize text-slate-600">
                    {item.severity}
                  </td>
                  <td className="px-3 py-2 text-right font-medium text-slate-900">
                    {formatINR(item.min_cost)} – {formatINR(item.max_cost)}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {cost.notes && cost.notes.length > 0 && (
        <div className="border-t border-slate-100 pt-3">
          <h4 className="text-[11px] font-medium text-slate-500 uppercase tracking-wider mb-1">
            Cost Engine Notes
          </h4>
          <ul className="space-y-0.5 text-[11px] text-slate-600 list-disc list-inside">
            {cost.notes.map((note, idx) => (
              <li key={idx}>{note}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
