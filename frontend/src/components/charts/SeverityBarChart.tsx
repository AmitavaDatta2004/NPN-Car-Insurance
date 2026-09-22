"use client";

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

interface SeverityBarChartProps {
  severityDistribution: Record<string, number>;
}

const SEVERITY_COLORS: Record<string, string> = {
  minor: "#10B981",    // emerald
  moderate: "#F59E0B", // amber
  severe: "#EF4444",   // rose
};

export default function SeverityBarChart({ severityDistribution }: SeverityBarChartProps) {
  const tiers = ["minor", "moderate", "severe"];
  const data = tiers.map((tier) => ({
    tier: tier.toUpperCase(),
    key: tier,
    count: severityDistribution[tier] || 0,
  }));

  const total = data.reduce((sum, item) => sum + item.count, 0);

  if (total === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-xs text-slate-400">
        No severity assessments recorded yet.
      </div>
    );
  }

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 15, right: 15, left: -20, bottom: 5 }}>
          <XAxis dataKey="tier" tick={{ fontSize: 11, fill: "#64748B" }} />
          <YAxis allowDecimals={false} tick={{ fontSize: 11, fill: "#64748B" }} />
          <Tooltip
            formatter={(value: number) => [`${value} Claims`, "Count"]}
            contentStyle={{
              backgroundColor: "#1E293B",
              borderRadius: "8px",
              border: "none",
              color: "#F8FAFC",
              fontSize: "12px",
            }}
          />
          <Bar dataKey="count" radius={[4, 4, 0, 0]}>
            {data.map((entry) => (
              <Cell
                key={entry.key}
                fill={SEVERITY_COLORS[entry.key] || "#6366F1"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
