"use client";

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from "recharts";

interface RouteDistributionChartProps {
  statusCounts: Record<string, number>;
}

const ROUTE_COLORS: Record<string, string> = {
  FAST_TRACK_ELIGIBLE: "#10B981", // emerald
  FRAUD_REVIEW: "#EF4444",        // rose
  MANUAL_DAMAGE_REVIEW: "#F59E0B",// amber
  MORE_EVIDENCE_REQUIRED: "#F97316", // orange
  TECHNICAL_REVIEW: "#64748B",    // slate
  COMPLETED: "#0D9488",           // teal
  SUBMITTED: "#3B82F6",           // blue
  DRAFT: "#94A3B8",               // slate-400
};

const ROUTE_LABELS: Record<string, string> = {
  FAST_TRACK_ELIGIBLE: "Fast-Track Eligible",
  FRAUD_REVIEW: "Fraud Review",
  MANUAL_DAMAGE_REVIEW: "Damage Review",
  MORE_EVIDENCE_REQUIRED: "Needs Evidence",
  TECHNICAL_REVIEW: "Technical Review",
  COMPLETED: "Completed",
  SUBMITTED: "Submitted",
  DRAFT: "Draft",
};

export default function RouteDistributionChart({ statusCounts }: RouteDistributionChartProps) {
  const data = Object.entries(statusCounts)
    .filter(([_, value]) => value > 0)
    .map(([key, value]) => ({
      name: ROUTE_LABELS[key] || key,
      key,
      value,
    }));

  if (data.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-xs text-slate-400">
        No claims recorded yet.
      </div>
    );
  }

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={50}
            outerRadius={80}
            paddingAngle={3}
            dataKey="value"
          >
            {data.map((entry) => (
              <Cell
                key={entry.key}
                fill={ROUTE_COLORS[entry.key] || "#6366F1"}
              />
            ))}
          </Pie>
          <Tooltip
            formatter={(value: number, name: string) => [`${value} Claims`, name]}
            contentStyle={{
              backgroundColor: "#1E293B",
              borderRadius: "8px",
              border: "none",
              color: "#F8FAFC",
              fontSize: "12px",
            }}
          />
          <Legend
            verticalAlign="bottom"
            height={36}
            iconType="circle"
            wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
