"use client";

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

interface LocationPartChartProps {
  locationDistribution: Record<string, number>;
}

const PART_NAMES: Record<string, string> = {
  front_bumper: "Front Bumper",
  rear_bumper: "Rear Bumper",
  hood: "Hood",
  door: "Door",
  headlamp: "Headlamp",
};

export default function LocationPartChart({ locationDistribution }: LocationPartChartProps) {
  const parts = ["front_bumper", "rear_bumper", "hood", "door", "headlamp"];
  const data = parts.map((partKey) => ({
    part: PART_NAMES[partKey] || partKey,
    count: locationDistribution[partKey] || 0,
  }));

  const total = data.reduce((sum, item) => sum + item.count, 0);

  if (total === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-xs text-slate-400">
        No damaged part classifications recorded yet.
      </div>
    );
  }

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 15, right: 15, left: -20, bottom: 5 }}>
          <XAxis
            dataKey="part"
            tick={{ fontSize: 10, fill: "#64748B" }}
            interval={0}
          />
          <YAxis allowDecimals={false} tick={{ fontSize: 11, fill: "#64748B" }} />
          <Tooltip
            formatter={(value: number) => [`${value} Claims`, "Occurrences"]}
            contentStyle={{
              backgroundColor: "#1E293B",
              borderRadius: "8px",
              border: "none",
              color: "#F8FAFC",
              fontSize: "12px",
            }}
          />
          <Bar dataKey="count" fill="#8B5CF6" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
