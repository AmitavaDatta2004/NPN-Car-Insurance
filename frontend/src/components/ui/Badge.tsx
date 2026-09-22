import React from "react";

export type BadgeVariant =
  | "default"
  | "primary"
  | "secondary"
  | "success"
  | "warning"
  | "danger";

interface BadgeProps {
  children: React.ReactNode;
  variant?: BadgeVariant;
  color?: string;
  bg?: string;
  border?: string;
  className?: string;
}

const VARIANT_STYLES: Record<BadgeVariant, { color: string; bg: string; border: string }> = {
  default: { color: "text-slate-700", bg: "bg-slate-100", border: "border-slate-200" },
  primary: { color: "text-indigo-700", bg: "bg-indigo-50", border: "border-indigo-200" },
  secondary: { color: "text-slate-600", bg: "bg-slate-50", border: "border-slate-200" },
  success: { color: "text-emerald-800", bg: "bg-emerald-50", border: "border-emerald-200" },
  warning: { color: "text-amber-800", bg: "bg-amber-50", border: "border-amber-200" },
  danger: { color: "text-rose-800", bg: "bg-rose-50", border: "border-rose-200" },
};

export default function Badge({
  children,
  variant,
  color,
  bg,
  border,
  className = "",
}: BadgeProps) {
  const styles = variant ? VARIANT_STYLES[variant] : null;

  const resolvedColor = color || styles?.color || "text-slate-700";
  const resolvedBg = bg || styles?.bg || "bg-slate-100";
  const resolvedBorder = border || styles?.border || "border-transparent";

  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold ${resolvedColor} ${resolvedBg} ${resolvedBorder} ${className}`}
    >
      {children}
    </span>
  );
}
