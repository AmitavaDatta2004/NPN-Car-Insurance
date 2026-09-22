import React from "react";

export type AlertVariant = "info" | "success" | "warning" | "danger";

interface AlertProps {
  children: React.ReactNode;
  variant?: AlertVariant;
  title?: string;
  className?: string;
}

const ALERT_STYLES: Record<
  AlertVariant,
  { container: string; title: string; text: string; icon: string }
> = {
  info: {
    container: "bg-blue-50 border-blue-200",
    title: "text-blue-900",
    text: "text-blue-800",
    icon: "ℹ️",
  },
  success: {
    container: "bg-emerald-50 border-emerald-200",
    title: "text-emerald-900",
    text: "text-emerald-800",
    icon: "✓",
  },
  warning: {
    container: "bg-amber-50 border-amber-200",
    title: "text-amber-900",
    text: "text-amber-800",
    icon: "⚠️",
  },
  danger: {
    container: "bg-rose-50 border-rose-200",
    title: "text-rose-900",
    text: "text-rose-800",
    icon: "✕",
  },
};

export default function Alert({
  children,
  variant = "info",
  title,
  className = "",
}: AlertProps) {
  const styles = ALERT_STYLES[variant];

  return (
    <div
      className={`rounded-xl border p-4 text-xs ${styles.container} ${className}`}
      role="alert"
    >
      <div className="flex items-start gap-2.5">
        <span className="text-sm select-none">{styles.icon}</span>
        <div className="flex-1">
          {title && (
            <h4 className={`font-semibold mb-0.5 ${styles.title}`}>{title}</h4>
          )}
          <div className={styles.text}>{children}</div>
        </div>
      </div>
    </div>
  );
}
