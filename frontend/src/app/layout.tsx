import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ClaimVision AI",
  description:
    "AI-assisted vehicle insurance claim triage — local presentation prototype.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
