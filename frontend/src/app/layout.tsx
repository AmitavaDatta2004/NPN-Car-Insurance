import type { Metadata } from "next";
import Navbar from "@/components/Navbar";
import "./globals.css";

export const metadata: Metadata = {
  title: "ClaimVision AI — Vehicle Insurance Claim Triage",
  description:
    "Explainable AI-assisted vehicle insurance claim assessment prototype.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50 text-slate-900 font-sans antialiased flex flex-col">
        <Navbar />
        <div className="flex-1">{children}</div>
        <footer className="border-t border-slate-200 bg-white py-6 text-center text-xs text-slate-500">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <p>
              ClaimVision AI • Local Judge Demonstration Prototype • Phase 15 Customer UI
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
