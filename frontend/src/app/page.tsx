import Link from "next/link";
import Button from "@/components/ui/Button";

export default function HomePage() {
  return (
    <main className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
      {/* Hero Section */}
      <div className="text-center max-w-3xl mx-auto">
        <span className="inline-flex items-center rounded-full bg-indigo-100 px-3 py-1 text-xs font-semibold text-indigo-700 mb-4">
          Judge Demonstration Prototype
        </span>
        <h1 className="text-4xl font-extrabold tracking-tight text-slate-900 sm:text-5xl sm:leading-tight">
          Explainable AI-Assisted Vehicle Claim Triage
        </h1>
        <p className="mt-4 text-lg text-slate-600 leading-relaxed">
          ClaimVision AI combines OpenCV evidence verification, deep-learning fraud risk scoring,
          damage severity tiering, and YOLO localization to triage auto claims in seconds.
        </p>

        <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
          <Link href="/claims/new">
            <Button size="lg" className="shadow-md">
              File a New Claim →
            </Button>
          </Link>
          <Link href="/claims">
            <Button variant="secondary" size="lg">
              View Active Claims
            </Button>
          </Link>
          <Link href="/reviewer/dashboard">
            <Button variant="outline" size="lg">
              Adjuster Dashboard
            </Button>
          </Link>
        </div>
      </div>

      {/* 4 Pillars Architecture */}
      <div className="mt-20">
        <h2 className="text-center text-xs font-bold uppercase tracking-wider text-slate-400">
          Integrated Multi-Stage Assessment Engine
        </h2>

        <div className="mt-6 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs hover:border-indigo-300 transition-all">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-100 text-blue-600 font-bold mb-4">
              1
            </div>
            <h3 className="text-base font-bold text-slate-900">Evidence Quality</h3>
            <p className="mt-2 text-xs text-slate-500 leading-relaxed">
              OpenCV Laplacian variance blur score, brightness, contrast, minimum resolution checks, and pHash duplicate identification.
            </p>
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs hover:border-indigo-300 transition-all">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-rose-100 text-rose-600 font-bold mb-4">
              2
            </div>
            <h3 className="text-base font-bold text-slate-900">Fraud Risk Scoring</h3>
            <p className="mt-2 text-xs text-slate-500 leading-relaxed">
              MobileNetV2 suspicious-image classifier trained with balanced resampling. High fraud scores route immediately to SIU review.
            </p>
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs hover:border-indigo-300 transition-all">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-amber-100 text-amber-600 font-bold mb-4">
              3
            </div>
            <h3 className="text-base font-bold text-slate-900">Severity & Location</h3>
            <p className="mt-2 text-xs text-slate-500 leading-relaxed">
              MobileNetV2/ViT classifies damage tier (Minor/Moderate/Severe) alongside dominant damaged-part location (Front Bumper, Hood, etc.).
            </p>
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs hover:border-indigo-300 transition-all">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-emerald-100 text-emerald-600 font-bold mb-4">
              4
            </div>
            <h3 className="text-base font-bold text-slate-900">YOLO Boxes & Costing</h3>
            <p className="mt-2 text-xs text-slate-500 leading-relaxed">
              Ultralytics YOLOv8 localizes damage regions. Rule-based cost table computes itemized repair/replacement ranges in INR.
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
