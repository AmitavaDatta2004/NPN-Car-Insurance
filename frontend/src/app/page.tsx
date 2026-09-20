/**
 * ClaimVision AI — landing page placeholder.
 *
 * Phase 0: placeholder only.
 * Full customer and reviewer UIs are implemented in Phases 14 and 15.
 */
export default function HomePage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-slate-50 p-8">
      <div className="max-w-2xl text-center">
        <h1 className="mb-4 text-4xl font-bold text-slate-900">
          ClaimVision AI
        </h1>
        <p className="mb-2 text-lg text-slate-600">
          AI-assisted vehicle insurance claim triage system
        </p>
        <p className="mb-8 text-sm text-slate-500">
          Local presentation prototype — Phase 0 skeleton
        </p>

        <div className="rounded-lg border border-slate-200 bg-white p-6 text-left shadow-sm">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">
            Implementation status
          </h2>
          <ul className="space-y-2 text-sm text-slate-700">
            <li>
              <span className="mr-2">✅</span>Phase 0 — Repository and controls
            </li>
            <li>
              <span className="mr-2">⏳</span>Phase 1 — Fraud dataset audit
            </li>
            <li>
              <span className="mr-2">⏳</span>Phase 2 — Fraud classifier
            </li>
            <li>
              <span className="mr-2">⏳</span>Phase 3–10 — ML models
            </li>
            <li>
              <span className="mr-2">⏳</span>Phase 11 — Unified inference
            </li>
            <li>
              <span className="mr-2">⏳</span>Phase 12–13 — Backend APIs
            </li>
            <li>
              <span className="mr-2">⏳</span>Phase 14 — Customer UI
            </li>
            <li>
              <span className="mr-2">⏳</span>Phase 15 — Reviewer dashboard
            </li>
          </ul>
        </div>

        <p className="mt-8 text-xs text-slate-400">
          Backend API:{" "}
          <a
            href="http://127.0.0.1:8000/api/v1/docs"
            className="underline hover:text-slate-600"
            target="_blank"
            rel="noreferrer"
          >
            http://127.0.0.1:8000/api/v1/docs
          </a>
        </p>
      </div>
    </main>
  );
}
