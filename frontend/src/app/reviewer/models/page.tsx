"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import {
  ModelBenchmarkResponse,
  ModelBenchmarkItem,
  ClassificationReportData,
  ClassMetric,
} from "@/types/claim";
import { getModelBenchmarks, setActiveModel } from "@/lib/api";
import Spinner from "@/components/ui/Spinner";
import Alert from "@/components/ui/Alert";
import Button from "@/components/ui/Button";

function ConfusionMatrixTable({
  labels,
  matrix,
  isWinner,
}: {
  labels: string[];
  matrix: number[][];
  isWinner?: boolean;
}) {
  const colTotals = labels.map((_, colIdx) =>
    matrix.reduce((acc, row) => acc + (row[colIdx] || 0), 0)
  );
  const totalSamples = matrix.reduce(
    (acc, row) => acc + row.reduce((rAcc, v) => rAcc + v, 0),
    0
  );

  return (
    <div className="mt-3 overflow-x-auto rounded-xl border border-slate-200 bg-slate-50/90 p-3 shadow-2xs">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
          Confusion Matrix (Counts)
        </span>
        <span className="text-[10px] text-slate-400">
          Actual (rows) vs. Predicted (cols)
        </span>
      </div>

      <table className="w-full border-collapse text-center text-[11px]">
        <thead>
          <tr>
            <th className="p-1 text-left text-[10px] font-semibold text-slate-400 uppercase">
              Actual \ Pred
            </th>
            {labels.map((lbl, idx) => (
              <th
                key={idx}
                className="p-1 text-[10px] font-semibold text-slate-700 capitalize max-w-[70px] truncate"
                title={lbl}
              >
                {lbl.replace("_", " ")}
              </th>
            ))}
            <th className="p-1 text-[10px] font-semibold text-slate-400 uppercase">
              Total
            </th>
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, rIdx) => {
            const rowTotal = row.reduce((a, b) => a + b, 0);
            return (
              <tr key={rIdx} className="border-t border-slate-200/60">
                <td className="p-1.5 text-left font-medium text-slate-700 capitalize text-[10px]">
                  {labels[rIdx]?.replace("_", " ")}
                </td>
                {row.map((val, cIdx) => {
                  const isDiagonal = rIdx === cIdx;
                  return (
                    <td
                      key={cIdx}
                      className={`p-1.5 font-mono text-[11px] font-semibold rounded-xs transition-colors ${
                        isDiagonal
                          ? val > 0
                            ? isWinner
                              ? "bg-emerald-100 text-emerald-950 font-bold border border-emerald-300"
                              : "bg-indigo-100 text-indigo-950 font-bold border border-indigo-300"
                            : "bg-slate-100 text-slate-400"
                          : val > 0
                          ? "bg-rose-50 text-rose-700 border border-rose-100"
                          : "text-slate-300"
                      }`}
                      title={`Actual: ${labels[rIdx]}, Predicted: ${labels[cIdx]} (${val})`}
                    >
                      {val}
                    </td>
                  );
                })}
                <td className="p-1.5 font-mono text-[10px] font-bold text-slate-600 bg-slate-100/60">
                  {rowTotal}
                </td>
              </tr>
            );
          })}
        </tbody>
        <tfoot>
          <tr className="border-t-2 border-slate-300 text-[10px] font-mono text-slate-500 bg-slate-100/40">
            <td className="p-1 text-left font-sans font-semibold text-slate-500 uppercase">
              Total
            </td>
            {colTotals.map((colTotal, cIdx) => (
              <td key={cIdx} className="p-1 font-bold text-slate-600">
                {colTotal}
              </td>
            ))}
            <td className="p-1 font-bold text-slate-800 bg-slate-200/70">
              {totalSamples}
            </td>
          </tr>
        </tfoot>
      </table>
    </div>
  );
}

function ClassificationReportTable({
  report,
  legacyMetrics,
}: {
  report?: ClassificationReportData;
  legacyMetrics?: Record<string, ClassMetric>;
}) {
  if (report) {
    return (
      <div className="mt-2 overflow-x-auto rounded-xl border border-slate-200 bg-white p-2.5 shadow-2xs">
        <div className="flex items-center justify-between mb-1.5 px-0.5">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
            Classification Report
          </span>
          <span className="text-[10px] text-slate-400 font-mono">
            Support: {report.total_support}
          </span>
        </div>
        <table className="w-full text-left text-[11px]">
          <thead>
            <tr className="border-b border-slate-200 text-slate-400 text-[10px] uppercase">
              <th className="pb-1 font-semibold">Class</th>
              <th className="pb-1 font-semibold text-right">Precision</th>
              <th className="pb-1 font-semibold text-right">Recall</th>
              <th className="pb-1 font-semibold text-right">F1-Score</th>
              <th className="pb-1 font-semibold text-right">Support</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 font-mono">
            {Object.entries(report.classes).map(([cls, row], idx) => (
              <tr key={idx} className="hover:bg-slate-50/60">
                <td className="py-1 font-sans font-medium capitalize text-slate-900">
                  {cls.replace("_", " ")}
                </td>
                <td className="py-1 text-right text-slate-700">
                  {(row.precision * 100).toFixed(1)}%
                </td>
                <td className="py-1 text-right text-slate-700">
                  {(row.recall * 100).toFixed(1)}%
                </td>
                <td className="py-1 text-right font-semibold text-slate-900">
                  {(row.f1_score * 100).toFixed(1)}%
                </td>
                <td className="py-1 text-right text-slate-500 font-mono">
                  {row.support}
                </td>
              </tr>
            ))}
          </tbody>
          <tfoot className="border-t-2 border-slate-200 text-[10px] font-mono">
            <tr className="text-slate-800 font-semibold bg-slate-50/50">
              <td className="py-1 font-sans">accuracy</td>
              <td className="py-1 text-right text-slate-400">-</td>
              <td className="py-1 text-right text-slate-400">-</td>
              <td className="py-1 text-right font-bold text-indigo-700">
                {(report.accuracy * 100).toFixed(1)}%
              </td>
              <td className="py-1 text-right text-slate-700">{report.total_support}</td>
            </tr>
            <tr className="text-slate-600">
              <td className="py-0.5 font-sans">macro avg</td>
              <td className="py-0.5 text-right">
                {(report.macro_avg.precision * 100).toFixed(1)}%
              </td>
              <td className="py-0.5 text-right">
                {(report.macro_avg.recall * 100).toFixed(1)}%
              </td>
              <td className="py-0.5 text-right font-semibold text-slate-900">
                {(report.macro_avg.f1_score * 100).toFixed(1)}%
              </td>
              <td className="py-0.5 text-right text-slate-500">
                {report.macro_avg.support}
              </td>
            </tr>
            <tr className="text-slate-600">
              <td className="py-0.5 font-sans">weighted avg</td>
              <td className="py-0.5 text-right">
                {(report.weighted_avg.precision * 100).toFixed(1)}%
              </td>
              <td className="py-0.5 text-right">
                {(report.weighted_avg.recall * 100).toFixed(1)}%
              </td>
              <td className="py-0.5 text-right font-semibold text-slate-900">
                {(report.weighted_avg.f1_score * 100).toFixed(1)}%
              </td>
              <td className="py-0.5 text-right text-slate-500">
                {report.weighted_avg.support}
              </td>
            </tr>
          </tfoot>
        </table>
      </div>
    );
  }

  if (legacyMetrics) {
    return (
      <div className="mt-2 overflow-x-auto rounded-lg border border-slate-200 bg-white p-2">
        <table className="w-full text-left text-[10px]">
          <thead>
            <tr className="border-b border-slate-100 text-slate-400 uppercase">
              <th className="pb-1 font-semibold">Class Tier</th>
              <th className="pb-1 font-semibold text-right">Precision</th>
              <th className="pb-1 font-semibold text-right">Recall</th>
              <th className="pb-1 font-semibold text-right">F1-Score</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-50 text-slate-700 font-mono">
            {Object.entries(legacyMetrics).map(([cls, m], idx) => (
              <tr key={idx} className="hover:bg-slate-50/50">
                <td className="py-1 font-sans font-medium capitalize text-slate-900">
                  {cls.replace("_", " ")}
                </td>
                <td className="py-1 text-right">{(m.precision * 100).toFixed(1)}%</td>
                <td className="py-1 text-right">{(m.recall * 100).toFixed(1)}%</td>
                <td className="py-1 text-right font-bold text-slate-900">
                  {(m.f1 * 100).toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  return null;
}

export default function ModelBenchmarkPage() {
  const [data, setData] = useState<ModelBenchmarkResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [switching, setSwitching] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [expandedMatrices, setExpandedMatrices] = useState<Record<string, boolean>>({
    mobilenet_v2: true,
    baseline_cnn: true,
    vit_tiny: true,
    efficientnet_b0: true,
    fraud_mnv2_opt: true,
    fraud_mnv2_balanced: true,
  });

  const toggleMatrix = (id: string) => {
    setExpandedMatrices((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await getModelBenchmarks();
      setData(res);
    } catch (err: unknown) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to load model benchmarks from backend."
      );
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleSwitch = async (task: string, modelId: string, modelName: string) => {
    setSwitching(`${task}-${modelId}`);
    try {
      await setActiveModel(task, modelId);
      if (data) {
        setData({
          ...data,
          active_models: {
            ...data.active_models,
            [task]: modelId,
          },
          severity_models: data.severity_models.map((m) =>
            task === "severity" ? { ...m, is_active: m.id === modelId } : m
          ),
          location_models: data.location_models.map((m) =>
            task === "location" ? { ...m, is_active: m.id === modelId } : m
          ),
        });
      }
      setToast(`Active ${task} model switched to ${modelName}`);
      setTimeout(() => setToast(null), 3500);
    } catch (err: unknown) {
      setError(
        err instanceof Error ? err.message : "Failed to switch active model."
      );
    } finally {
      setSwitching(null);
    }
  };

  if (isLoading) {
    return (
      <div className="flex h-96 items-center justify-center">
        <Spinner size="lg" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="mx-auto max-w-4xl px-4 py-12 space-y-4">
        <Alert variant="danger" title="Error Loading Benchmarks">
          {error || "Unable to fetch model benchmarks."} Ensure the FastAPI backend is running on port 8000.
        </Alert>
        <Button variant="outline" size="sm" onClick={fetchData}>
          Retry Connection
        </Button>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-10">
      {/* Top Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-b border-slate-200 pb-5">
        <div>
          <div className="flex items-center gap-2 text-xs font-medium text-slate-500 mb-1">
            <Link href="/" className="hover:text-indigo-600">
              ClaimVision AI
            </Link>
            <span>/</span>
            <Link href="/reviewer/dashboard" className="hover:text-indigo-600">
              Reviewer Workspace
            </Link>
            <span>/</span>
            <span className="text-slate-900">Model Benchmarks</span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 flex items-center gap-3">
            <span>Model Performance Benchmarks</span>
          </h1>
          <p className="mt-1 text-xs text-slate-500">
            Comparative evaluation across all trained architectures. Includes exact confusion matrices and scikit-learn classification reports from training notebooks.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Link href="/reviewer/dashboard">
            <Button variant="outline" size="sm">
              Analytics Dashboard
            </Button>
          </Link>
          <Link href="/reviewer/queue">
            <Button variant="secondary" size="sm">
              Review Queue
            </Button>
          </Link>
        </div>
      </div>

      {toast && (
        <div className="fixed bottom-5 right-5 z-50 rounded-xl bg-slate-900 px-4 py-3 text-xs font-semibold text-white shadow-xl animate-fade-in flex items-center gap-2 border border-slate-700">
          <span>✓</span>
          <span>{toast}</span>
        </div>
      )}

      {/* 1. Damage Severity Models Comparison */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <span>💥 1. Damage Severity Classification (3 Architectures)</span>
            </h2>
            <p className="text-xs text-slate-500">
              Evaluated on the Car Damage Severity test set (248 untouched images, 3 classes: Minor, Moderate, Severe).
            </p>
          </div>
          <span className="text-xs text-slate-500 font-mono">
            Active: <strong className="text-indigo-700 capitalize">{data.active_models.severity?.replace("_", " ")}</strong>
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {data.severity_models.map((model: ModelBenchmarkItem) => {
            const isMatrixOpen = expandedMatrices[model.id] !== false;
            return (
              <div
                key={model.id}
                className={`rounded-2xl border p-5 transition-all relative flex flex-col justify-between ${
                  model.is_winner
                    ? "border-emerald-500 bg-emerald-50/15 shadow-md ring-2 ring-emerald-500/20"
                    : model.is_active
                    ? "border-indigo-600 bg-indigo-50/20 shadow-sm"
                    : "border-slate-200 bg-white shadow-xs hover:border-slate-300"
                }`}
              >
                <div>
                  {/* Winner / Active Badges */}
                  <div className="flex items-center gap-1.5 absolute -top-2.5 right-4">
                    {model.is_winner && (
                      <span className="rounded-full bg-emerald-600 px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-white shadow-xs">
                        🏆 Best Model
                      </span>
                    )}
                    {model.is_active && (
                      <span className="rounded-full bg-indigo-600 px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-white shadow-xs">
                        Active in Pipeline
                      </span>
                    )}
                  </div>

                  <div className="space-y-1 mb-4">
                    <h3 className="text-sm font-bold text-slate-900">{model.name}</h3>
                    <p className="text-[11px] font-mono text-slate-500">{model.architecture}</p>
                    <div className="text-[10px] font-mono text-slate-400 truncate">
                      {model.weights_file}
                    </div>
                  </div>

                  {/* KPI Grid */}
                  <div className="grid grid-cols-2 gap-2 text-xs py-3 border-y border-slate-100 font-mono">
                    <div>
                      <span className="text-[10px] text-slate-400 block font-sans">Accuracy</span>
                      <span className="font-bold text-slate-900">{(model.accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-400 block font-sans">Macro F1</span>
                      <span className="font-bold text-indigo-700">{(model.macro_f1 * 100).toFixed(1)}%</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-400 block font-sans">CPU Latency</span>
                      <span className="font-bold text-slate-900">{model.latency_ms.toFixed(1)} ms</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-400 block font-sans">Model Size</span>
                      <span className="font-bold text-slate-900">{model.size_mb.toFixed(1)} MB</span>
                    </div>
                  </div>

                  {/* Winner / Alternative Explanation */}
                  {model.is_winner && model.winner_reason && (
                    <div className="rounded-xl border border-emerald-200 bg-emerald-50/80 p-3 my-3 text-xs text-emerald-950">
                      <span className="font-bold flex items-center gap-1 text-emerald-800 uppercase tracking-wide text-[10px] mb-1">
                        <span>🏆</span> Why this model is best
                      </span>
                      <p className="leading-relaxed text-[11px] text-emerald-900">
                        {model.winner_reason}
                      </p>
                    </div>
                  )}

                  {!model.is_winner && model.winner_reason && (
                    <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 my-3 text-xs text-slate-600">
                      <span className="font-bold uppercase tracking-wide text-[10px] text-slate-500 mb-1 block">
                        Evaluation Finding
                      </span>
                      <p className="leading-relaxed text-[11px]">
                        {model.winner_reason}
                      </p>
                    </div>
                  )}

                  {/* Highlights */}
                  <div className="my-3 space-y-1">
                    {model.highlights.map((h, idx) => (
                      <div key={idx} className="flex items-center gap-1.5 text-[11px] text-slate-600">
                        <span className="text-emerald-500 font-bold">✓</span>
                        <span>{h}</span>
                      </div>
                    ))}
                  </div>

                  {/* Confusion Matrix & Classification Report */}
                  <div className="pt-2 border-t border-slate-100">
                    <button
                      onClick={() => toggleMatrix(model.id)}
                      className="flex w-full items-center justify-between text-left text-xs font-semibold text-slate-700 hover:text-indigo-600 py-1"
                    >
                      <span>{isMatrixOpen ? "Hide Metrics Breakdown" : "View Confusion Matrix & Classification Report"}</span>
                      <span>{isMatrixOpen ? "▲" : "▼"}</span>
                    </button>

                    {isMatrixOpen && (
                      <div className="mt-2 space-y-2">
                        {model.confusion_matrix && model.confusion_matrix.length > 0 && (
                          <ConfusionMatrixTable
                            labels={model.class_labels || ["Minor", "Moderate", "Severe"]}
                            matrix={model.confusion_matrix}
                            isWinner={model.is_winner}
                          />
                        )}

                        <ClassificationReportTable
                          report={model.classification_report}
                          legacyMetrics={model.per_class_metrics}
                        />
                      </div>
                    )}
                  </div>
                </div>

                <Button
                  variant={model.is_active ? "secondary" : "outline"}
                  size="sm"
                  className="w-full mt-4"
                  disabled={model.is_active || switching === `severity-${model.id}`}
                  isLoading={switching === `severity-${model.id}`}
                  onClick={() => handleSwitch("severity", model.id, model.name)}
                >
                  {model.is_active ? "Active Model" : "Select for Pipeline"}
                </Button>
              </div>
            );
          })}
        </div>
      </div>

      {/* 2. Damaged Part Location Models Comparison */}
      <div className="space-y-4 pt-4 border-t border-slate-200">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <span>📍 2. Damaged Part Location Classification (2 Architectures)</span>
            </h2>
            <p className="text-xs text-slate-500">
              Evaluated on vehicle components (Headlamp, Front Bumper, Hood, Door, Rear Bumper).
            </p>
          </div>
          <span className="text-xs text-slate-500 font-mono">
            Active: <strong className="text-purple-700 capitalize">{data.active_models.location?.replace("_", " ")}</strong>
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {data.location_models.map((model: ModelBenchmarkItem) => {
            const isMatrixOpen = expandedMatrices[model.id] !== false;
            return (
              <div
                key={model.id}
                className={`rounded-2xl border p-5 transition-all relative flex flex-col justify-between ${
                  model.is_winner
                    ? "border-purple-500 bg-purple-50/15 shadow-md ring-2 ring-purple-500/20"
                    : model.is_active
                    ? "border-purple-600 bg-purple-50/20 shadow-sm"
                    : "border-slate-200 bg-white shadow-xs hover:border-slate-300"
                }`}
              >
                <div>
                  <div className="flex items-center gap-1.5 absolute -top-2.5 right-4">
                    {model.is_winner && (
                      <span className="rounded-full bg-purple-600 px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-white shadow-xs">
                        🏆 Best Model
                      </span>
                    )}
                    {model.is_active && (
                      <span className="rounded-full bg-slate-900 px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-white shadow-xs">
                        Active in Pipeline
                      </span>
                    )}
                  </div>

                  <div className="space-y-1 mb-4">
                    <h3 className="text-sm font-bold text-slate-900">{model.name}</h3>
                    <p className="text-[11px] font-mono text-slate-500">{model.architecture}</p>
                    <div className="text-[10px] font-mono text-slate-400 truncate">
                      {model.weights_file}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs py-3 border-y border-slate-100 font-mono">
                    <div>
                      <span className="text-[10px] text-slate-400 block font-sans">Accuracy</span>
                      <span className="font-bold text-slate-900">{(model.accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-400 block font-sans">Macro F1</span>
                      <span className="font-bold text-purple-700">{(model.macro_f1 * 100).toFixed(1)}%</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-400 block font-sans">CPU Latency</span>
                      <span className="font-bold text-slate-900">{model.latency_ms.toFixed(1)} ms</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-400 block font-sans">Model Size</span>
                      <span className="font-bold text-slate-900">{model.size_mb.toFixed(1)} MB</span>
                    </div>
                  </div>

                  {model.is_winner && model.winner_reason && (
                    <div className="rounded-xl border border-purple-200 bg-purple-50/80 p-3 my-3 text-xs text-purple-950">
                      <span className="font-bold flex items-center gap-1 text-purple-800 uppercase tracking-wide text-[10px] mb-1">
                        <span>🏆</span> Why this model is best
                      </span>
                      <p className="leading-relaxed text-[11px] text-purple-900">
                        {model.winner_reason}
                      </p>
                    </div>
                  )}

                  {!model.is_winner && model.winner_reason && (
                    <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 my-3 text-xs text-slate-600">
                      <span className="font-bold uppercase tracking-wide text-[10px] text-slate-500 mb-1 block">
                        Evaluation Finding
                      </span>
                      <p className="leading-relaxed text-[11px]">
                        {model.winner_reason}
                      </p>
                    </div>
                  )}

                  <div className="my-3 space-y-1">
                    {model.highlights.map((h, idx) => (
                      <div key={idx} className="flex items-center gap-1.5 text-[11px] text-slate-600">
                        <span className="text-purple-500 font-bold">✓</span>
                        <span>{h}</span>
                      </div>
                    ))}
                  </div>

                  {/* Confusion Matrix & Classification Report */}
                  <div className="pt-2 border-t border-slate-100">
                    <button
                      onClick={() => toggleMatrix(model.id)}
                      className="flex w-full items-center justify-between text-left text-xs font-semibold text-slate-700 hover:text-purple-600 py-1"
                    >
                      <span>{isMatrixOpen ? "Hide Metrics Breakdown" : "View Confusion Matrix & Classification Report"}</span>
                      <span>{isMatrixOpen ? "▲" : "▼"}</span>
                    </button>

                    {isMatrixOpen && (
                      <div className="mt-2 space-y-2">
                        {model.confusion_matrix && model.confusion_matrix.length > 0 && (
                          <ConfusionMatrixTable
                            labels={model.class_labels || ["headlamp", "front_bumper", "hood", "door", "rear_bumper"]}
                            matrix={model.confusion_matrix}
                            isWinner={model.is_winner}
                          />
                        )}

                        <ClassificationReportTable
                          report={model.classification_report}
                          legacyMetrics={model.per_class_metrics}
                        />
                      </div>
                    )}
                  </div>
                </div>

                <Button
                  variant={model.is_active ? "secondary" : "outline"}
                  size="sm"
                  className="w-full mt-4"
                  disabled={model.is_active || switching === `location-${model.id}`}
                  isLoading={switching === `location-${model.id}`}
                  onClick={() => handleSwitch("location", model.id, model.name)}
                >
                  {model.is_active ? "Active Model" : "Select for Pipeline"}
                </Button>
              </div>
            );
          })}
        </div>
      </div>

      {/* 3. Fraud Authenticity Screening Models */}
      <div className="space-y-4 pt-4 border-t border-slate-200">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <span>🛡️ 3. Fraud Authenticity Screening Models (2 Configurations)</span>
            </h2>
            <p className="text-xs text-slate-500">
              Evaluated on 1,214 held-out test claims (1,143 genuine claims, 71 suspicious fraud cases).
            </p>
          </div>
          <span className="text-xs text-slate-500 font-mono">
            Operating Mode: <strong className="text-rose-700">Mode 1: Max-F1 Target (th = 0.52)</strong>
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {data.fraud_models.map((model: ModelBenchmarkItem) => {
            const isMatrixOpen = expandedMatrices[model.id] !== false;
            return (
              <div
                key={model.id}
                className={`rounded-2xl border p-5 transition-all relative flex flex-col justify-between ${
                  model.is_winner
                    ? "border-rose-500 bg-rose-50/15 shadow-md ring-2 ring-rose-500/20"
                    : "border-slate-200 bg-white shadow-xs hover:border-slate-300"
                }`}
              >
                <div>
                  <div className="flex items-center gap-1.5 absolute -top-2.5 right-4">
                    {model.is_winner && (
                      <span className="rounded-full bg-rose-600 px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-white shadow-xs">
                        🏆 Best Model
                      </span>
                    )}
                    {model.is_active && !model.is_winner && (
                      <span className="rounded-full bg-slate-700 px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-white shadow-xs">
                        Baseline Comparison
                      </span>
                    )}
                  </div>

                  <div className="space-y-1 mb-4">
                    <h3 className="text-sm font-bold text-slate-900">{model.name}</h3>
                    <p className="text-[11px] font-mono text-slate-500">{model.architecture}</p>
                    <div className="text-[10px] font-mono text-slate-400 truncate">
                      {model.weights_file}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs py-3 border-y border-slate-100 font-mono">
                    <div>
                      <span className="text-[10px] text-slate-400 block font-sans">Accuracy</span>
                      <span className="font-bold text-slate-900">{(model.accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-400 block font-sans">Macro F1</span>
                      <span className="font-bold text-rose-700">{(model.macro_f1 * 100).toFixed(1)}%</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-400 block font-sans">Inference Latency</span>
                      <span className="font-bold text-slate-900">{model.latency_ms.toFixed(1)} ms</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-400 block font-sans">Model Footprint</span>
                      <span className="font-bold text-slate-900">{model.size_mb.toFixed(1)} MB</span>
                    </div>
                  </div>

                  {model.is_winner && model.winner_reason && (
                    <div className="rounded-xl border border-rose-200 bg-rose-50/80 p-3 my-3 text-xs text-rose-950">
                      <span className="font-bold flex items-center gap-1 text-rose-800 uppercase tracking-wide text-[10px] mb-1">
                        <span>🏆</span> Why this model is best
                      </span>
                      <p className="leading-relaxed text-[11px] text-rose-900">
                        {model.winner_reason}
                      </p>
                    </div>
                  )}

                  {!model.is_winner && model.winner_reason && (
                    <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 my-3 text-xs text-slate-600">
                      <span className="font-bold uppercase tracking-wide text-[10px] text-slate-500 mb-1 block">
                        Evaluation Finding
                      </span>
                      <p className="leading-relaxed text-[11px]">
                        {model.winner_reason}
                      </p>
                    </div>
                  )}

                  <div className="my-3 space-y-1">
                    {model.highlights.map((h, idx) => (
                      <div key={idx} className="flex items-center gap-1.5 text-[11px] text-slate-600">
                        <span className="text-rose-500 font-bold">✓</span>
                        <span>{h}</span>
                      </div>
                    ))}
                  </div>

                  {/* Confusion Matrix & Classification Report */}
                  <div className="pt-2 border-t border-slate-100">
                    <button
                      onClick={() => toggleMatrix(model.id)}
                      className="flex w-full items-center justify-between text-left text-xs font-semibold text-slate-700 hover:text-rose-600 py-1"
                    >
                      <span>{isMatrixOpen ? "Hide Metrics Breakdown" : "View Confusion Matrix & Classification Report"}</span>
                      <span>{isMatrixOpen ? "▲" : "▼"}</span>
                    </button>

                    {isMatrixOpen && (
                      <div className="mt-2 space-y-2">
                        {model.confusion_matrix && model.confusion_matrix.length > 0 && (
                          <ConfusionMatrixTable
                            labels={model.class_labels || ["Genuine", "Suspicious"]}
                            matrix={model.confusion_matrix}
                            isWinner={model.is_winner}
                          />
                        )}

                        <ClassificationReportTable
                          report={model.classification_report}
                          legacyMetrics={model.per_class_metrics}
                        />
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 4. YOLO Detection Models */}
      <div className="space-y-4 pt-4 border-t border-slate-200">
        <div>
          <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <span>🔍 4. YOLOv8 Damage & Part Detectors</span>
          </h2>
          <p className="text-xs text-slate-500">
            Real-time bounding box detection trained on converted COCO annotations.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {data.detection_models.map((model: ModelBenchmarkItem) => (
            <div
              key={model.id}
              className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs space-y-3"
            >
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-bold text-slate-900">{model.name}</h3>
                  <p className="text-[11px] font-mono text-slate-500">{model.architecture}</p>
                </div>
                <span className="rounded-full bg-indigo-600 px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-white">
                  Real-time Vision
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs py-3 border-y border-slate-100 font-mono">
                <div>
                  <span className="text-[10px] text-slate-400 block font-sans">mAP50</span>
                  <span className="font-bold text-indigo-700">{(model.macro_f1).toFixed(3)}</span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-400 block font-sans">Precision</span>
                  <span className="font-bold text-slate-900">{(model.accuracy * 100).toFixed(1)}%</span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-400 block font-sans">Inference Latency</span>
                  <span className="font-bold text-slate-900">{model.latency_ms.toFixed(1)} ms</span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-400 block font-sans">Model Size</span>
                  <span className="font-bold text-slate-900">{model.size_mb.toFixed(1)} MB</span>
                </div>
              </div>

              <div className="space-y-1">
                {model.highlights.map((h, idx) => (
                  <div key={idx} className="flex items-center gap-1.5 text-[11px] text-slate-600">
                    <span className="text-indigo-500 font-bold">✓</span>
                    <span>{h}</span>
                  </div>
                ))}
              </div>

              {model.winner_reason && (
                <div className="rounded-xl border border-slate-200 bg-slate-50/80 p-3 text-xs text-slate-600 space-y-1">
                  <span className="font-bold text-[10px] uppercase tracking-wider text-slate-500 block">
                    Model Purpose
                  </span>
                  <p className="text-[11px] leading-relaxed">
                    {model.winner_reason}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
