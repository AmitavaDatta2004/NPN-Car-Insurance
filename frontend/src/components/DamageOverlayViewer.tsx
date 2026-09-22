"use client";

import React, { useState } from "react";
import { DetectionItem } from "@/types/claim";
import { getImageUrl } from "@/lib/api";

interface DamageOverlayViewerProps {
  imagePath: string;
  detections: DetectionItem[];
}

export default function DamageOverlayViewer({
  imagePath,
  detections,
}: DamageOverlayViewerProps) {
  const [showBoxes, setShowBoxes] = useState(true);
  const [mode, setMode] = useState<"side-by-side" | "toggle">("side-by-side");
  const [activeTab, setActiveTab] = useState<"original" | "annotated">("annotated");

  const fullImgUrl = getImageUrl(imagePath);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-3">
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-xs font-medium text-slate-700 cursor-pointer">
            <input
              type="checkbox"
              checked={showBoxes}
              onChange={(e) => setShowBoxes(e.target.checked)}
              className="h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
            />
            Show YOLO Damage Bounding Boxes ({detections.length})
          </label>
        </div>

        <div className="flex items-center gap-1 rounded-lg bg-slate-100 p-1 text-xs">
          <button
            onClick={() => setMode("side-by-side")}
            className={`rounded px-2.5 py-1 font-medium transition-all ${
              mode === "side-by-side"
                ? "bg-white text-slate-900 shadow-xs"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            Side-by-Side
          </button>
          <button
            onClick={() => setMode("toggle")}
            className={`rounded px-2.5 py-1 font-medium transition-all ${
              mode === "toggle"
                ? "bg-white text-slate-900 shadow-xs"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            Tab Toggle
          </button>
        </div>
      </div>

      {mode === "side-by-side" ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {/* Original View */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs font-semibold text-slate-500">
              <span>Original Evidence Photo</span>
              <span className="text-[11px] text-slate-400">Untouched</span>
            </div>
            <div className="relative overflow-hidden rounded-xl border border-slate-200 bg-slate-100 shadow-sm aspect-4/3 flex items-center justify-center">
              {fullImgUrl ? (
                <img
                  src={fullImgUrl}
                  alt="Original evidence"
                  className="h-full w-full object-contain"
                />
              ) : (
                <span className="text-xs text-slate-400">No image</span>
              )}
            </div>
          </div>

          {/* Annotated View */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs font-semibold text-indigo-600">
              <span>AI Detection Overlay</span>
              <span className="text-[11px] text-slate-400">YOLOv8n</span>
            </div>
            <div className="relative overflow-hidden rounded-xl border border-slate-200 bg-slate-100 shadow-sm aspect-4/3 flex items-center justify-center">
              {fullImgUrl ? (
                <>
                  <img
                    src={fullImgUrl}
                    alt="Annotated evidence"
                    className="h-full w-full object-contain"
                  />
                  {showBoxes && detections.length > 0 && (
                    <svg className="absolute inset-0 h-full w-full pointer-events-none">
                      {detections.map((det, idx) => {
                        const [cx, cy, w, h] = det.box_normalized;
                        const x = Math.max(0, (cx - w / 2) * 100);
                        const y = Math.max(0, (cy - h / 2) * 100);
                        const width = Math.min(100 - x, w * 100);
                        const height = Math.min(100 - y, h * 100);

                        return (
                          <g key={idx}>
                            <rect
                              x={`${x}%`}
                              y={`${y}%`}
                              width={`${width}%`}
                              height={`${height}%`}
                              fill="rgba(239, 68, 68, 0.15)"
                              stroke="#ef4444"
                              strokeWidth="2.5"
                              strokeDasharray="4 2"
                            />
                            <rect
                              x={`${x}%`}
                              y={`${Math.max(0, y - 4)}%`}
                              width="65"
                              height="16"
                              fill="#ef4444"
                              rx="3"
                            />
                            <text
                              x={`${x + 1}%`}
                              y={`${Math.max(0, y - 4) + 2.8}%`}
                              fill="#ffffff"
                              fontSize="10"
                              fontWeight="bold"
                              fontFamily="sans-serif"
                            >
                              {det.label} {(det.confidence * 100).toFixed(0)}%
                            </text>
                          </g>
                        );
                      })}
                    </svg>
                  )}
                </>
              ) : (
                <span className="text-xs text-slate-400">No image</span>
              )}
            </div>
          </div>
        </div>
      ) : (
        /* Tab Toggle View */
        <div className="space-y-2">
          <div className="flex gap-2">
            <button
              onClick={() => setActiveTab("annotated")}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold ${
                activeTab === "annotated"
                  ? "bg-indigo-600 text-white"
                  : "bg-slate-100 text-slate-600 hover:bg-slate-200"
              }`}
            >
              AI Annotated View
            </button>
            <button
              onClick={() => setActiveTab("original")}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold ${
                activeTab === "original"
                  ? "bg-indigo-600 text-white"
                  : "bg-slate-100 text-slate-600 hover:bg-slate-200"
              }`}
            >
              Original View
            </button>
          </div>

          <div className="relative overflow-hidden rounded-xl border border-slate-200 bg-slate-100 shadow-sm aspect-16/9 flex items-center justify-center max-h-[450px]">
            {fullImgUrl ? (
              <>
                <img
                  src={fullImgUrl}
                  alt="Evidence"
                  className="h-full w-full object-contain"
                />
                {activeTab === "annotated" && showBoxes && detections.length > 0 && (
                  <svg className="absolute inset-0 h-full w-full pointer-events-none">
                    {detections.map((det, idx) => {
                      const [cx, cy, w, h] = det.box_normalized;
                      const x = (cx - w / 2) * 100;
                      const y = (cy - h / 2) * 100;
                      return (
                        <rect
                          key={idx}
                          x={`${x}%`}
                          y={`${y}%`}
                          width={`${w * 100}%`}
                          height={`${h * 100}%`}
                          fill="rgba(239, 68, 68, 0.15)"
                          stroke="#ef4444"
                          strokeWidth="3"
                        />
                      );
                    })}
                  </svg>
                )}
              </>
            ) : (
              <span className="text-xs text-slate-400">No image</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
