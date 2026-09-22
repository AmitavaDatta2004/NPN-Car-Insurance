"use client";

import React, { useState, useRef, useCallback } from "react";
import { DetectionItem } from "@/types/claim";
import { getImageUrl } from "@/lib/api";

interface DamageOverlayViewerProps {
  imagePath: string;
  detections: DetectionItem[];
}

function parseBox(boxNorm: number[]) {
  if (!boxNorm || boxNorm.length < 4) return { x: 10, y: 10, w: 30, h: 30 };
  const [a, b, c, d] = boxNorm;
  // Box is [cx, cy, w, h] normalized in [0, 1]
  let x = (a - c / 2) * 100;
  let y = (b - d / 2) * 100;
  let w = c * 100;
  let h = d * 100;

  // Safeguard bounds
  x = Math.max(0, Math.min(95, x));
  y = Math.max(0, Math.min(95, y));
  w = Math.max(5, Math.min(100 - x, w));
  h = Math.max(5, Math.min(100 - y, h));
  return { x, y, w, h };
}

export default function DamageOverlayViewer({
  imagePath,
  detections,
}: DamageOverlayViewerProps) {
  const [mode, setMode] = useState<"annotated" | "slider" | "side-by-side" | "original">("annotated");
  const [sliderPos, setSliderPos] = useState<number>(50); // percentage 0 - 100
  const [isDragging, setIsDragging] = useState(false);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [imgError, setImgError] = useState(false);

  const sliderContainerRef = useRef<HTMLDivElement>(null);
  const fullImgUrl = getImageUrl(imagePath);

  const handleMove = useCallback(
    (clientX: number) => {
      if (!sliderContainerRef.current) return;
      const rect = sliderContainerRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
      const percent = (x / rect.width) * 100;
      setSliderPos(percent);
    },
    []
  );

  const handleTouchMove = useCallback(
    (e: React.TouchEvent) => {
      if (e.touches.length > 0) {
        handleMove(e.touches[0].clientX);
      }
    },
    [handleMove]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (isDragging) {
        handleMove(e.clientX);
      }
    },
    [isDragging, handleMove]
  );

  return (
    <div className="space-y-4">
      {/* View Mode Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-3">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
            Evidence Studio
          </span>
          <span className="rounded-full bg-indigo-50 px-2.5 py-0.5 text-[11px] font-mono font-semibold text-indigo-700 border border-indigo-200">
            {detections.length} {detections.length === 1 ? "Damage Box" : "Damage Boxes"}
          </span>
        </div>

        <div className="flex items-center gap-1 rounded-lg bg-slate-100 p-1 text-xs font-medium">
          <button
            onClick={() => setMode("annotated")}
            className={`rounded px-3 py-1 transition-all ${
              mode === "annotated"
                ? "bg-white text-slate-900 shadow-xs font-semibold"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            Annotated (Box)
          </button>
          <button
            onClick={() => setMode("slider")}
            className={`rounded px-3 py-1 transition-all ${
              mode === "slider"
                ? "bg-white text-slate-900 shadow-xs font-semibold"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            Split Slider
          </button>
          <button
            onClick={() => setMode("side-by-side")}
            className={`rounded px-3 py-1 transition-all ${
              mode === "side-by-side"
                ? "bg-white text-slate-900 shadow-xs font-semibold"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            Side-by-Side
          </button>
          <button
            onClick={() => setMode("original")}
            className={`rounded px-3 py-1 transition-all ${
              mode === "original"
                ? "bg-white text-slate-900 shadow-xs font-semibold"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            Original
          </button>
        </div>
      </div>

      {/* Image Error Fallback */}
      {imgError && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-xs text-rose-800 space-y-1">
          <p className="font-semibold">Unable to load evidence image from server.</p>
          <p className="font-mono text-[11px] text-rose-600 truncate">Path: {fullImgUrl || imagePath}</p>
          <button
            onClick={() => setImgError(false)}
            className="mt-2 rounded bg-rose-600 px-3 py-1 text-white font-medium hover:bg-rose-700"
          >
            Retry Loading
          </button>
        </div>
      )}

      {/* 1. Default Mode: Annotated Evidence with YOLO Boxes (Direct & Clear) */}
      {mode === "annotated" && (
        <div className="space-y-3">
          <div className="relative w-full overflow-hidden rounded-xl bg-slate-950 border border-slate-800 shadow-lg flex items-center justify-center p-1">
            {fullImgUrl ? (
              <div className="relative inline-block max-w-full">
                <img
                  src={fullImgUrl}
                  alt="Vehicle Evidence with YOLO Bounding Boxes"
                  onError={() => setImgError(true)}
                  className="block max-h-[500px] w-auto max-w-full object-contain mx-auto rounded-lg"
                />
                <svg
                  className="absolute inset-0 w-full h-full pointer-events-none"
                  viewBox="0 0 100 100"
                  preserveAspectRatio="none"
                >
                  {detections.map((det, idx) => {
                    const { x, y, w, h } = parseBox(det.box_normalized);
                    const isHovered = hoveredIndex === idx;

                    return (
                      <g key={idx} className="transition-all duration-150">
                        {/* Semi-transparent highlight box */}
                        <rect
                          x={`${x}%`}
                          y={`${y}%`}
                          width={`${w}%`}
                          height={`${h}%`}
                          fill={isHovered ? "rgba(239, 68, 68, 0.35)" : "rgba(239, 68, 68, 0.20)"}
                          stroke="#ef4444"
                          strokeWidth={isHovered ? "3" : "2"}
                          rx="2"
                        />
                        {/* Label Badge */}
                        <rect
                          x={`${x}%`}
                          y={`${Math.max(0, y - 5.5)}%`}
                          width="24"
                          height="5.5"
                          fill="#ef4444"
                          rx="1.5"
                        />
                        <text
                          x={`${x + 1}%`}
                          y={`${Math.max(3.8, y - 1.5)}%`}
                          fill="#ffffff"
                          fontSize="3.2"
                          fontWeight="bold"
                          fontFamily="monospace"
                        >
                          {det.label.toUpperCase()} {(det.confidence * 100).toFixed(0)}%
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>
            ) : (
              <div className="flex h-64 items-center justify-center text-xs text-slate-400">
                No image provided
              </div>
            )}
          </div>
          <p className="text-center text-[11px] text-slate-500">
            Bounding boxes outline AI-detected vehicle damage areas and confidence.
          </p>
        </div>
      )}

      {/* 2. Interactive Before/After Split Slider View */}
      {mode === "slider" && (
        <div className="space-y-2">
          <div
            ref={sliderContainerRef}
            onMouseDown={() => setIsDragging(true)}
            onMouseUp={() => setIsDragging(false)}
            onMouseLeave={() => setIsDragging(false)}
            onMouseMove={handleMouseMove}
            onTouchMove={handleTouchMove}
            className="relative overflow-hidden rounded-xl border border-slate-800 bg-slate-950 shadow-lg select-none cursor-ew-resize flex items-center justify-center p-1"
          >
            {fullImgUrl ? (
              <div className="relative inline-block max-w-full">
                {/* Background Layer: Annotated Image with YOLO Bounding Boxes */}
                <img
                  src={fullImgUrl}
                  alt="Annotated evidence"
                  className="block max-h-[500px] w-auto max-w-full object-contain mx-auto rounded-lg"
                />
                <svg
                  className="absolute inset-0 w-full h-full pointer-events-none"
                  viewBox="0 0 100 100"
                  preserveAspectRatio="none"
                >
                  {detections.map((det, idx) => {
                    const { x, y, w, h } = parseBox(det.box_normalized);
                    const isHovered = hoveredIndex === idx;

                    return (
                      <g key={idx}>
                        <rect
                          x={`${x}%`}
                          y={`${y}%`}
                          width={`${w}%`}
                          height={`${h}%`}
                          fill={isHovered ? "rgba(239, 68, 68, 0.35)" : "rgba(239, 68, 68, 0.20)"}
                          stroke="#ef4444"
                          strokeWidth={isHovered ? "3" : "2"}
                          rx="2"
                        />
                        <rect
                          x={`${x}%`}
                          y={`${Math.max(0, y - 5.5)}%`}
                          width="24"
                          height="5.5"
                          fill="#ef4444"
                          rx="1.5"
                        />
                        <text
                          x={`${x + 1}%`}
                          y={`${Math.max(3.8, y - 1.5)}%`}
                          fill="#ffffff"
                          fontSize="3.2"
                          fontWeight="bold"
                          fontFamily="monospace"
                        >
                          {det.label.toUpperCase()} {(det.confidence * 100).toFixed(0)}%
                        </text>
                      </g>
                    );
                  })}
                </svg>

                {/* Foreground Layer: Clipped Original Image */}
                <div
                  className="absolute inset-0 overflow-hidden pointer-events-none rounded-lg"
                  style={{ width: `${sliderPos}%` }}
                >
                  <img
                    src={fullImgUrl}
                    alt="Original evidence"
                    className="block max-h-[500px] w-auto max-w-none object-contain"
                    style={{
                      width: sliderContainerRef.current ? `${sliderContainerRef.current.clientWidth}px` : "100%",
                    }}
                  />
                  <span className="absolute bottom-2 left-2 rounded bg-slate-900/80 px-2 py-0.5 text-[10px] font-mono font-bold text-slate-300">
                    Original
                  </span>
                </div>

                {/* Vertical Divider Handle */}
                <div
                  className="absolute top-0 bottom-0 w-0.5 bg-white shadow-2xl z-20 pointer-events-none"
                  style={{ left: `${sliderPos}%` }}
                >
                  <div className="absolute top-1/2 -translate-y-1/2 -left-3.5 h-7 w-7 rounded-full bg-white shadow-xl border-2 border-indigo-600 flex items-center justify-center text-indigo-700 text-xs font-bold select-none">
                    ↔
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex h-64 items-center justify-center text-xs text-slate-400">
                No image provided
              </div>
            )}
          </div>
          <p className="text-center text-[11px] text-slate-400">
            Drag the slider horizontally to compare original evidence against YOLO detection overlays.
          </p>
        </div>
      )}

      {/* 3. Side-by-Side View */}
      {mode === "side-by-side" && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs font-semibold text-slate-500">
              <span>Original Photo</span>
              <span className="text-[10px] text-slate-400 font-mono">Pristine</span>
            </div>
            <div className="relative overflow-hidden rounded-xl border border-slate-800 bg-slate-950 shadow-sm flex items-center justify-center p-1">
              {fullImgUrl ? (
                <img
                  src={fullImgUrl}
                  alt="Original"
                  className="block max-h-[380px] w-auto max-w-full object-contain mx-auto rounded-lg"
                />
              ) : (
                <span className="text-xs text-slate-400">No image</span>
              )}
            </div>
          </div>

          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs font-semibold text-indigo-600">
              <span>YOLOv8 Detection Overlay</span>
              <span className="text-[10px] text-indigo-400 font-mono">AI Bounding Boxes</span>
            </div>
            <div className="relative overflow-hidden rounded-xl border border-slate-800 bg-slate-950 shadow-sm flex items-center justify-center p-1">
              {fullImgUrl ? (
                <div className="relative inline-block max-w-full">
                  <img
                    src={fullImgUrl}
                    alt="Annotated"
                    className="block max-h-[380px] w-auto max-w-full object-contain mx-auto rounded-lg"
                  />
                  <svg
                    className="absolute inset-0 w-full h-full pointer-events-none"
                    viewBox="0 0 100 100"
                    preserveAspectRatio="none"
                  >
                    {detections.map((det, idx) => {
                      const { x, y, w, h } = parseBox(det.box_normalized);
                      const isHovered = hoveredIndex === idx;

                      return (
                        <g key={idx}>
                          <rect
                            x={`${x}%`}
                            y={`${y}%`}
                            width={`${w}%`}
                            height={`${h}%`}
                            fill={isHovered ? "rgba(239, 68, 68, 0.35)" : "rgba(239, 68, 68, 0.20)"}
                            stroke="#ef4444"
                            strokeWidth="2"
                            rx="2"
                          />
                          <rect
                            x={`${x}%`}
                            y={`${Math.max(0, y - 5.5)}%`}
                            width="24"
                            height="5.5"
                            fill="#ef4444"
                            rx="1.5"
                          />
                          <text
                            x={`${x + 1}%`}
                            y={`${Math.max(3.8, y - 1.5)}%`}
                            fill="#ffffff"
                            fontSize="3.2"
                            fontWeight="bold"
                            fontFamily="monospace"
                          >
                            {det.label.toUpperCase()} {(det.confidence * 100).toFixed(0)}%
                          </text>
                        </g>
                      );
                    })}
                  </svg>
                </div>
              ) : (
                <span className="text-xs text-slate-400">No image</span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 4. Original Only View */}
      {mode === "original" && (
        <div className="relative w-full overflow-hidden rounded-xl bg-slate-950 border border-slate-800 shadow-lg flex items-center justify-center p-1">
          {fullImgUrl ? (
            <img
              src={fullImgUrl}
              alt="Pristine Evidence"
              className="block max-h-[500px] w-auto max-w-full object-contain mx-auto rounded-lg"
            />
          ) : (
            <div className="flex h-64 items-center justify-center text-xs text-slate-400">
              No image provided
            </div>
          )}
        </div>
      )}

      {/* Detections List Chips */}
      {detections.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Detected Damage Regions ({detections.length})
            </span>
          </div>
          <div className="flex flex-wrap gap-2">
            {detections.map((det, idx) => (
              <div
                key={idx}
                onMouseEnter={() => setHoveredIndex(idx)}
                onMouseLeave={() => setHoveredIndex(null)}
                className={`flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs transition-all cursor-pointer ${
                  hoveredIndex === idx
                    ? "border-rose-400 bg-rose-50 text-rose-900 shadow-xs"
                    : "border-slate-200 bg-white text-slate-700 hover:border-slate-300"
                }`}
              >
                <span className="h-2 w-2 rounded-full bg-rose-500" />
                <span className="font-semibold capitalize">{det.label}</span>
                <span className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[10px] text-slate-600">
                  {(det.confidence * 100).toFixed(0)}% conf
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
