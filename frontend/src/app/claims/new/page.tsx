"use client";

import React, { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { createClaim, submitClaim, uploadImage } from "@/lib/api";
import Card from "@/components/ui/Card";

export default function NewClaimPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState("Initiating claim...");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const processFile = async (file: File) => {
    // Basic file validation
    const validTypes = ["image/jpeg", "image/png", "image/webp"];
    if (!validTypes.includes(file.type)) {
      setErrorMsg("Please upload a valid image file (JPG, PNG, or WebP).");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setErrorMsg("Image size exceeds the 10MB limit. Please upload a smaller image.");
      return;
    }

    setErrorMsg(null);
    setIsProcessing(true);

    try {
      // 1. Create claim draft
      setProcessingStatus("Creating claim session...");
      const randomPolicy = `POL-${Math.floor(100000 + Math.random() * 900000)}`;
      const newClaim = await createClaim({
        policy_number: randomPolicy,
        vehicle_make: "Vehicle",
        vehicle_model: "Auto",
        vehicle_year: new Date().getFullYear(),
        vehicle_segment: "sedan",
        incident_description: "Direct vehicle photo damage scan",
        incident_date: new Date().toISOString().slice(0, 10),
      });

      // 2. Upload photo evidence
      setProcessingStatus("Uploading evidence photograph...");
      await uploadImage(newClaim.id, file);

      // 3. Submit claim
      setProcessingStatus("Submitting for assessment...");
      await submitClaim(newClaim.id);

      // 4. Redirect directly to processing/scan view
      router.push(`/claims/${newClaim.id}/processing`);
    } catch (err: unknown) {
      setErrorMsg(
        err instanceof Error
          ? err.message
          : "Failed to upload and initiate claim assessment. Please try again."
      );
      setIsProcessing(false);
    }
  };

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    processFile(files[0]);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (isProcessing) return;
    handleFiles(e.dataTransfer.files);
  };

  return (
    <main className="mx-auto max-w-2xl px-4 py-12 sm:px-6 lg:px-8">
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
          Scan Vehicle Damage
        </h1>
        <p className="mt-2 text-sm text-slate-500">
          Upload a clear photograph of the vehicle to start instant automated assessment.
        </p>
      </div>

      {errorMsg && (
        <div className="mb-6 flex items-start gap-3 rounded-xl border border-rose-200 bg-rose-50 p-4 text-xs text-rose-700">
          <span className="text-base font-bold leading-none">⚠️</span>
          <div className="flex-1">
            <p className="font-semibold">{errorMsg}</p>
          </div>
          <button
            onClick={() => setErrorMsg(null)}
            className="text-rose-500 hover:text-rose-800 font-bold ml-2"
          >
            ✕
          </button>
        </div>
      )}

      <Card className="p-8">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          className="hidden"
          disabled={isProcessing}
          onChange={(e) => handleFiles(e.target.files)}
        />

        {isProcessing ? (
          <div className="flex flex-col items-center justify-center py-16 text-center space-y-4">
            <div className="h-12 w-12 animate-spin rounded-full border-4 border-indigo-600 border-t-transparent" />
            <div>
              <p className="text-base font-bold text-slate-900">{processingStatus}</p>
              <p className="text-xs text-slate-500 mt-1">
                Please wait while we prepare your evidence...
              </p>
            </div>
          </div>
        ) : (
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`flex flex-col items-center justify-center rounded-2xl border-2 border-dashed p-12 text-center cursor-pointer transition-all ${
              isDragging
                ? "border-indigo-600 bg-indigo-50/60 scale-[1.01]"
                : "border-slate-300 hover:border-indigo-400 bg-slate-50/60 hover:bg-indigo-50/20"
            }`}
          >
            <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-indigo-100 text-indigo-600 shadow-xs">
              <svg
                className="h-8 w-8"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
            </div>

            <h3 className="text-base font-bold text-slate-900">
              Click or drag vehicle photograph here
            </h3>
            <p className="mt-1 text-xs text-slate-500 max-w-sm">
              Supports JPG, PNG, and WebP up to 10MB.
            </p>

            <div className="mt-6">
              <span className="inline-flex items-center justify-center rounded-lg bg-indigo-600 px-5 py-2.5 text-xs font-semibold text-white shadow-sm hover:bg-indigo-700 transition-colors">
                Select Photo from Computer
              </span>
            </div>
          </div>
        )}
      </Card>
    </main>
  );
}
