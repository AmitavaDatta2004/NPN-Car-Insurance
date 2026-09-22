"use client";

import React, { useRef, useState } from "react";
import { ImageRecord } from "@/types/claim";
import { getImageUrl, uploadImage } from "@/lib/api";
import Button from "./ui/Button";

interface ImageUploaderProps {
  claimId: string;
  images: ImageRecord[];
  onUploadSuccess: (newImage: ImageRecord) => void;
  onDeleteImage?: (imageId: string) => void;
}

export default function ImageUploader({
  claimId,
  images,
  onUploadSuccess,
  onDeleteImage,
}: ImageUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const file = files[0];

    // Client-side quick checks
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      setErrorMsg("Only JPG, PNG, and WebP images are allowed.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setErrorMsg("Image exceeds maximum allowed size of 10MB.");
      return;
    }

    setErrorMsg(null);
    setIsUploading(true);

    try {
      const uploaded = await uploadImage(claimId, file);
      onUploadSuccess(uploaded);
    } catch (err: unknown) {
      setErrorMsg(
        err instanceof Error ? err.message : "Failed to upload image. Please try again."
      );
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  return (
    <div className="space-y-4">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-8 text-center cursor-pointer transition-colors ${
          isDragging
            ? "border-indigo-500 bg-indigo-50/50"
            : "border-slate-300 hover:border-indigo-400 bg-slate-50/50"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-indigo-100 text-indigo-600">
          <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
        </div>
        <p className="text-sm font-semibold text-slate-900">
          {isUploading ? "Uploading photograph..." : "Click or drag vehicle photograph here"}
        </p>
        <p className="mt-1 text-xs text-slate-500">
          Supports JPG, PNG, WebP up to 10MB. Min resolution 224×224 px.
        </p>
      </div>

      {errorMsg && (
        <div className="rounded-lg bg-rose-50 p-3 text-xs text-rose-700 border border-rose-200">
          ⚠️ {errorMsg}
        </div>
      )}

      {images.length > 0 && (
        <div className="mt-4 space-y-2">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
            Uploaded Evidence ({images.length})
          </h4>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {images.map((img) => (
              <div
                key={img.id}
                className="flex items-center gap-3 rounded-lg border border-slate-200 bg-white p-2.5 shadow-sm"
              >
                <img
                  src={getImageUrl(img.local_path)}
                  alt={img.filename}
                  className="h-16 w-16 rounded-md object-cover border border-slate-100 bg-slate-100"
                />
                <div className="flex-1 min-w-0">
                  <p className="truncate text-xs font-medium text-slate-800">
                    {img.filename}
                  </p>
                  <p className="text-[11px] text-slate-500">
                    {img.width}×{img.height} px • {(img.file_size_bytes / 1024).toFixed(1)} KB
                  </p>
                  <span className="inline-block mt-0.5 text-[10px] text-emerald-600 font-medium">
                    ✓ Verified
                  </span>
                </div>
                {onDeleteImage && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteImage(img.id);
                    }}
                    className="text-slate-400 hover:text-rose-600"
                  >
                    ✕
                  </Button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
