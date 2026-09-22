"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { useForm } from "react-hook-form";
import * as z from "zod";
import { Claim, ImageRecord } from "@/types/claim";
import { createClaim, deleteImage, getImageUrl, submitClaim } from "@/lib/api";
import Button from "@/components/ui/Button";
import Card from "@/components/ui/Card";
import ImageUploader from "@/components/ImageUploader";
import StepIndicator from "@/components/StepIndicator";

const claimSchema = z.object({
  policy_number: z.string().min(3, "Policy number must be at least 3 characters"),
  vehicle_make: z.string().min(2, "Vehicle make is required"),
  vehicle_model: z.string().min(1, "Vehicle model is required"),
  vehicle_year: z.coerce.number().min(1990).max(2030),
  vehicle_segment: z.string().min(1, "Please select vehicle segment"),
  incident_description: z.string().min(5, "Please describe what happened (min 5 characters)"),
  incident_date: z.string().min(1, "Please select incident date"),
});

type ClaimFormData = z.infer<typeof claimSchema>;

export default function NewClaimWizardPage() {
  const router = useRouter();
  const [currentStep, setCurrentStep] = useState(1);
  const [claim, setClaim] = useState<Claim | null>(null);
  const [images, setImages] = useState<ImageRecord[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});
  const [declarationAccepted, setDeclarationAccepted] = useState(false);

  const {
    register,
    handleSubmit,
    getValues,
  } = useForm<ClaimFormData>({
    defaultValues: {
      policy_number: "POL-987452",
      vehicle_make: "Hyundai",
      vehicle_model: "Creta",
      vehicle_year: 2022,
      vehicle_segment: "suv",
      incident_description: "Minor bumper collision while reversing out of parking space.",
      incident_date: new Date().toISOString().slice(0, 10),
    },
  });

  // Step 1: Create draft claim on backend with Zod validation
  const onStep1Submit = async (data: ClaimFormData) => {
    const parseResult = claimSchema.safeParse(data);
    if (!parseResult.success) {
      const errMap: Record<string, string> = {};
      for (const issue of parseResult.error.issues) {
        if (issue.path[0]) {
          errMap[String(issue.path[0])] = issue.message;
        }
      }
      setFormErrors(errMap);
      return;
    }

    setFormErrors({});
    setIsSubmitting(true);
    setErrorMsg(null);
    try {
      const newClaim = await createClaim(parseResult.data);
      setClaim(newClaim);
      setCurrentStep(2);
    } catch (err: unknown) {
      setErrorMsg(err instanceof Error ? err.message : "Failed to create claim draft.");
    } finally {
      setIsSubmitting(false);
    }
  };

  // Step 2: Handle image uploads
  const handleUploadSuccess = (newImg: ImageRecord) => {
    setImages((prev) => [...prev, newImg]);
  };

  const handleDeleteImage = async (imgId: string) => {
    try {
      await deleteImage(imgId);
      setImages((prev) => prev.filter((i) => i.id !== imgId));
    } catch (err: unknown) {
      setErrorMsg(err instanceof Error ? err.message : "Failed to remove image.");
    }
  };

  const onStep2Next = () => {
    if (images.length === 0) {
      setErrorMsg("Please upload at least one vehicle damage photograph before proceeding.");
      return;
    }
    setErrorMsg(null);
    setCurrentStep(3);
  };

  // Step 3: Final submit
  const onFinalSubmit = async () => {
    if (!claim) return;
    if (!declarationAccepted) {
      setErrorMsg("Please confirm the declaration before submitting.");
      return;
    }
    setIsSubmitting(true);
    setErrorMsg(null);
    try {
      await submitClaim(claim.id);
      router.push(`/claims/${claim.id}/processing`);
    } catch (err: unknown) {
      setErrorMsg(err instanceof Error ? err.message : "Failed to submit claim.");
      setIsSubmitting(false);
    }
  };

  return (
    <main className="mx-auto max-w-3xl px-4 py-8 sm:px-6 lg:px-8">
      <div className="mb-6 text-center">
        <h1 className="text-3xl font-extrabold text-slate-900">File an Auto Claim</h1>
        <p className="mt-1 text-sm text-slate-500">
          Fast-track your claim assessment in 3 simple steps.
        </p>
      </div>

      <StepIndicator currentStep={currentStep} />

      {errorMsg && (
        <div className="mb-6 rounded-xl border border-rose-200 bg-rose-50 p-4 text-xs text-rose-700">
          ⚠️ {errorMsg}
        </div>
      )}

      {/* Step 1: Vehicle & Incident Form */}
      {currentStep === 1 && (
        <Card>
          <h2 className="text-lg font-bold text-slate-900 mb-4">
            Step 1: Vehicle & Incident Details
          </h2>
          <form onSubmit={handleSubmit(onStep1Submit)} className="space-y-4">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Policy Number *
                </label>
                <input
                  {...register("policy_number")}
                  className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                  placeholder="e.g. POL-123456"
                />
                {formErrors.policy_number && (
                  <p className="mt-1 text-xs text-rose-600">{formErrors.policy_number}</p>
                )}
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Vehicle Segment *
                </label>
                <select
                  {...register("vehicle_segment")}
                  className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 bg-white"
                >
                  <option value="compact">Compact / Hatchback</option>
                  <option value="sedan">Sedan</option>
                  <option value="suv">SUV / MUV</option>
                  <option value="luxury">Luxury</option>
                </select>
                {formErrors.vehicle_segment && (
                  <p className="mt-1 text-xs text-rose-600">{formErrors.vehicle_segment}</p>
                )}
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Vehicle Make *
                </label>
                <input
                  {...register("vehicle_make")}
                  className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                  placeholder="e.g. Hyundai, Honda, Toyota"
                />
                {formErrors.vehicle_make && (
                  <p className="mt-1 text-xs text-rose-600">{formErrors.vehicle_make}</p>
                )}
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Vehicle Model *
                </label>
                <input
                  {...register("vehicle_model")}
                  className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                  placeholder="e.g. Creta, Civic, Corolla"
                />
                {formErrors.vehicle_model && (
                  <p className="mt-1 text-xs text-rose-600">{formErrors.vehicle_model}</p>
                )}
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Model Year *
                </label>
                <input
                  type="number"
                  {...register("vehicle_year")}
                  className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                />
                {formErrors.vehicle_year && (
                  <p className="mt-1 text-xs text-rose-600">{formErrors.vehicle_year}</p>
                )}
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Incident Date *
                </label>
                <input
                  type="date"
                  {...register("incident_date")}
                  className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                />
                {formErrors.incident_date && (
                  <p className="mt-1 text-xs text-rose-600">{formErrors.incident_date}</p>
                )}
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">
                Incident Description *
              </label>
              <textarea
                rows={3}
                {...register("incident_description")}
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                placeholder="Briefly describe how and where the vehicle damage occurred..."
              />
              {formErrors.incident_description && (
                <p className="mt-1 text-xs text-rose-600">{formErrors.incident_description}</p>
              )}
            </div>


            <div className="pt-4 flex justify-end">
              <Button type="submit" isLoading={isSubmitting}>
                Save & Continue to Uploads →
              </Button>
            </div>
          </form>
        </Card>
      )}

      {/* Step 2: Evidence Image Upload */}
      {currentStep === 2 && claim && (
        <Card>
          <h2 className="text-lg font-bold text-slate-900 mb-1">
            Step 2: Upload Vehicle Damage Evidence
          </h2>
          <p className="text-xs text-slate-500 mb-4">
            Upload clear, well-lit photographs of the damaged area. OpenCV will automatically verify resolution and blur.
          </p>

          <ImageUploader
            claimId={claim.id}
            images={images}
            onUploadSuccess={handleUploadSuccess}
            onDeleteImage={handleDeleteImage}
          />

          <div className="mt-6 pt-4 border-t border-slate-100 flex items-center justify-between">
            <Button variant="secondary" onClick={() => setCurrentStep(1)}>
              ← Back to Details
            </Button>
            <Button onClick={onStep2Next} disabled={images.length === 0}>
              Review & Submit ({images.length} photo{images.length !== 1 ? "s" : ""}) →
            </Button>
          </div>
        </Card>
      )}

      {/* Step 3: Review & Submit */}
      {currentStep === 3 && claim && (
        <Card className="space-y-6">
          <div>
            <h2 className="text-lg font-bold text-slate-900 mb-1">
              Step 3: Review & Confirm Declaration
            </h2>
            <p className="text-xs text-slate-500">
              Please inspect the details below before submitting for automated AI triage.
            </p>
          </div>

          <div className="rounded-lg bg-slate-50 p-4 space-y-3 text-xs">
            <div className="grid grid-cols-2 gap-2">
              <div>
                <span className="text-slate-400">Policy:</span>{" "}
                <span className="font-semibold text-slate-800">{claim.policy_number}</span>
              </div>
              <div>
                <span className="text-slate-400">Vehicle:</span>{" "}
                <span className="font-semibold text-slate-800">
                  {claim.vehicle_year} {claim.vehicle_make} {claim.vehicle_model}
                </span>
              </div>
              <div>
                <span className="text-slate-400">Segment:</span>{" "}
                <span className="font-semibold text-slate-800 capitalize">
                  {claim.vehicle_segment}
                </span>
              </div>
              <div>
                <span className="text-slate-400">Incident Date:</span>{" "}
                <span className="font-semibold text-slate-800">{claim.incident_date}</span>
              </div>
            </div>
            <div>
              <span className="text-slate-400">Description:</span>
              <p className="mt-1 text-slate-700 italic font-normal">
                &ldquo;{claim.incident_description}&rdquo;
              </p>
            </div>
          </div>

          <div>
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">
              Attached Photographs ({images.length})
            </h4>
            <div className="flex flex-wrap gap-3">
              {images.map((img) => (
                <img
                  key={img.id}
                  src={getImageUrl(img.local_path)}
                  alt={img.filename}
                  className="h-20 w-20 rounded-lg object-cover border border-slate-200"
                />
              ))}
            </div>
          </div>

          <div className="rounded-lg border border-slate-200 p-4">
            <label className="flex items-start gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={declarationAccepted}
                onChange={(e) => setDeclarationAccepted(e.target.checked)}
                className="mt-0.5 h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
              />
              <span className="text-xs text-slate-600 leading-relaxed">
                I declare that the photographs and accident details provided above are authentic,
                accurate, and represent the actual damage incurred. I understand that fraudulent claims
                are subject to legal prosecution and policy cancellation.
              </span>
            </label>
          </div>

          <div className="pt-2 flex items-center justify-between">
            <Button variant="secondary" onClick={() => setCurrentStep(2)}>
              ← Back to Uploads
            </Button>
            <Button
              onClick={onFinalSubmit}
              disabled={!declarationAccepted}
              isLoading={isSubmitting}
            >
              Submit Claim for AI Assessment 🚀
            </Button>
          </div>
        </Card>
      )}
    </main>
  );
}
