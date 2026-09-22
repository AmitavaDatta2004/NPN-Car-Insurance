/**
 * api.ts — Centralized HTTP client and formatting utilities for ClaimVision AI.
 */

import {
  AssessmentResult,
  Claim,
  ImageRecord,
  ReviewCorrectionPayload,
  ReviewDecisionPayload,
  DashboardSummary,
  ModelBenchmarkResponse,
  TimelineEvent,
  TriageRoute,
} from "@/types/claim";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000/api/v1";
const BACKEND_BASE_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public details?: unknown
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  const headers = new Headers(options.headers || {});

  if (!(options.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(url, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let errorMsg = `HTTP ${response.status} ${response.statusText}`;
    try {
      const errData = await response.json();
      if (errData && errData.detail) {
        errorMsg =
          typeof errData.detail === "string"
            ? errData.detail
            : JSON.stringify(errData.detail);
      }
    } catch {
      // Use fallback error message
    }
    throw new ApiError(response.status, errorMsg);
  }

  return response.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Claim API Methods
// ---------------------------------------------------------------------------

export async function createClaim(data: {
  policy_number: string;
  vehicle_make: string;
  vehicle_model: string;
  vehicle_year: number;
  vehicle_segment: string;
  incident_description?: string;
  incident_date?: string;
}): Promise<Claim> {
  return request<Claim>("/claims", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function updateClaim(
  claimId: string,
  data: Partial<Claim>
): Promise<Claim> {
  return request<Claim>(`/claims/${claimId}`, {
    method: "PATCH",
    body: JSON.stringify(data),
  });
}

export async function getClaim(claimId: string): Promise<Claim> {
  return request<Claim>(`/claims/${claimId}`);
}

export async function listClaims(): Promise<Claim[]> {
  return request<Claim[]>("/claims");
}

export async function uploadImage(
  claimId: string,
  file: File
): Promise<ImageRecord> {
  const formData = new FormData();
  formData.append("file", file);

  return request<ImageRecord>(`/claims/${claimId}/images`, {
    method: "POST",
    body: formData,
  });
}

export async function deleteImage(imageId: string): Promise<{ deleted: boolean }> {
  return request<{ deleted: boolean }>(`/images/${imageId}`, {
    method: "DELETE",
  });
}

export async function submitClaim(claimId: string): Promise<Claim> {
  return request<Claim>(`/claims/${claimId}/submit`, {
    method: "POST",
  });
}

export async function assessClaim(claimId: string): Promise<AssessmentResult> {
  return request<AssessmentResult>(`/claims/${claimId}/assess`, {
    method: "POST",
  });
}

export async function getAssessmentStatus(claimId: string): Promise<{
  claim_id: string;
  status: string;
  is_complete: boolean;
  has_assessment: boolean;
  updated_at: string;
}> {
  return request(`/assessments/${claimId}/status`);
}

export async function getClaimAssessment(
  claimId: string
): Promise<AssessmentResult> {
  return request<AssessmentResult>(`/claims/${claimId}/assessment`);
}

export async function getClaimTimeline(
  claimId: string
): Promise<TimelineEvent[]> {
  return request<TimelineEvent[]>(`/claims/${claimId}/timeline`);
}

// ---------------------------------------------------------------------------
// Reviewer & Dashboard API Methods
// ---------------------------------------------------------------------------

export async function getReviewQueue(): Promise<Claim[]> {
  return request<Claim[]>("/reviews/queue");
}

export async function saveReviewCorrection(
  claimId: string,
  payload: ReviewCorrectionPayload
): Promise<Claim> {
  return request<Claim>(`/reviews/${claimId}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export async function submitReviewDecision(
  claimId: string,
  payload: ReviewDecisionPayload
): Promise<Claim> {
  return request<Claim>(`/reviews/${claimId}/decision`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getDashboardSummary(): Promise<DashboardSummary> {
  return request<DashboardSummary>("/dashboard/summary");
}

export async function getModelBenchmarks(): Promise<ModelBenchmarkResponse> {
  return request<ModelBenchmarkResponse>("/models/benchmark");
}

export async function setActiveModel(
  task: string,
  modelId: string
): Promise<{ status: string; task: string; active_model: string }> {
  return request<{ status: string; task: string; active_model: string }>("/models/active", {
    method: "POST",
    body: JSON.stringify({ task, model_id: modelId }),
  });
}

// ---------------------------------------------------------------------------
// Formatters & UI Helpers
// ---------------------------------------------------------------------------

export function formatINR(amount: number): string {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(amount);
}

export function formatDate(isoString: string): string {
  if (!isoString) return "—";
  try {
    const d = new Date(isoString);
    return d.toLocaleDateString("en-IN", {
      day: "numeric",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return isoString;
  }
}

export function getImageUrl(localPath: string): string {
  if (!localPath) return "";
  if (localPath.startsWith("http://") || localPath.startsWith("https://")) {
    return localPath;
  }
  // Extract relative path inside uploads/
  const norm = localPath.replace(/\\/g, "/");
  const idx = norm.indexOf("uploads/");
  const rel = idx !== -1 ? norm.substring(idx) : `uploads/${norm.split("/").pop()}`;

  // In the browser, prefer same-origin relative URL so Next.js rewrites cleanly proxy it
  if (typeof window !== "undefined") {
    return `/${rel}`;
  }
  return `${BACKEND_BASE_URL}/${rel}`;
}

export function getRouteInfo(route: TriageRoute | string): {
  label: string;
  color: string;
  bg: string;
  border: string;
  description: string;
} {
  switch (route) {
    case "FAST_TRACK_ELIGIBLE":
      return {
        label: "Fast-Track Instant Settlement",
        color: "text-emerald-800",
        bg: "bg-emerald-50",
        border: "border-emerald-200",
        description:
          "Minor or moderate damage within automated policy limits. Approved for immediate automated settlement.",
      };
    case "FRAUD_REVIEW":
      return {
        label: "Fraud Review Required",
        color: "text-rose-800",
        bg: "bg-rose-50",
        border: "border-rose-200",
        description:
          "Suspicious image signal or duplicate detected. Automated processing paused for Special Investigation Unit (SIU) review.",
      };
    case "MANUAL_DAMAGE_REVIEW":
      return {
        label: "Adjuster Settlement Review",
        color: "text-amber-800",
        bg: "bg-amber-50",
        border: "border-amber-200",
        description:
          "Severe collision damage or high repair estimate detected. Routed to an insurance claims adjuster for final payout approval.",
      };
    case "MORE_EVIDENCE_REQUIRED":
      return {
        label: "More Evidence Required",
        color: "text-orange-800",
        bg: "bg-orange-50",
        border: "border-orange-200",
        description:
          "Uploaded photo is blurry, dark, low-resolution, or corrupt. Please resubmit clear evidence.",
      };
    case "TECHNICAL_REVIEW":
    default:
      return {
        label: "Technical Review",
        color: "text-slate-800",
        bg: "bg-slate-50",
        border: "border-slate-200",
        description:
          "Model or system evaluation anomaly. Requires technical inspection.",
      };
  }
}

export function getStatusBadge(status: string): {
  label: string;
  color: string;
  bg: string;
} {
  switch (status) {
    case "DRAFT":
      return { label: "Draft", color: "text-slate-700", bg: "bg-slate-100" };
    case "SUBMITTED":
      return { label: "Submitted", color: "text-blue-700", bg: "bg-blue-100" };
    case "ASSESSING_FRAUD":
    case "ASSESSING_DAMAGE":
    case "VALIDATING":
      return {
        label: "Assessing",
        color: "text-indigo-700",
        bg: "bg-indigo-100",
      };
    case "FAST_TRACK_ELIGIBLE":
      return {
        label: "Fast-Track",
        color: "text-emerald-700",
        bg: "bg-emerald-100",
      };
    case "FRAUD_REVIEW":
      return {
        label: "Fraud Review",
        color: "text-rose-700",
        bg: "bg-rose-100",
      };
    case "MANUAL_DAMAGE_REVIEW":
      return {
        label: "Damage Review",
        color: "text-amber-700",
        bg: "bg-amber-100",
      };
    case "MORE_EVIDENCE_REQUIRED":
      return {
        label: "Needs Evidence",
        color: "text-orange-700",
        bg: "bg-orange-100",
      };
    case "COMPLETED":
      return {
        label: "Completed",
        color: "text-teal-700",
        bg: "bg-teal-100",
      };
    default:
      return { label: status, color: "text-slate-700", bg: "bg-slate-100" };
  }
}
