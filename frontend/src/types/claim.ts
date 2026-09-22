/**
 * claim.ts — TypeScript interfaces matching ClaimVision AI API contracts.
 */

export type ClaimStatus =
  | "DRAFT"
  | "SUBMITTED"
  | "VALIDATING"
  | "MORE_EVIDENCE_REQUIRED"
  | "ASSESSING_FRAUD"
  | "FRAUD_REVIEW"
  | "ASSESSING_DAMAGE"
  | "MANUAL_DAMAGE_REVIEW"
  | "FAST_TRACK_ELIGIBLE"
  | "TECHNICAL_REVIEW"
  | "COMPLETED";

export type TriageRoute =
  | "FAST_TRACK_ELIGIBLE"
  | "FRAUD_REVIEW"
  | "MANUAL_DAMAGE_REVIEW"
  | "MORE_EVIDENCE_REQUIRED"
  | "TECHNICAL_REVIEW";

export interface ImageRecord {
  id: string;
  claim_id: string;
  filename: string;
  local_path: string;
  file_size_bytes: number;
  mime_type: string;
  sha256: string;
  width: number;
  height: number;
  created_at: string;
}

export interface TimelineEvent {
  id: string;
  claim_id: string;
  from_status: string;
  to_status: string;
  reason: string;
  actor: string;
  timestamp: string;
}

export interface QualitySummary {
  blur_score: number;
  brightness: number;
  contrast: number;
  acceptable: boolean;
  route: string;
  warnings: string[];
}

export interface FraudSummary {
  probability: number;
  risk_level: "low" | "medium" | "high";
  route: string;
  model_version: string;
  warnings: string[];
}

export interface SeveritySummary {
  predicted_class: "minor" | "moderate" | "severe";
  confidence: number;
  probabilities: Record<string, number>;
  model_version: string;
  warnings: string[];
}

export interface LocationSummary {
  predicted_part: string;
  confidence: number;
  probabilities: Record<string, number>;
  top3: [string, number][];
  model_type: string;
  model_version: string;
  warning?: string;
}

export interface DetectionItem {
  class_id: number;
  label: string;
  confidence: number;
  box_xyxy: [number, number, number, number];
  box_normalized: [number, number, number, number];
}

export interface CostBreakdownItem {
  part: string;
  severity: string;
  action: "repair" | "replace";
  min_cost: number;
  max_cost: number;
  labor_min?: number;
  labor_max?: number;
  paint_min?: number;
  paint_max?: number;
}

export interface CostSummary {
  currency: string;
  min_cost: number;
  max_cost: number;
  breakdown: CostBreakdownItem[];
  vehicle_segment: string;
  notes: string[];
}

export interface AssessmentResult {
  claim_id: string;
  image_path: string;
  route: TriageRoute;
  reason_codes: string[];
  quality: QualitySummary | null;
  fraud: FraudSummary | null;
  severity: SeveritySummary | null;
  location: LocationSummary | null;
  detections: DetectionItem[];
  cost: CostSummary | null;
  model_versions: Record<string, string>;
  inference_ms: number;
  warnings: string[];
}

export interface ReviewRecord {
  reviewer_id: string;
  decision: string;
  notes: string;
  overrides: Record<string, unknown>;
  timestamp: string;
}

export interface CostRange {
  min: number;
  max: number;
}

export interface DashboardSummary {
  total_claims: number;
  status_counts: Record<string, number>;
  fast_track_count: number;
  fraud_review_count: number;
  damage_review_count: number;
  completed_count: number;
  average_cost: CostRange;
  severity_distribution: Record<string, number>;
  location_distribution: Record<string, number>;
  fraud_score_distribution: number[];
  override_rate: number;
}

export interface ReviewCorrectionPayload {
  reviewer_id?: string;
  notes?: string;
  overrides?: Record<string, unknown>;
}

export interface ReviewDecisionPayload {
  reviewer_id?: string;
  decision: "APPROVED" | "REJECTED" | "REQUEST_INFO";
  notes?: string;
  overrides?: Record<string, unknown>;
}

export interface Claim {
  id: string;
  policy_number: string;
  vehicle_make: string;
  vehicle_model: string;
  vehicle_year: number;
  vehicle_segment: string;
  incident_description: string;
  incident_date: string;
  status: ClaimStatus;
  images: ImageRecord[];
  assessment: AssessmentResult | null;
  review: ReviewRecord | null;
  timeline: TimelineEvent[];
  created_at: string;
  updated_at: string;
}
