"""pipeline — unified inference orchestrator.

Implemented in Phase 11 (notebook 13).

Planned public API:
    assess_claim(image_paths, config) -> AssessmentResult

The orchestrator calls modules in order:
    quality -> fraud -> detection -> severity -> costing -> decision
High fraud risk stops the pipeline before damage and cost stages.
"""
