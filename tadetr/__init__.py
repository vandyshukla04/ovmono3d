"""TA-DETR — Terrain-Anchored Detection with Herd Scale Coupling.

Model package. HARD RULE: nothing under tadetr/ imports cubercnn or detectron2 —
the eval boundary is an instances_predictions.pth file, which is framework-agnostic.
Spec: tadetr/TADETR_SPEC.md. Plan of record: the TA-DETR section of
DetAny3D/AEROVIEW_PLAN.md (mirrors in ~/.claude/plans and /mnt/d/aeroview/plan_backup).
"""
