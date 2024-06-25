# Android Edge Vision for Logistics Diagrams

Generated on 2026-04-26T04:29:37Z from README narrative plus project blueprint requirements.

## On-device inference pipeline

```mermaid
flowchart TD
    N1["Step 1\nAligned with yard teams to prioritise use cases (container IDs, PPE, defects, vehi"]
    N2["Step 2\nBuilt data pipeline: image collection/annotation, EDA for class balance and hard c"]
    N1 --> N2
    N3["Step 3\nTrained and exported to TensorFlow Lite with 32/16/8-bit quantisation; added on-de"]
    N2 --> N3
    N4["Step 4\nRaised precision/recall by tuning IoU and NMS, pruning, threshold calibration; pro"]
    N3 --> N4
    N5["Step 5\nSet up evaluation harness, telemetry, retraining loop; partnered with Android engi"]
    N4 --> N5
```

## Quantization impact chart

```mermaid
flowchart LR
    N1["Inputs\nImages or camera frames entering the inference workflow"]
    N2["Decision Layer\nQuantization impact chart"]
    N1 --> N2
    N3["User Surface\nOperator-facing UI or dashboard surface described in the README"]
    N2 --> N3
    N4["Business Outcome\nInference or response latency"]
    N3 --> N4
```

## Evidence Gap Map

```mermaid
flowchart LR
    N1["Present\nREADME, diagrams.md, local SVG assets"]
    N2["Missing\nSource code, screenshots, raw datasets"]
    N1 --> N2
    N3["Next Task\nReplace inferred notes with checked-in artifacts"]
    N2 --> N3
```
