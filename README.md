# Android Edge Vision for Logistics

> **Domain:** Logistics

## Overview

Logistics operations need reliable, real-time inspection without cloud connectivity or expensive hardware dependence. Manual checks for container numbers, PPE compliance, vehicle defects, and registration details are slow, inconsistent, and hard to audit across busy yards and depots. Camera feeds from varied lighting and angles degrade accuracy; network latency and data-privacy constraints block server-side inference. The business required an affordable, on-device solution running smoothly on mid-range Android phones and tablets, standardising detection quality, supporting continuous improvement. Without it, turnarounds lengthen, exceptions go unnoticed, rework and claims rise, compliance exposure grows, directly impacting throughput, safety KPIs, and operating margins.

## Approach

- Aligned with yard teams to prioritise use cases (container IDs, PPE, defects, vehicle details); set acceptance thresholds, dataset scope, edge constraints
- Built data pipeline: image collection/annotation, EDA for class balance and hard cases, augmentation for glare, low-light, occlusion; chose mobile-friendly detectors and OCR
- Trained and exported to TensorFlow Lite with 32/16/8-bit quantisation; added on-device pre/post-processing and benchmarking on mid-range Android hardware
- Raised precision/recall by tuning IoU and NMS, pruning, threshold calibration; profiled CPU/memory to meet FPS without thermal throttling
- Set up evaluation harness, telemetry, retraining loop; partnered with Android engineers for offline-first UX and background sync on connectivity

## Skills & Technologies

- TensorFlow Lite
- Model Quantization
- Object Detection
- Non-Maximum Suppression Tuning
- Android SDK Integration
- Optical Character Recognition
- Data Annotation
- Exploratory Data Analysis
- Performance Benchmarking
- On-Device Deployment
