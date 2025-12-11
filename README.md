# Real-Time-Fraudulent-Call-Detection-in-Arabic

## Table of Contents
1. [Project Overview](#project-overview)
2. [Objective](#objective)
3. [Dataset Summary](#dataset-summary)
4. [System Features](#system-features)
5. [Model Architecture](#model-architecture)
6. [Models Used](#models-used)
7. [Performance Summary](#performance-summary)
8. [Installation](#installation)
9. [Usage Example](#usage-example)
10. [Repository Structure](#repository-structure)
11. [Limitations](#limitations)
12. [Team](#team)

---

### Project Overview
This repository presents an end-to-end Arabic Fraud Call Detection System that combines Automatic Speech Recognition (ASR) with Natural Language Processing (NLP) to identify fraudulent phone-call interactions.

The system utilizes Whisper Large-v3 for transcription and fine-tuned Arabic transformer models to classify calls as Fraud or Safe based on linguistic cues.

Designed as a lightweight, efficient, and scalable pipeline, the project enables real-time fraud detection suitable for Arabic-speaking environments with diverse dialects and noisy conditions.

---
### Objective
To develop a reliable and practical AI system capable of detecting fraudulent behavior in Arabic phone calls while supporting multiple dialects and real-world noisy acoustic conditions.

---

## Dataset Summary
The dataset consists of **1,570 Arabic phone-call recordings**, divided into:

- **1,500 synthetic calls** (750 Fraud, 750 Safe)  
- **70 real calls** (35 Fraud, 35 Safe)

The dataset covers multiple fraud categories, including OTP theft, bank-credential scams, impersonation, investment fraud, customer-service fraud, and normal safe conversations.

Each call includes:
- Audio file  
- Script text  
- ASR transcript  
- Metadata information (dialect, category, duration, noise level, source)

### Dataset Files Included

This repository includes the metadata file used in the project:
[metadata.csv](metadata/metadata.csv)

---

## System Features

- High-accuracy Arabic ASR using Whisper Large-v3  
- Fraud/Safe classification using fine-tuned Arabic transformer models  
- Strong performance on dialectal and noisy phone audio  
- Segment-level and call-level prediction  
- Provides a fraud-risk percentage for each call  
- Highlights the key terms and phrases that triggered the fraud detection

---

## Model Architecture
- **Audio Input:** real or synthetic phone-call recordings  
- **ASR Transcription (Whisper Large-v3):** converts audio into Arabic text  
- **Segment Extraction:** divides transcripts into smaller units for finer analysis  
- **NLP Classification:** AraBERT, CAMeLBERT, and MARBERT estimate fraud probability per segment  
- **Weighted Probability Aggregation:** produces a stable call-level score  
- **Final Decision:** outputs Fraud/Safe label + risk percentage + highlighted suspicious terms

---

##  Models Used

### **ASR**
- **Whisper Large-v3**  
Selected for its strong performance on multi-dialect Arabic and noisy phone environments.

### **NLP**
Fine-tuned transformer models:
- **AraBERT**  
- **CAMeLBERT**  
- **MARBERT** (best robustness on dialectal ASR output)

---

## Performance Summary

| Component             | Metric         | Result        |
|----------------------|----------------|---------------|
| ASR                  | Average WER    | ~0.20         |
| NLP (segment-level)  | Macro F1       | 0.93–0.94     |
| NLP (call-level)     | Accuracy       | up to 98% using threshold-based aggregation |

At the call level, threshold-based aggregation achieved the most stable and accurate performance. Majority voting and weighted probability voting were evaluated as comparison baselines, but the threshold method consistently provided the most reliable final decisions.

---

## Installation
pip install transformers jiwer openai-whisper torch numpy pandas

---

## Usage Example
import whisper, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load ASR
asr = whisper.load_model("large-v3")

# Load MARBERT (local)
tokenizer = AutoTokenizer.from_pretrained("models/MARBERT_final", local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained("models/MARBERT_final", local_files_only=True)

# Transcribe
text = asr.transcribe("call.wav")["text"]

# Classify
inputs = tokenizer(text, return_tensors="pt")
logits = model(**inputs).logits





