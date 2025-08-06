# 🧠 AddressCorrector: LLMs for Real-World Address Rewriting (Supervised Approach)

A supervised LLM-based system for rewriting, correcting, and standardizing U.S. addresses using fine-tuned LLaMA models and Retrieval-Augmented Generation (RAG). Designed to reduce delivery failures and optimize logistics through intelligent address normalization.

---

## 📌 Overview

In the logistics and e-commerce industries, incorrect or incomplete address data can result in billions of dollars in failed deliveries. Traditional rule-based address parsers struggle with variability and real-world noise in address formats.

This project presents an LLM-based solution that leverages supervised fine-tuning and retrieval-augmented generation to:

- Parse raw address strings into structured formats
- Correct errors and rewrite malformed addresses
- Improve geolocation accuracy and delivery success rates

---

## 🚀 Key Features

- ✅ **Supervised Fine-Tuning**: Leveraged LLaMA-3.2 to learn address patterns from the U.S. National Address Database (NAD)
- 🔍 **Retrieval-Augmented Generation (RAG)**: Enhances address correction using real-world references for context-aware rewriting
- 🧾 **Structured Output**: Produces normalized, field-wise address formats suitable for logistics pipelines
- 📍 **Geo-Evaluation**: Benchmarked on geolocation proximity and correction accuracy

---

## 📊 Dataset

- **Source**: [U.S. Transportation Department – National Address Database (NAD)](https://www.transportation.gov)
- **Format**: Raw, semi-structured, and cleaned address entries
- **Use Cases**:
  - Structured parsing (e.g., separating street, city, ZIP)
  - Error correction (typos, reordering, missing elements)
  - Real-world noisy input simulation

---

## 🧠 Model Architecture

- **Base Model**: LLaMA-3.2 (fine-tuned on NAD data)
- **Augmentation**: RAG-based inference using local vector store of known address samples
- **Approach**:
  1. Token-level supervised training for address structuring
  2. Retrieval of geographically similar addresses for context
  3. Inference with RAG-enhanced prompt generation

---

