# Overview

Copper-CPSP-infrastructure is an integrated data-driven pipeline for rapid discovery and optimization of high-performance copper alloys. It combines macro‑level semantic modeling, micro‑level data extraction, and rapid alloy design to transform unstructured metallurgical literature into actionable insights and predictive models.

https://private-user-images.githubusercontent.com/111862264/598269691-b8b50f40-0579-4836-985d-fde013fb708e.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Nzk4MDc1NDAsIm5iZiI6MTc3OTgwNzI0MCwicGF0aCI6Ii8xMTE4NjIyNjQvNTk4MjY5NjkxLWI4YjUwZjQwLTA1NzktNDgzNi05ODVkLWZkZTAxM2ZiNzA4ZS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNTI2JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDUyNlQxNDU0MDBaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT00ZTFlYzI4ODg0MmNlNjUyYjRiN2ZjN2Q3NWQyZWE5ZDQ5M2Y1MzJkODdmYTAxNDY1Y2EwMGY5MWRjYjczMjM3JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcifQ.Oxmbvd3E8TRO0c_s9V-qkyTh5hWZYWrn54TxOXueVR8
The infrastructure is built around three core modules:

* Macro‑ and meso‑level semantic modeling: Large‑scale topic modeling of >120k publications using BERTopic + UMAP and LDA to map research trends and emerging topics in copper alloy design.
* Micro‑level data extraction: A hierarchical text mining pipeline coupled with heterogeneous table parsing and a vision‑language model (VLM) image classification pipeline to extract experimental compositions, processing conditions, microstructures, and properties from XML/HTML documents, tables, and figures.
* Rapid alloy design: Feature optimization and performance prediction based on thermodynamics, feature engineering, and machine‑learning models (e.g., graph‑based CPSP chain) to co‑optimize hardness and electrical conductivity. Includes validation experiments to demonstrate the pipeline’s predictive capability.

This repository provides scripts, datasets, and documentation for replicating the pipeline and applying it to copper alloys. See the docs/ folder for detailed usage instructions and examples.

Project Structure

* semantic_modeling/ – Topic modeling and trend analysis (BERTopic, UMAP, LDA).
* text_mining/ – Hierarchical text mining and entity extraction.
* table_parsing/ – XML pre‑parsing and LLM‑based table extraction.
* image_classification/ – VLM‑based multi‑panel splitting and image classification.
* design/ – Feature engineering, CPSP chain construction, and property prediction.
* docs/ – Documentation and supplementary materials.
* data/ – Example datasets and extracted features.

Citation

If you use this infrastructure in your research, please cite our manuscript.
