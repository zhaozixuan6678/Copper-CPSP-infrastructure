# Overview

Copper-CPSP-infrastructure is an integrated data-driven pipeline for rapid discovery and optimization of high-performance copper alloys. It combines macro‑level semantic modeling, micro‑level data extraction, and rapid alloy design to transform unstructured metallurgical literature into actionable insights and predictive models.

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
