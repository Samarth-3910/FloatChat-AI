# Project: Visual FloatChat - AI-Powered Oceanographic Data Analysis System

**Role:** AI & Backend Developer (or appropriate role)
**Technologies:** Python, LangChain, Google Gemini, Flask, Streamlit, PostgreSQL, ChromaDB, Plotly, NetCDF

## Description
Developed an end-to-end AI-powered conversational system to democratize access to complex oceanographic ARGO float data. The system bridges the gap between raw NetCDF datasets and non-technical users by enabling natural language querying, interactive visualization, and automated data processing.

## Key Contributions
*   **AI-Driven Query Engine:** Built a Retrieval-Augmented Generation (RAG) pipeline using **LangChain** and **Google Gemini** to interpret natural language queries (e.g., "Show salinity profiles near the equator") and generate SQL queries or direct answers.
*   **Data Engineering Pipeline:** Engineered an ETL pipeline to ingest and process massive **ARGO NetCDF** observation files, converting them into structured formats (Parquet/SQL) for efficient retrieval.
*   **Vector Search Integration:** Implemented **faiss/ChromaDB** for semantic search over metadata and summaries, enabling fast context-aware retrieval for the LLM.
*   **Interactive Dashboard:** Designed and deployed a **Streamlit** frontend with **Plotly** and **Folium** to visualize ocean trajectories, depth-time plots, and profile comparisons interactively.
*   **Backend Architecture:** Developed a robust **Flask** backend to handle API requests, manage database connections, and orchestrate the flow between the user interface and the AI model.

## Impact
*   Drastically reduced the time required for domain experts to extract insights from raw ocean data.
*   Enabled non-technical stakeholders to explore complex datasets without needing knowledge of NetCDF tools or programming.
*   Successfully demonstrated a Proof-of-Concept (PoC) using Indian Ocean ARGO data, with scalable architecture ready for multi-source data integration (Satellite, Gliders).
