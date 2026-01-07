# 🌊 Visual FloatChat

**Visual FloatChat** is an AI-powered conversational system that makes oceanographic (ARGO float) data accessible through natural language and interactive visualizations.

## 🚀 Overview

This project bridges the gap between complex NetCDF ocean data and users by providing:
1.  **Interactive Dashboard:** For exploring maps, charts, and trends.
2.  **AI Chat Assistant:** For querying data in plain English (e.g., *"What is the temperature near Miami?"*).

## 🛠️ Tech Stack

*   **Languages:** Python 3.10+
*   **Frontend:** Streamlit, Plotly, Folium
*   **Backend:** Flask, LangChain, Google Gemini
*   **Data:** PostgreSQL, ChromaDB (Vector Store), Parquet, XArray

## 📂 Repository Structure

The project is organized into the following key directories:

*   **`AnalyticalGenAI/`**: Contains the core application logic.
    *   `app.py`: Flask backend server for the AI assistant.
    *   `oceanography_dashboard.py`: Streamlit frontend application.
    *   `run.py`: Auto-runner script to launch both servers.
    *   `chroma_db/`: Vector database storage.
*   **`DataEngineering/`**: Contains the ETL pipeline for data processing.
    *   `Bronze_Data/`: Storage for raw Bronze layer files.
    *   `main.ipynb`: Main notebook running the extraction and transformation pipeline.
    *   `silver_layer.ipynb`, `gold_layer.ipynb`: Notebooks for intermediate data processing.
*   **`Others/`**: Supplementary files and resources.

## ⚙️ Getting Started

### 1. Prerequisites
*   Python 3.10+
*   PostgreSQL installed and running locally.

### 2. Installation
Clone the repository:
```bash
git clone https://github.com/Samarth-3910/FloatChat-AI
cd FloatChat-AI
```

### 3. Configuration
Create a `.env` file in the root (or `AnalyticalGenAI` folder) with your API key:
```env
GOOGLE_API_KEY="AIzaSy...Your...Key"
```
*Note: Ensure PostgreSQL connection strings are correctly set in the scripts if needed.*

### 4. Run the Application
Navigate to the application folder and use the auto-runner:

```bash
cd AnalyticalGenAI
python run.py
```
This script will:
1.  Check and install requirements (`requirements.txt`).
2.  Start the Flask backend.
3.  Launch the Streamlit dashboard at **http://localhost:8501**.

---
*Hire us:*

*   [Samarth Keshari](https://www.linkedin.com/in/samarth-keshari-b61089294/)
*   [Nikhil Tiwari](https://www.linkedin.com/in/nikhil-tiwari-90194a318/)
*   [Harsh Pal](https://www.linkedin.com/in/harsh-pal-32247728a/)
*   [Kapil Patel](https://www.linkedin.com/in/scorpiankapil/)
*   [Komal Patel](https://www.linkedin.com/in/komal-patel-5b96a8239/)

<br>

<div align="center">
  <img src="https://media.giphy.com/media/xT9IgG50Fb7Mi0prBC/giphy.gif" width="400" alt="Thank You">
</div>