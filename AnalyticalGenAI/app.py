from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json

# --- LangChain / RAG Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_core.messages import HumanMessage

# --- Flask App ---
app = Flask(__name__)
CORS(app)  # allow frontend calls

# --- API KEY ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyCdekbd5x-BqJHpDFJkniVejo_I9p6uMwQ"

# --- LLM Setup ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=5)

# ---------------- LLM Setup ----------------
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=5)

# ---------------- SIMPLE RAG CHAIN ----------------
class SimpleRagChain:
    def __init__(self, retriever, df):
        self.retriever = retriever
        self.df = df  # keep raw data
        # Column descriptions
        self.column_desc = {
            "lat": "Latitude",
            "lon": "Longitude",
            "year": "Year",
            "month": "Month",
            "sst": "Sea Surface Temperature (SST, °C)",
            "poc": "Particulate Organic Carbon (POC, mg/m³)",
            "pic": "Particulate Inorganic Carbon (PIC, mg/m³)",
            "aot_862": "Aerosol Optical Thickness (AOT_862)",
            "chlor_a": "Chlorophyll-a (mg/m³)",
            "Kd_490": "Water Turbidity / Clarity (Kd_490, m⁻¹)",
        }

    def find_nearest(self, lat, lon, top_k=3):
        """Find top-k nearest locations using raw lat/lon"""
        self.df["distance"] = self.df.apply(
            lambda row: geodesic((lat, lon), (float(row["lat"]), float(row["lon"]))).km,
            axis=1,
        )
        nearest = self.df.nsmallest(top_k, "distance")
        results = []
        for _, row in nearest.iterrows():
            result = {desc: row[col] for col, desc in self.column_desc.items() if col in row}
            result["Distance from query (km)"] = f"{row['distance']:.2f}"
            results.append(result)
        return {"answer": results}

    def format_nearest_human_readable(self, lat, lon, top_k=3):
        """Return a human-readable string for nearest points"""
        data = self.find_nearest(lat, lon, top_k)["answer"]
        lines = [f"🌊 Oceanographic data near ({lat}, {lon}):"]
        for i, res in enumerate(data, 1):
            lines.append(f"\n📍 Result {i}:")
            for k, v in res.items():
                lines.append(f"   - {k}: {v}")
        return "\n".join(lines)

# ---------------- VECTORSTORE BUILDER ----------------
def build_vectorstore(parquet_path, persist_directory="./chroma_db", batch_size=1000):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    df = pd.read_parquet(parquet_path).astype(str)

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return SimpleRagChain(vectorstore.as_retriever(search_kwargs={"k": 5}), df)

    vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    
    for batch in np.array_split(df, len(df)//batch_size + 1):
        docs = [
            Document(
                page_content=(
                    f"Location: ({row['lat']}, {row['lon']}), "
                    f"Year: {row['year']}, Month: {row['month']}, "
                    f"SST: {row['sst']}°C, POC: {row['poc']}, PIC: {row['pic']}, "
                    f"AOT_862: {row['aot_862']}, Chlor_a: {row['chlor_a']}, "
                    f"Kd_490: {row['Kd_490']}"
                ),
                metadata=row.to_dict()
            )
            for _, row in batch.iterrows()
        ]
        vectorstore.add_documents(docs)
        vectorstore.persist()

    return SimpleRagChain(vectorstore.as_retriever(search_kwargs={"k": 5}), df)

# ---------------- TOOLS ----------------
def _tool1_impl(user_input: str) -> str:
    prompt = f"""
    Extract the city name only from the user query.
    If user provided lat/lon, return "COORDINATES".
    If nothing, return "UNKNOWN".
    Query: {user_input}
    """
    response = llm([HumanMessage(content=prompt)])
    return response.content.strip()

def _tool2_impl(city_name: str):
    geolocator = Nominatim(user_agent="rag_location_app")
    location = geolocator.geocode(city_name)
    if not location:
        return f"❌ Could not find coordinates for {city_name}."
    return rag_chain.format_nearest_human_readable(location.latitude, location.longitude, top_k=3)

def _tool3_impl(coords: str):
    try:
        lat, lon = map(float, coords.split(","))
        return rag_chain.format_nearest_human_readable(lat, lon, top_k=3)
    except Exception as e:
        return f"❌ Invalid coordinates: {str(e)}"

tool1 = Tool(name="extract_city", func=_tool1_impl, description="Extract city from query")
tool2 = Tool(name="city_to_results", func=_tool2_impl, description="Get oceanographic data for a city")
tool3 = Tool(name="coords_to_results", func=_tool3_impl, description="Get oceanographic data for coordinates")

agent_executor = initialize_agent(
    tools=[tool1, tool2, tool3],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)

# ---------------- AGENT RUNNER ----------------
def run_with_agent(user_input: str):
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    if user_input.lower().strip() in greetings:
        return "👋 Hi there! How can I help you with SST or oceanographic data today?"

    try:
        return agent_executor.run(input=user_input)
    except Exception as e:
        return f"❌ Agent Error: {e}"

# ---------------- INITIALIZE VECTORSTORE ----------------
rag_chain = build_vectorstore("dummy_ocean_data.parquet")

# ---------------- API ENDPOINT ----------------
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_prompt = data.get("prompt", "")
    response = run_with_agent(user_prompt)
    return jsonify({"response": response})

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)