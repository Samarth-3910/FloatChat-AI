from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
 
# --- LangChain / RAG Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd 
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from langchain_core.tools import Tool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

# --- Flask App ---
app = Flask(__name__)
CORS(app)

# --- API KEY ---
os.environ["GOOGLE_API_KEY"] = ""

# --- LLM Setup ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=5)

# ---------------- SIMPLE RAG CHAIN ----------------
class SimpleRagChain:
    def __init__(self, retriever, df):
        self.retriever = retriever
        self.df = df
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
        data = self.find_nearest(lat, lon, top_k)["answer"]
        lines = [f"🌊 Oceanographic data near ({lat}, {lon}):"]
        for i, res in enumerate(data, 1):
            lines.append(f"\n📍 Result {i}:")
            for k, v in res.items():
                lines.append(f"   - {k}: {v}")
        return "\n".join(lines)

# ---------------- VECTORSTORE BUILDER ----------------
def build_vectorstore(parquet_path, persist_directory="./chroma_db", batch_size=1000):
    if not os.path.exists(parquet_path):
        print(f"Warning: {parquet_path} not found. Creating dummy DataFrame.")
        data = {
            'lat': ['34.0', '35.0', '36.0'],
            'lon': ['-119.0', '-118.0', '-117.0'],
            'year': ['2023', '2023', '2024'],
            'month': ['06', '07', '01'],
            'sst': ['20.5', '22.1', '15.9'],
            'poc': ['0.8', '0.9', '0.5'],
            'pic': ['0.1', '0.2', '0.1'],
            'aot_862': ['0.05', '0.06', '0.04'],
            'chlor_a': ['0.5', '0.7', '0.3'],
            'Kd_490': ['0.04', '0.05', '0.03']
        }
        df = pd.DataFrame(data).astype(str)
    else:
        df = pd.read_parquet(parquet_path).astype(str)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"Loading existing vectorstore from {persist_directory}")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return SimpleRagChain(vectorstore.as_retriever(search_kwargs={"k": 5}), df)

    print(f"Building new vectorstore to {persist_directory}")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    
    for batch in np.array_split(df, max(1, len(df)//batch_size + 1)):
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
        if docs:
            vectorstore.add_documents(docs)

    if len(df) > 0:
        vectorstore.persist()

    return SimpleRagChain(vectorstore.as_retriever(search_kwargs={"k": 5}), df)

# ---------------- TOOLS ----------------
geolocator = Nominatim(user_agent="rag_location_app")

def _tool1_impl(user_input: str) -> str:
    prompt = f"""
    Analyze the user's query and extract the key location information.
    1. If the query contains numerical latitude and longitude pairs (e.g., '34.0, -118.0'), respond with "COORDINATES".
    2. If the query contains a clear city, state, or area name (e.g., 'San Diego', 'Atlantic Ocean'), respond with ONLY the name (e.g., "San Diego").
    3. If the query does not contain a recognizable location, respond with "UNKNOWN".
    Query: {user_input}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

def _tool2_impl(city_name: str) -> str:
    try:
        location = geolocator.geocode(city_name)
        if not location:
            return f"❌ Could not find coordinates for {city_name}. Please try a more specific location."
        return rag_chain.format_nearest_human_readable(location.latitude, location.longitude, top_k=3)
    except Exception as e:
        return f"❌ Error during city lookup: {str(e)}"

def _tool3_impl(coords: str) -> str:
    try:
        lat, lon = map(float, coords.split(","))
        return rag_chain.format_nearest_human_readable(lat, lon, top_k=3)
    except Exception as e:
        return f"❌ Invalid coordinates format or data error: {str(e)}"

tool1 = Tool(
    name="extract_location_type", 
    func=_tool1_impl, 
    description="Tool to classify the user's query as a 'CITY NAME', 'COORDINATES', or 'UNKNOWN'."
)
tool2 = Tool(
    name="get_data_by_city", 
    func=_tool2_impl, 
    description="Use this tool to get the nearest oceanographic data after 'extract_location_type' returns a CITY NAME."
)
tool3 = Tool(
    name="get_data_by_coords", 
    func=_tool3_impl, 
    description="Use this tool to get the nearest oceanographic data after 'extract_location_type' returns 'COORDINATES'. Input must be a comma-separated string of latitude and longitude (e.g., '34.0,-118.0')."
)

# Define the prompt locally
template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

# Construct the ReAct agent
agent = create_react_agent(llm, [tool1, tool2, tool3], prompt)

# Create an agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[tool1, tool2, tool3],
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    memory=memory
)

# ---------------- AGENT RUNNER ----------------
def run_with_agent(user_input: str):
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    if user_input.lower().strip() in greetings:
        return "👋 Hi there! I am an Oceanographic Data Assistant. I can look up nearby sea surface temperature (SST), Chlorophyll-a (Chlor_a), and other data based on a location. Try asking for data near 'San Diego' or at coordinates '34.0, -118.0'."

    try:
        return agent_executor.invoke({"input": user_input})["output"]
    except Exception as e:
        print(f"Full Agent Error: {e}")
        return f"❌ Agent Error: I encountered an issue while processing your request. The model may have generated an invalid response. Please rephrase your query. Original error: {e}"

# ---------------- INITIALIZE VECTORSTORE ----------------
try:
    rag_chain = build_vectorstore("dummy_ocean_data.parquet")
    print("\n✅ RAG Chain and Vectorstore initialized successfully.")
except Exception as e:
    print(f"\nFATAL ERROR during Vectorstore initialization: {e}")
    rag_chain = None

# ---------------- API ENDPOINT ----------------
@app.route("/api/chat", methods=["POST"])
def chat():
    if rag_chain is None:
        return jsonify({"response": "❌ System initialization failed. Cannot process request."}), 500

    data = request.get_json()
    user_prompt = data.get("prompt", "")
    response = run_with_agent(user_prompt)
    return jsonify({"response": response})

# ---------------- MAIN ----------------
if __name__ == "__main__":
    if rag_chain is not None:
        app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False) 
    else:
        print("Application startup failed due to RAG Chain error.")
