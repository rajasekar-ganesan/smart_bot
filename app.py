from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# ==============================
# LOAD ENV
# ==============================
load_dotenv()

# ==============================
# IMPORTS
# ==============================
from rag import create_or_load_vectorstore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

import requests
from datetime import datetime
import pytz

# ==============================
# CONFIG
# ==============================
PDF_PATH = r"G:\smart\data\pdf1.pdf"

# ==============================
# INIT APP
# ==============================
app = FastAPI()

print("🚀 Smart AI Assistant (FINAL)")

# ==============================
# LOAD RAG
# ==============================
vectorstore = create_or_load_vectorstore(PDF_PATH)
retriever = vectorstore.as_retriever()

# ==============================
# GEMINI MODEL
# ==============================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # ✅ safer (avoid quota issues)
    temperature=0.3
)

# ==============================
# TOOL 1: TIME
# ==============================
def time_tool(city: str = "Chennai") -> str:
    try:
        ist = pytz.timezone("Asia/Kolkata")
        now = datetime.now(ist)
        return now.strftime("%I:%M %p")
    except:
        return "Unable to fetch time"

time_tool_obj = Tool(
    name="Time",
    func=time_tool,
    description="Use this to get current local time"
)

# ==============================
# TOOL 2: WEATHER
# ==============================
def weather_tool(city: str) -> str:
    try:
        city = city.split()[0]  # ✅ fix: clean city name

        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
        geo_res = requests.get(geo_url).json()

        if "results" not in geo_res:
            return f"City '{city}' not found"

        lat = geo_res["results"][0]["latitude"]
        lon = geo_res["results"][0]["longitude"]

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current_weather=true"
        )

        weather_res = requests.get(weather_url).json()
        current = weather_res["current_weather"]

        temp = current["temperature"]
        wind = current["windspeed"]

        current_time = time_tool(city)

        return (
            f"Current weather in {city}:\n"
            f"🌡️ Temperature: {temp}°C\n"
            f"💨 Wind Speed: {wind} km/h\n"
            f"🕒 Local time: {current_time}"
        )

    except Exception as e:
        return f"Weather error: {str(e)}"

weather = Tool(
    name="Weather",
    func=weather_tool,
    description="Use this ONLY when user asks about weather of a city"
)

# ==============================
# TOOL 3: CALCULATOR
# ==============================
def calculator_tool(input_text: str) -> str:
    try:
        return str(eval(input_text, {"__builtins__": None}, {}))
    except:
        return "Invalid math expression"

calculator = Tool(
    name="Calculator",
    func=calculator_tool,
    description="Use this for math calculations"
)

# ==============================
# TOOL 4: RAG
# ==============================
def rag_tool(query: str) -> str:
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found in the document."

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer ONLY from the context below.
    If answer is not present, say "Not found in document".

    {context}

    Question: {query}
    """

    response = llm.invoke(prompt)
    return response.content.strip()

rag = Tool(
    name="PDF_QA",
    func=rag_tool,
    description=(
        "Use this tool ONLY when question is about the uploaded PDF document."
    )
)

# ==============================
# AGENT
# ==============================
tools = [time_tool_obj, weather, calculator, rag]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# ==============================
# REQUEST MODEL
# ==============================
class ChatRequest(BaseModel):
    question: str

# ==============================
# ROOT (AUTO REDIRECT)
# ==============================
@app.get("/")
def home():
    return RedirectResponse(url="/docs")

# ==============================
# CHAT ENDPOINT
# ==============================
@app.post("/chat")
def chat(request: ChatRequest):
    try:
        response = agent.run(request.question)
        return {
            "question": request.question,
            "answer": response
        }
    except Exception as e:
        return {"error": str(e)}