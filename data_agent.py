import os
import re
import sys

import requests
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

# Inside Docker Compose the service name "api" is used.
# For local runs: set API_URL=http://localhost:8000
API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 10  # seconds

def normalize_machine_id(machine_id: str) -> str:
    """'machine 1', 'MCH-1', 'mch-01' -> 'MCH-01'."""
    match = re.search(r"\d+", machine_id)
    if match:
        return f"MCH-{int(match.group()):02d}"
    return machine_id.strip().upper()

def api_get(path: str, params: dict | None = None):
    response = requests.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()

# 1. DEFINE THE TOOLS

@tool
def check_machine_status(machine_id: str) -> str:
    """
    Check recent operational status and torque values
    of a specific manufacturing machine, e.g. 'MCH-01'.
    """
    try:
        machine_id = normalize_machine_id(machine_id)
        data = api_get("/api/machine_status", {"machine_id": machine_id})
        if not data:
            return f"No data found for machine ID: {machine_id}. It does not exist in the database."
        return str(data)
    except Exception as e:
        return f"Error fetching data: {e}"


@tool
def get_all_machines() -> str:
    """Get only the list of machine IDs. Contains no status or sensor data."""
    try:
        return str(api_get("/api/all_machines"))
    except Exception as e:
        return f"Error: {e}"


@tool
def get_error_summary() -> str:
    """Get a summary of all errors across all machines."""
    try:
        return str(api_get("/api/errors"))
    except Exception as e:
        return f"Error: {e}" 

@tool
def get_fleet_status() -> str:
    """Get the latest reading and status of every machine at once."""
    try:
        machines = api_get("/api/all_machines")
        latest = {}
        for m in machines:
            data = api_get("/api/machine_status", {"machine_id": m})
            latest[m] = data[-1] if data else "no data"
        return str(latest)
    except Exception as e:
        return f"Error: {e}"        



# 2. SET UP OLLAMA LLM

llm = ChatOllama(model="llama3.2", temperature=0)



# 3. BUILD THE AGENT (using langgraph)

tools = [check_machine_status, get_all_machines, get_error_summary, get_fleet_status]
agent = create_react_agent(llm, tools)



# 4. RUN IT
SYSTEM_PROMPT = (
    "You are a strict manufacturing data analyst. Answer the user's "
    "questions using only the tool results. If you don't have the data, "
    "say so. NEVER make up data. Quote values and timestamps exactly as returned. "
    "Do not invent thresholds or ranges."
)


def run_data_agent(question: str) -> str:
    """Run the tool-calling agent on one question and return the final answer."""
    result = agent.invoke({"messages": [("system", SYSTEM_PROMPT), ("user", question)]})
    return result["messages"][-1].content

if __name__ == "__main__":
    print("\n Starting Manufacturing Data Agent...\n")


    
    query = " ".join(sys.argv[1:]) or (
        "Check machine MCH-01 and tell me if there are any errors in the torque values."
    )

    messages = [
        ("system", "You are a strict manufacturing data analyst. Answer the user's "
                   "questions using only the tool results. If you don't have the data, "
                   "say so. NEVER make up data. Quote values and timestamps exactly as returned. Do not invent thresholds or ranges."),
        ("user", query),
    ]

    
    result = agent.invoke({"messages": messages})

    print("\n Agent's reasoning process:")
    for msg in result["messages"]:
       print(f"- {msg.type}: {msg.content}")
    print("\n Final Answer:")
    print(result["messages"][-1].content)