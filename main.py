import os
import logging
import json
import uvicorn
import requests
import re
import math
from fastapi import FastAPI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import List, Dict

import tools

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile") 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Tool LLM Agent")

# --- Agent Configuration ---
AVAILABLE_TOOLS = {
    "get_news": {
        "function": tools.get_news,
        "description": "Fetches recent news articles about a specific topic, person, or company.",
        "args": {"topic": {"type": "string"}}
    },
    "get_weather": {
        "function": tools.get_weather,
        "description": "Provides the current weather, climate, or forecast for a given city.",
        "args": {"city": {"type": "string"}}
    },
    "get_stock_price": {
        "function": tools.get_stock_price,
        "description": "Gets the latest stock price for a company using its stock ticker symbol (e.g., AAPL for Apple).",
        "args": {"ticker_symbol": {"type": "string"}}
    },
    "get_wikipedia_summary": {
        "function": tools.get_wikipedia_summary,
        "description": "Retrieves a concise summary of a topic from Wikipedia.",
        "args": {"search_term": {"type": "string"}}
    },
    "calculator": {
        "function": tools.calculator,
        "description": "Use ONLY for evaluating simple, direct arithmetic expressions like '4 * (3 + 2)'. This tool cannot handle abstract math concepts, variables, or word problems like 'what is calculus?'.",
        "args": {"expression": {"type": "string"}}
    }
}

# Native tool calling definitions for Groq API
GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": prop["description"],
            "parameters": {
                "type": "object",
                "properties": prop["args"],
                "required": list(prop["args"].keys())
            }
        }
    }
    for name, prop in AVAILABLE_TOOLS.items()
]


class QueryRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = Field(default_factory=list)

def is_purely_math_query(query: str) -> bool:
    math_pattern = r"^[\d\s\+\-\*/\(\)\.]+$"
    return bool(re.fullmatch(math_pattern, query.strip()))


def get_active_model(api_key: str) -> tuple[str, list]:
    configured = os.getenv("GROQ_MODEL_NAME")
    try:
        res = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            timeout=5
        )
        if res.status_code == 200:
            data = res.json().get("data", [])
            available = [m["id"] for m in data if not any(x in m.get("id", "").lower() for x in ["whisper", "guard"])]
            
            if configured and configured in available and "8b" not in configured:
                return configured, available
            
            preferences = [
                "openai/gpt-oss-20b",
                "openai/gpt-oss-120b",
                "llama-3.3-70b-versatile",
                "llama3-70b-8192",
                "llama3-8b-8192"
            ]
            for candidate in preferences:
                if candidate in available:
                    return candidate, available
            
            if available:
                return available[0], available
    except Exception as e:
        logger.error(f"Error checking available Groq models: {e}")
    
    return configured or "openai/gpt-oss-20b", []


def orchestrate_agent(query: str, history: List[Dict[str, str]]) -> dict:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            "query": query,
            "result": "Configuration Error: `GROQ_API_KEY` is not set. Please check your Vercel Environment Variables and verify 'Production' is checked."
        }

    model_to_use, available_models = get_active_model(api_key)
    logger.info(f"Orchestrating with LLM ({model_to_use}) for query: {query}")

    try:
        system_instruction = (
            "You are a helpful and intelligent multi-tool assistant. "
            "Use the provided tools when the user's request requires live data or actions (such as checking weather, stock prices, news, Wikipedia, or math calculations). "
            "If no tool is needed, respond directly with a helpful conversational answer."
        )
        messages = [{"role": "system", "content": system_instruction}] + history + [{"role": "user", "content": query}]

        payload = {
            "model": model_to_use,
            "messages": messages,
            "temperature": 0.0,
            "tools": GROQ_TOOLS,
            "tool_choice": "auto"
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            json=payload
        )
        response.raise_for_status()
        choice_msg = response.json()['choices'][0]['message']

        tool_calls = choice_msg.get("tool_calls")
        if tool_calls:
            first_call = tool_calls[0]["function"]
            tool_name = first_call.get("name")
            try:
                arguments = json.loads(first_call.get("arguments", "{}"))
            except Exception:
                arguments = {}

            logger.info(f"LLM called tool: '{tool_name}' with arguments: {arguments}")

            if tool_name in AVAILABLE_TOOLS:
                result = AVAILABLE_TOOLS[tool_name]["function"](**arguments)
                return {"query": query, "result": result}
            else:
                return {"query": query, "result": f"Error: The LLM chose a tool ('{tool_name}') that does not exist."}
        else:
            # Direct conversational response
            content = choice_msg.get("content", "")
            return {"query": query, "result": content}

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error from LLM: {e.response.text}")
        try:
            error_details = e.response.json().get("error", {})
            error_code = error_details.get("code")
            error_msg = error_details.get("message", e.response.text)

            if error_code == "rate_limit_exceeded":
                match = re.search(r"Please try again in (\d+\.?\d*)s", error_msg)
                wait_time = math.ceil(float(match.group(1))) if match else 10
                return {"query": query, "result": f"Rate limit reached. Please wait {wait_time}s and try again."}
            elif error_code == "invalid_api_key":
                return {
                    "query": query,
                    "result": f"Authentication Error: Groq API key is invalid or expired ({error_msg}). Please verify `GROQ_API_KEY` in Vercel."
                }
            elif error_code in ("model_not_found", "model_decommissioned") or e.response.status_code == 404:
                return {
                    "query": query,
                    "result": f"Model Error: Model '{model_to_use}' is unavailable. Your Groq API key has access to these models: {available_models}."
                }
            else:
                return {
                    "query": query,
                    "result": f"Groq API Error ({e.response.status_code}): {error_msg}"
                }
        except (json.JSONDecodeError, AttributeError):
            return {"query": query, "result": f"An API error occurred (Status Code: {e.response.status_code}): {e.response.text}"}
            
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return {"query": query, "result": f"An unexpected error occurred in the agent logic: {e}"}


@app.post("/orchestrate")
async def orchestrate(request: QueryRequest):
    query = request.query.strip()
    history = request.history
    logger.info(f"Received query: '{query}' with history length: {len(history)}")

    if is_purely_math_query(query):
        logger.info("Math intent detected. Bypassing LLM and using calculator tool directly.")
        result = tools.calculator(expression=query)
        return {"query": query, "result": result}
    else:
        return orchestrate_agent(query, history)

# --- Local Development Setup ---
if os.getenv("VERCEL") != "1":
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory="public", html=True), name="static")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
