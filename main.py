import os
import logging
import json
import uvicorn
import requests
import re
import math # We need this to round up the seconds
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

# --- Agent Configuration (Tools and System Prompt are unchanged) ---
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

SYSTEM_PROMPT = """
You are a smart, tool-using assistant. Your goal is to accurately determine the user's intent and select the appropriate tool.

Here is your thought process:
1.  First, analyze the entire conversation history to understand the context.
2.  Second, focus on the user's **latest query** to identify their primary, immediate intent.
3.  Compare this intent against the descriptions of the available tools.

Your Rules:
- If the user's latest query CLEARLY and DIRECTLY matches the description of a tool, you MUST choose that tool. For example, queries about 'weather', 'climate', or 'forecast' should always use the 'get_weather' tool.
- If the query is a conversational follow-up that does not explicitly ask for a tool (e.g., "why is that?", "tell me more", "how?"), or if no tool is a good fit, you MUST use the 'fallback' tool to continue the conversation.

You must respond ONLY with a valid JSON object with 'tool_name' and 'arguments' keys.

Here are the available tools:
{tools_json}
"""


class QueryRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = Field(default_factory=list)

def is_purely_math_query(query: str) -> bool:
    math_pattern = r"^[\d\s\+\-\*/\(\)\.]+$"
    return bool(re.fullmatch(math_pattern, query.strip()))


def orchestrate_agent(query: str, history: List[Dict[str, str]]) -> dict:
    logger.info(f"Orchestrating with LLM ({MODEL_NAME}) for query: {query}")
    llm_choice_str = "" 

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            "query": query,
            "result": "Configuration Error: `GROQ_API_KEY` is not set. Please check your Vercel Environment Variables and verify 'Production' is checked."
        }

    try:
        tools_json_string = json.dumps([{name: props['description'], 'args': props['args']} for name, props in AVAILABLE_TOOLS.items()], indent=2)
        prompt_with_tools = SYSTEM_PROMPT.format(tools_json=tools_json_string)
        
        messages = [{"role": "system", "content": prompt_with_tools}] + history + [{"role": "user", "content": query}]

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            json={ "model": MODEL_NAME, "messages": messages, "temperature": 0.0, "response_format": {"type": "json_object"} }
        )
        response.raise_for_status()
        llm_choice_str = response.json()['choices'][0]['message']['content']
        llm_choice = json.loads(llm_choice_str)
        
        tool_name = llm_choice.get("tool_name")
        arguments = llm_choice.get("arguments", {})

        logger.info(f"LLM chose tool: '{tool_name}' with arguments: {arguments}")

        if tool_name in AVAILABLE_TOOLS:
            result = AVAILABLE_TOOLS[tool_name]["function"](**arguments)
            return {"query": query, "result": result}
        
        elif tool_name == "fallback":
            logger.info("No suitable tool found. Falling back to direct LLM call.")
            fallback_messages = [{"role": "system", "content": "You are a helpful and conversational assistant."}] + history + [{"role": "user", "content": query}]
            direct_response_payload = { "model": MODEL_NAME, "messages": fallback_messages }
            direct_response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key.strip()}"},
                json=direct_response_payload
            )
            direct_response.raise_for_status()
            return {"query": query, "result": direct_response.json()['choices'][0]['message']['content']}
        
        else:
            return {"query": query, "result": f"Error: The LLM chose a tool ('{tool_name}') that does not exist."}
    
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
                return {"query": query, "result": f"Groq Authentication Error: {error_msg}. Please verify GROQ_API_KEY in Vercel."}
            else:
                return {"query": query, "result": f"Groq API Error ({e.response.status_code}): {error_msg}"}
        except Exception:
            return {"query": query, "result": f"API Error (Status {e.response.status_code}): {e.response.text}"}
            
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return {"query": query, "result": f"Agent error: {e}"}


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


