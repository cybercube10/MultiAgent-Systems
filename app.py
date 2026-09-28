import os
import certifi

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_agent


# Fix SSL certificate path
os.environ["SSL_CERT_FILE"] = certifi.where()

# Load variables from .env
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")


# Create Tavily search tool
search_tool = TavilySearchResults(
    max_results=2,
    tavily_api_key=tavily_api_key
)


# Test the search tool
result = search_tool.invoke("latest news on AI")
print(result)


# Create Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5,
    api_key=groq_api_key
)


# Give the agent access to the search tool
tools = [search_tool]


# Create the agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a helpful assistant. "
        "Use the search tool when you need current information."
    )
)


# Run the agent
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Find the capital of India."
            }
        ]
    }
)

print(response)