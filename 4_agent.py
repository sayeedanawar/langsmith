import os

import requests
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["LANGCHAIN_PROJECT"] = "AGENT"
load_dotenv()

search_tool = DuckDuckGoSearchRun()


@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a given city
    """
    url = f"https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}"
    response = requests.get(url)
    return response.json()


llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest", temperature=0, max_output_tokens=1000
)

# Gemini-compatible structured ReAct chat prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the following questions as best you can. "
            "You have access to the following tools:\n\n{tools}\n\n"
            "Use the following format:\n\n"
            "Thought: you should always think about what to do\n"
            "Action: the action to take, should be one of [{tool_names}]\n"
            "Action Input: the input to the action\n"
            "Observation: the result of the action\n"
            "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
            "Thought: I now know the final answer\n"
            "Final Answer: the final answer to the original input question",
        ),
        ("human", "{input}\n\n{agent_scratchpad}"),
    ]
)

# Create the ReAct agent
agent = create_react_agent(
    llm=llm, tools=[search_tool, get_weather_data], prompt=prompt
)

# Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent, tools=[search_tool, get_weather_data], verbose=True, max_iterations=5
)

# Invoke
response = agent_executor.invoke({"input": "best place to visit in kolkata"})
print(response)
print(response["output"])
