import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["LANGCHAIN_PROJECT"] = "sequential_project"
load_dotenv()
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", temperature=0, max_output_tokens=1000
)

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text \n {text}",
    input_variables=["text"],
)
config = {
    "run_name": "sequencial_chain",
    "tags": ["llm", "summarization", "report generation"],
    "metadata": {
        "model_1": "gpt-4o-mini",
        "model_temp": 0,
        "parser": "stroutputparser",
    },
}

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "Unemployment in India due to AI"}, config=config)

print(result)
