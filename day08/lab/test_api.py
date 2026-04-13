import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv(override=True)
llm= ChatOpenAI(model="gpt-4o-mini")
# llm= ChatGoogleGenerativeAI(model="gemini-2.5-flash")
print(llm.invoke("xin chào").content)
