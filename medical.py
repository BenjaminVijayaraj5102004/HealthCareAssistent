from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from Retrevel import retriever
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=openai_api_key,
)

template = """
You are a medical assistant. You will be given a list of symptoms and possible conditions. 
Your task is to suggest the most likely condition based on the symptoms provided. 
If you cannot find a match, give some helpful general health suggestions.

If the user asks anything unrelated to medical topics like(weather,sports and technology ,exc.), politely respond with:
"I am trained to assist only with medical-related questions."

Here are the symptoms: {symptoms}

Chat History:
{chat_history}

Here is the answer from the retriever (knowledge base): {answer}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_medical_query(symptoms, chat_history):
    answer = retriever.invoke(symptoms)
    history_str = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in chat_history])

    result = chain.invoke({
        "symptoms": symptoms,
        "answer": answer,
        "chat_history": history_str
    })
    return result.content
