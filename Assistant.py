from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from Retrevel import retriever
from langchain_core.prompts import MessagesPlaceholder
# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI model
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=openai_api_key,
)

# Define the prompt template
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
# Create the prompt template

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

# Add this before the while loop
chat_history = []  # Initialize chat history list

while True:
    # Get user input
    print("===========Medical ChatBot===========")
    symptoms = input("Enter symptoms (or 'exit' to quit): ")
    if symptoms.lower() == "exit":
        break
    

    # Run the retriever
    answer = retriever.invoke(symptoms)

    # Prepare chat history as a string
    history_str = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in chat_history])

    # Run the chain with chat history
    result = chain.invoke({
        "symptoms": symptoms,
        "answer": answer,
        "chat_history": history_str
    })
 # Append the current interaction to history

    # Print the response
    print(result.content)

    # Append current interaction to history
    chat_history.append((symptoms, result.content))


