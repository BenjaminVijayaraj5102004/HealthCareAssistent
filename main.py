from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from Retrevel import retriever

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI model
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=openai_api_key,
)

# Define the prompt template
template="""
You are a medical assistant. You will be given a list of symptoms and possible conditions. Your task is to suggest the most likely condition based on the symptoms provided. If you cannot find a match give some suggestion related to that.
  
  here is the symptoms: {symptoms}

  here is the answer: {answer}


"""
# Create the prompt template

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model



while True:
    # Get user input
    symptoms = input("Enter symptoms (or 'exit' to quit): ")
    if symptoms.lower() == "exit":
        break
    # Run the chain
    answer = retriever.invoke(symptoms)
    result = chain.invoke({"symptoms": symptoms, "answer":answer})
    print(result.content)