# Task 4: LangChain Food Order Assistant
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate


# Load LLM


llm = OllamaLLM(model="llama3.2")


# Memory


memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=False
)

# Prompt Template


prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
You are a helpful assistant for QuickBite Food Delivery.

Remember the customer's delivery address and dietary preference once they mention them.
Use this information in future replies without asking again.

Conversation History:
{history}

User:
{input}

QuickBite:
"""
)

# Conversation Chain


chat = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=False
)

print("=" * 50)
print(" QuickBite Food Delivery Assistant ")
print(" Type 'exit' to end the conversation")
print("=" * 50)

turns = 0

# Chat Loop


while True:

    user_input = input("\nUser: ")

    if user_input.lower() == "exit":
        break

    reply = chat.predict(input=user_input)

    print("\nQuickBite:", reply)

    turns += 1

# Summary
print("\nConversation Ended")
print(f"Total Turns : {turns}")

print("\nMemory Buffer")
print("-" * 40)
print(memory.buffer)