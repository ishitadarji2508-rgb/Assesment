# Food Delivery Order Tracking Assistant

import re

from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain

# Simulated Order Database

orders = {

    "#101": "Preparing",

    "#102": "Out for Delivery",

    "#103": "Delivered",

    "#104": "Cancelled"

}

# Track Queried Orders

queried_orders = set()

# Load LLM

llm = OllamaLLM(
    model="llama3.2"
)

# Conversation Memory

memory = ConversationBufferMemory(
    memory_key="history",
    input_key="input",
    return_messages=False
) 

# Prompt Template

prompt = PromptTemplate(

    input_variables=[
        "history",
        "input"
    ],

    template="""
You are a helpful food delivery order tracking assistant.

You can answer customer questions about their food order.

Conversation History:
{history}

Customer:
{input}

Assistant:
"""

)

# Conversation Chain

conversation = ConversationChain(

    llm=llm,

    memory=memory,

    prompt=prompt,

    verbose=False

)

# Terminal Chat Loop

print("\nFood Delivery Order Tracking Assistant")
print("Type 'quit' to exit.\n")

while True:

    user_input = input("You : ").strip()

    if user_input.lower() == "quit":
        break

    # Find Order ID

    order_match = re.search(
        r"#\d+",
        user_input
    )

    if order_match:

        order_id = order_match.group()

        queried_orders.add(order_id)

        order_status = orders.get(
            order_id,
            "Order ID not found."
        )

        prompt_input = (
            f"{user_input}\n"
            f"Order Status: {order_status}"
        )

    else:

        prompt_input = (
            f"{user_input}\n"
            "No order ID was provided."
        )

    # Generate Response

    response = conversation.predict(
        input=prompt_input
    )

    print(
        f"\nAssistant : {response}\n"
    )
    
# Session Summary

print("\nSession Summary")
print("-" * 40)

print(
    f"Unique Order IDs Queried : {len(queried_orders)}"
)

if len(queried_orders) > 0:

    print(
        "Orders Checked :",
        ", ".join(
            sorted(queried_orders)
        )
    )

else:

    print(
        "No order IDs were queried."
    )

print("\nThank you for using the Food Delivery Order Tracking Assistant.")