# QuickBite AI
# Intelligent Food Delivery Assistant

import json
import streamlit as st

from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

from prompt import system_prompt
from tools import get_delivery_estimate

# Page Configuration

st.set_page_config(
    page_title="QuickBite AI",
    layout="wide"
)

st.title("QuickBite AI")
st.subheader("Intelligent Food Delivery Assistant")

# Load LLM

llm = ChatOllama(
    model="llama3.2",
    temperature=0.3
)

# Load Menu

with open("menu.json", "r") as file:
    menu_data = json.load(file)

# Session State

if "messages" not in st.session_state:
    st.session_state.messages = []

if "address" not in st.session_state:
    st.session_state.address = ""

if "diet" not in st.session_state:
    st.session_state.diet = "Vegetarian"

if "memory" not in st.session_state:

    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# Sidebar


with st.sidebar:

    st.header("Customer Details")

    st.session_state.address = st.text_input(
        "Delivery Address",
        value=st.session_state.address
    )

    st.session_state.diet = st.selectbox(
        "Dietary Preference",
        [
            "Vegetarian",
            "Non-Vegetarian"
        ]
    )

    st.divider()

    st.write("Current Details")

    st.write(
        f"Address : {st.session_state.address}"
    )

    st.write(
        f"Dietary Preference : {st.session_state.diet}"
    )

    st.divider()

    if st.button("Clear Chat"):

        st.session_state.messages.clear()

        st.session_state.memory.clear()

        st.rerun()

# Menu Search Function

def search_menu(user_query):

    query = user_query.lower()

    matching_items = []

    for item in menu_data:

        # Vegetarian Filter

        if (
            st.session_state.diet == "Vegetarian"
            and item["veg"] == False
        ):
            continue

        # Cuisine Match

        if item["cuisine"].lower() in query:

            matching_items.append(item)

            continue

        # Dish Name Match

        if item["name"].lower() in query:

            matching_items.append(item)

    # If nothing matches

    if len(matching_items) == 0:

        for item in menu_data:

            if (
                st.session_state.diet == "Vegetarian"
                and item["veg"] == False
            ):
                continue

            matching_items.append(item)

    return matching_items[:3]

# Build Menu Context

def build_menu_context(user_query):

    recommendations = search_menu(user_query)

    menu_context = ""

    for item in recommendations:

        menu_context += (
            f"- {item['name']} | "
            f"{item['cuisine']} | "
            f"₹{item['price']}\n"
        )

    return menu_context

# Prompt Template

prompt = ChatPromptTemplate.from_messages(

    [

        (
            "system",

            system_prompt +

            """

Customer Address:
{address}

Dietary Preference:
{diet}

Recommended Menu:

{menu}

Use the customer's address and dietary preference while answering.

If the customer asks about delivery time,
always use the get_delivery_estimate tool before replying.

"""
        ),

        MessagesPlaceholder(
            variable_name="chat_history"
        ),

        ("human", "{input}"),

        MessagesPlaceholder(
            variable_name="agent_scratchpad"
        )

    ]

)

# Create Agent

tools = [
    get_delivery_estimate
]

agent = create_tool_calling_agent(

    llm,
    tools,
    prompt

)

agent_executor = AgentExecutor(

    agent=agent,
    tools=tools,
    verbose=False

)

# Chat Input

user_query = st.chat_input(
    "Ask anything about food or delivery..."
)

# Process User Query

if user_query:

    # Save User Message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    # Get Menu Recommendations

    menu_context = build_menu_context(user_query)

    # Agent Input

    agent_input = {
        "input": user_query,
        "address": st.session_state.address,
        "diet": st.session_state.diet,
        "menu": menu_context,
        "chat_history": st.session_state.memory.chat_memory.messages
    }

    # Generate Response

    with st.spinner("Generating response..."):

        try:

            response = agent_executor.invoke(agent_input)

            assistant_reply = response["output"]

        except Exception as error:

            assistant_reply = (
                "Sorry, something went wrong.\n\n"
                f"{error}"
            )

    # Save Conversation in Memory

    st.session_state.memory.chat_memory.add_user_message(
        user_query
    )

    st.session_state.memory.chat_memory.add_ai_message(
        assistant_reply
    )

    # Save Assistant Response

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_reply
        }
    )

    st.rerun()
# Display Conversation

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

# Footer

st.divider()

st.caption(
    "QuickBite AI | Intelligent Food Delivery Assistant"
)