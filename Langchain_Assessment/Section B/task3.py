import streamlit as st
from langchain_ollama import OllamaLLM

# Page Configuration

st.set_page_config(
    page_title="QuickBite AI",
    page_icon="🍔"
)

st.title("🍽️ QuickBite Dish Description Generator")
# Load LLM
llm = OllamaLLM(model="llama3.2")

# Session State

if "description" not in st.session_state:
    st.session_state.description = ""

# User Inputs
dish = st.text_input(
    "Dish Name",
    placeholder="Paneer Tikka Masala"
)

cuisine = st.text_input(
    "Cuisine Type",
    placeholder="North Indian"
)

length = st.selectbox(
    "Description Length",
    ["Short", "Medium", "Long"]
)

# Prompt Builder

def create_prompt():

    prompt = f"""
You are an expert food marketing writer.

Generate a {length.lower()} promotional description.

Dish Name:
{dish}

Cuisine Type:
{cuisine}

Instructions:
- Write in an appetising customer-facing tone.
- Mention flavour, aroma and freshness.
- Encourage customers to order.
- Suitable for a food delivery application.
"""

    return prompt
# Generate Description

def generate_text():

    prompt = create_prompt()

    print("\n========== PROMPT ==========")
    print(prompt)
    print("============================\n")

    with st.spinner("Generating description…"):

        output = llm.invoke(prompt)

    st.session_state.description = output
# Generate Button
if st.button("Generate Description"):

    if dish.strip() == "" or cuisine.strip() == "":
        st.warning("Please fill all fields.")
    else:
        generate_text()

# Regenerate Button

if st.session_state.description:

    if st.button("Regenerate"):
        generate_text()

# Output Container

if st.session_state.description:

    with st.container(border=True):

        st.subheader("AI Generated Description")

        st.write(st.session_state.description)

    st.write(
        f"Character Count: {len(st.session_state.description)}"
    )