# Task 2 - Few Shot Complaint Classifier Prompt Builder

# List of labelled examples
examples = [
    {"text": "My order arrived 45 minutes late.", "label": "Late Delivery"},
    {"text": "I ordered a pizza but received a burger.", "label": "Wrong Item"},
    {"text": "My drink was missing from the order.", "label": "Missing Item"},
    {"text": "The food was cold and tasted bad.", "label": "Poor Quality"}
]


# Function to add a new example
def add_example(text, label):
    examples.append({"text": text, "label": label})
    print("\nNew example added successfully!")
    print("Total examples:", len(examples))


# Function to display all examples
def show_examples():
    print("\nCurrent Training Examples")
    print("-" * 40)

    for i, ex in enumerate(examples, start=1):
        print(f"{i}. Input : {ex['text']}")
        print(f"   Output: {ex['label']}\n")


# Function to build few-shot prompt
def build_few_shot_prompt(complaint_text):

    prompt = ""
    prompt += "You are a complaint classifier.\n"
    prompt += "Choose only one category from:\n"
    prompt += "Late Delivery, Wrong Item, Missing Item, Poor Quality\n\n"

    for ex in examples:
        prompt += f"Input: {ex['text']}\n"
        prompt += f"Output: {ex['label']}\n\n"

    prompt += f"Input: {complaint_text}\n"
    prompt += "Output:"

    return prompt


# Display original examples
show_examples()


# Test 1
print("\nPrompt 1")
print("=" * 60)
print(build_few_shot_prompt("My food reached almost one hour late."))


# Add one more example
add_example(
    "The restaurant forgot to pack my dessert.",
    "Missing Item"
)


# Display updated examples
show_examples()


# Test 2
print("\nPrompt 2")
print("=" * 60)
print(build_few_shot_prompt("The fries were soggy and not fresh."))