# Task 1 - Structured Prompt Builder for Food Delivery Chatbot

# System prompt
system_prompt = (
    "You are a food delivery support agent. "
    "Be professional and helpful. "
    "Your response should not be more than 80 words."
)

# User message template
user_template = """
Customer Name: {customer_name}
Order ID: {order_id}
Issue Type: {issue_type}
"""

# Allowed issue types
allowed_issues = ["late delivery", "missing item", "wrong item"]


# Function to validate issue type
def validate_issue(issue_type):
    if issue_type.lower() not in allowed_issues:
        raise ValueError(
            "Invalid issue type. Allowed values are: "
            "'late delivery', 'missing item', 'wrong item'."
        )


# Function to create prompt
def build_prompt(customer_name, order_id, issue_type):

    validate_issue(issue_type)

    user_prompt = user_template.format(
        customer_name=customer_name,
        order_id=order_id,
        issue_type=issue_type
    )

    print("=" * 50)
    print("SYSTEM PROMPT")
    print(system_prompt)

    print("\nUSER PROMPT")
    print(user_prompt)
    print("=" * 50)
    print()


# Test cases
try:
    build_prompt("Riya", "FD1001", "late delivery")
    build_prompt("Hiral", "FD1002", "missing item")

    # Invalid test case
    build_prompt("Neha", "FD1003", "payment issue")

except ValueError as e:
    print("Error:", e)