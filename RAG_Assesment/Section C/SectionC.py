from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from datetime import datetime


policy_text = """
Food Delivery Policy

Customers can request a refund if they receive a missing item, damaged food, incorrect food, or if an item is not delivered. Refund requests should normally be submitted within 24 hours after the order is delivered. Every refund request is verified by the customer support team. If the complaint is found to be genuine, the refund is processed using the original payment method. Depending on the payment provider, the refund may take between five and seven business days to reflect in the customer's account.

Orders can be cancelled without any cancellation charges if the restaurant has not yet started preparing the food. Once the restaurant begins preparing the order, cancellation may not be possible. In some situations, only a partial refund may be approved depending on how much of the order has already been prepared.

The normal delivery time is between thirty and forty-five minutes. However, delivery may take longer because of heavy traffic, bad weather, festivals, road closures, or a high number of orders. Customers can track their delivery partner using the mobile application.

If a restaurant is unable to provide a menu item, it may contact the customer and suggest a replacement. The replacement will only be sent after receiving the customer's approval. If the customer rejects the replacement, that item will be removed from the bill and the amount will be refunded.

Customers should always check the package at the time of delivery. If food is damaged or any item is missing, it should be reported immediately using the Help section of the application. The support team investigates every complaint carefully before taking action.

Delivery partners are trained to handle food safely and follow company guidelines. Customers should always provide the correct delivery address and contact number. Incorrect delivery details may result in delays or failed deliveries.

Repeated false refund requests may lead to additional account verification or temporary account restrictions. Every complaint is reviewed fairly to protect both customers and restaurant partners.

Customer support is available through live chat, email and phone support throughout the day. Customers should keep their order ID ready while contacting support because it helps the team locate the order quickly.

Refunds for missing items are generally approved after successful verification. Refunds for late delivery are considered only when the delay is caused by the delivery process and not by unavoidable situations such as heavy rain or severe traffic conditions.

The company regularly updates its delivery policies to improve customer satisfaction and maintain fair service standards for customers, delivery partners and restaurant partners.
"""


def chunk_text(text, chunk_size=100, overlap=20):

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size
        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start = end - overlap

    return chunks


policy_chunks = chunk_text(policy_text)

print("\nFood Delivery Help Desk Chatbot")
print("-" * 40)
print("Chunks Created :", len(policy_chunks))


print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully.")

print("Creating vector embeddings for policy chunks...")

chunk_embeddings = model.encode(policy_chunks)
chunk_embeddings = np.array(chunk_embeddings, dtype="float32")


dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

print("\nFAISS Index Created Successfully")

print("\nPipeline Summary")
print("-" * 25)
print("Chunks      :", len(policy_chunks))
print("Embeddings  :", len(chunk_embeddings))
print("Index Size  :", index.ntotal)

examples = [
    {
        "text": "My order arrived 40 minutes late.",
        "label": "Late Delivery"
    },
    {
        "text": "I received a burger instead of pizza.",
        "label": "Wrong Item"
    },
    {
        "text": "My cold drink was missing from the order.",
        "label": "Missing Item"
    },
    {
        "text": "The food was cold and tasted bad.",
        "label": "Poor Quality"
    }
]


policy_questions = 0
complaints_classified = 0


def retrieve(query, k=3):

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype="float32")

    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = []

    for idx in indices[0]:
        retrieved_chunks.append(policy_chunks[idx])

    return retrieved_chunks


def build_rag_prompt(query, retrieved_chunks):

    prompt = "System Role:\n"
    prompt += "You are a helpful food delivery support assistant.\n\n"

    prompt += "Retrieved Context:\n"

    for i, chunk in enumerate(retrieved_chunks, start=1):

        prompt += f"\nContext {i}:\n"
        prompt += chunk + "\n"

    prompt += "\nUser Question:\n"
    prompt += query + "\n\n"

    prompt += "Answer only from the given context. "
    prompt += "If the answer is not available, reply with 'I don't know'."

    return prompt


def classify_complaint(text):

    text = text.lower()

    if "late" in text or "delay" in text:
        return "Late Delivery", examples[0]["text"]

    elif "wrong" in text or "instead" in text:
        return "Wrong Item", examples[1]["text"]

    elif "missing" in text or "not received" in text:
        return "Missing Item", examples[2]["text"]

    else:
        return "Poor Quality", examples[3]["text"]


def save_log(mode, query, output):

    with open("session_log.txt", "a", encoding="utf-8") as file:

        file.write(f"Time : {datetime.now()}\n")
        file.write(f"Mode : {mode}\n")
        file.write(f"Query : {query}\n")
        file.write(f"Output : {output}\n")
        file.write("-" * 50 + "\n")


print("\nSystem is ready.")
print("Policy search and complaint classifier loaded successfully.")

while True:

    print("\n")
    print("=" * 45)
    print("Food Delivery Help Desk")
    print("=" * 45)
    print("1. Ask a Policy Question")
    print("2. Classify a Complaint")
    print("3. Exit")

    choice = input("\nEnter your choice: ").strip()

    if choice == "1":

        question = input("\nEnter your policy question: ").strip()

        if question == "":
            print("Question cannot be empty.")
            continue

        policy_questions += 1

        retrieved_chunks = retrieve(question)

        print("\nRetrieved Context")
        print("-" * 40)

        for i, chunk in enumerate(retrieved_chunks, start=1):
            print(f"\nContext {i}")
            print(chunk[:250] + "...")

        rag_prompt = build_rag_prompt(question, retrieved_chunks)

        print("\nFinal RAG Prompt")
        print("-" * 40)
        print(rag_prompt)

        answer = "Based on the retrieved policy, the answer has been generated from the available context."

        print("\nSimulated Answer")
        print("-" * 40)
        print(answer)

        save_log("RAG", question, answer)

    elif choice == "2":

        complaint = input("\nEnter your complaint: ").strip()

        if complaint == "":
            print("Complaint cannot be empty.")
            continue

        complaints_classified += 1

        category, example = classify_complaint(complaint)

        print("\nFew-Shot Prompt")
        print("-" * 40)

        for ex in examples:
            print(f"Input : {ex['text']}")
            print(f"Output: {ex['label']}\n")

        print(f"Input : {complaint}")
        print("Output :")

        print("\nPredicted Category :", category)
        print("Closest Example    :", example)

        save_log("Classify", complaint, category)

    elif choice == "3":

        print("\n")
        print("=" * 45)
        print("Session Summary")
        print("=" * 45)
        print("Policy Questions Asked :", policy_questions)
        print("Complaints Classified  :", complaints_classified)
        print("\nSession log saved successfully.")
        print("Thank You!")

        break

    else:
        print("Invalid choice. Please enter 1, 2 or 3.")