# Task 4 - RAG Powered Food Delivery Policy Q&A Pipeline

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Food Delivery Policy (300+ words)

policy_text = """
Food Delivery Policy

Customers can request a refund if they receive a missing item, damaged food,
or an incorrect order. Refund requests should normally be made within
24 hours after the order is delivered. The support team reviews each request
before approving the refund. If the complaint is verified, the refund is
processed using the customer's original payment method. Processing may take
5 to 7 business days depending on the payment provider.

Orders can be cancelled without any charges if the restaurant has not started
preparing the food. Once food preparation begins, cancellation may not be
possible. In some situations, a partial refund may be provided depending on
the preparation stage and restaurant policy.

The expected delivery window is between 30 and 45 minutes. Delivery times may
increase because of heavy traffic, bad weather, public holidays or high demand.
Customers can track the delivery partner using the mobile application.

If an item is unavailable, the restaurant may contact the customer to offer
a replacement. If the customer does not agree with the replacement, the item
will be removed from the order and the corresponding amount will be refunded.

Delivery partners are expected to handle food carefully and follow all safety
guidelines. Customers should verify the order at the time of delivery and
report any issues immediately through the Help section of the application.

Repeated false refund requests may lead to account verification or temporary
restrictions. Every complaint is reviewed individually to ensure fairness for
both customers and restaurant partners.

Support services are available throughout the day using live chat, email and
phone support. Customers should always keep their order ID while contacting
support because it helps the support team verify the order quickly.

Refunds for missing items are usually approved after verification. Refunds for
late delivery are considered only if the delay is significant and caused by
the delivery process instead of unavoidable situations like weather conditions.
"""

# Function to split text into chunks

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


chunks = chunk_text(policy_text)

print("=" * 60)
print("Total Chunks Created:", len(chunks))
print("=" * 60)

# Create Embeddings

print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_embeddings = model.encode(chunks)
chunk_embeddings = np.array(chunk_embeddings, dtype="float32")

# Build FAISS Index

dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

print("FAISS Index Ready")
print("Total Chunks Indexed:", index.ntotal)

print("\nPipeline Summary")
print("-" * 30)
print("Chunks Created :", len(chunks))
print("Embeddings     :", len(chunk_embeddings))
print("Index Size     :", index.ntotal)

# Retrieve Function

def retrieve(query, k=3):

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype="float32")

    distances, indices = index.search(query_embedding, k)

    retrieved = []

    for idx in indices[0]:
        retrieved.append(chunks[idx])

    return retrieved

# Build RAG Prompt

def build_rag_prompt(query, retrieved_chunks):

    prompt = ""
    prompt += "System Role:\n"
    prompt += "You are a helpful food delivery support assistant.\n\n"

    prompt += "Retrieved Context:\n"

    for i, chunk in enumerate(retrieved_chunks, start=1):
        prompt += f"\nContext {i}:\n"
        prompt += chunk + "\n"

    prompt += "\nUser Question:\n"
    prompt += query + "\n\n"

    prompt += (
        "Answer only using the provided context. "
        "If the answer is not available, reply with 'I don't know.'"
    )

    return prompt

# Test the Pipeline


query = "What is the refund policy for missing items?"

retrieved_chunks = retrieve(query)

print("\n")
print("=" * 60)
print("Retrieved Chunks")
print("=" * 60)

for i, chunk in enumerate(retrieved_chunks, start=1):
    print(f"\nChunk {i}")
    print("-" * 50)

    # Show only first 300 characters for better readability
    preview = chunk[:300]

    if len(chunk) > 300:
        preview += "..."

    print(preview)

rag_prompt = build_rag_prompt(query, retrieved_chunks)

print("\n")
print("=" * 60)
print("Final RAG Prompt")
print("=" * 60)

print(rag_prompt)

print("\n")
print("=" * 60)
print("Simulated Answer")
print("=" * 60)

print("Refunds for missing items are generally approved after verification by the support team.")