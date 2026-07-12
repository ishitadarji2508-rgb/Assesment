# Task 3 - Semantic Search Over Restaurant FAQs

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# Restaurant FAQ Data
faq_list = [
    "Delivery usually takes between 30 and 45 minutes.",
    "You can cancel an order before the restaurant starts preparing it.",
    "Refunds are provided for missing or damaged food items.",
    "If a menu item is unavailable, it may be replaced with a similar item after confirmation.",
    "You can contact customer support through the Help section of the app.",
    "Payments can be made using Cash, UPI, Debit Card or Credit Card."
]


# Load Sentence Transformer Model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


# Generate Embeddings
print("Generating FAQ embeddings...")
faq_embeddings = model.encode(faq_list)

# Convert embeddings into float32 for FAISS
faq_embeddings = np.array(faq_embeddings, dtype="float32")


# Create FAISS Index
dimension = faq_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(faq_embeddings)

print("\nFAISS Index Created Successfully!")
print(f"Total FAQs Stored: {index.ntotal}")


# Function to Search FAQs
def search_faq(query, k=2):

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype="float32")

    distances, indices = index.search(query_embedding, k)

    print("\n" + "=" * 60)
    print("User Query:")
    print(query)

    print("\nTop Matching FAQs")
    print("-" * 60)

    for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), start=1):

        print(f"\nResult {rank}")
        print(f"FAQ      : {faq_list[idx]}")
        print(f"Distance : {distance:.4f}")


# Test Query 1
search_faq("How do I get my money back?")

# Test Query 2
search_faq("How can I talk to customer care?")

# Test Query 3 (Extra)
search_faq("Can I stop my order after placing it?")