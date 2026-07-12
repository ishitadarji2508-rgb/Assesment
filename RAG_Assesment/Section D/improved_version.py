from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Read policy file
with open("Policy.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Split text into fixed-size chunks
words = text.split()
chunks = []
chunk_size = 100

for i in range(0, len(words), chunk_size):
    chunk = " ".join(words[i:i + chunk_size])
    chunks.append(chunk)

print("Total Chunks:", len(chunks))

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = model.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

print("\nFood Delivery RAG System Started")

while True:

    question = input("\nEnter your question (type 'quit' to exit): ").strip()

    if question.lower() == "quit":
        print("Program Closed")
        break

    if question == "":
        print("Please enter a valid question.")
        continue

    # Encode user question
    query_embedding = model.encode([question])
    query_embedding = np.array(query_embedding).astype("float32")

    # Search top 2 relevant chunks
    k = min(2, len(chunks))
    distances, indices = index.search(query_embedding, k)

    print("\nRetrieved Chunks")
    print("-" * 50)

    retrieved_chunks = []

    for index_no in indices[0]:
        retrieved_chunks.append(chunks[index_no])
        print(chunks[index_no])
        print()

    # Build RAG Prompt
    prompt = f"""
System:
You are a helpful AI assistant.

Context 1:
{retrieved_chunks[0]}

Context 2:
{retrieved_chunks[1]}

User Question:
{question}

Answer:
Use only the information provided in the context.
If the answer is not available in the context, reply:
"I don't know."
"""

    print("-" * 50)
    print("\nStructured RAG Prompt")
    print(prompt)

    print("Simulated Answer:")
    print("This is a placeholder answer based on the retrieved context.")