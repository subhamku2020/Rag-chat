from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle, os

# Silence tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Initialize DeepSeek client ---
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

# --- Load .txt and split into chunks ---
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [text[i:i+500] for i in range(0, len(text), 500)]

# --- Paths for caching ---
INDEX_PATH = "cached.index"
CHUNKS_PATH = "chunks.pkl"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Load or create embeddings/index ---
if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
    print("🔁 Loading cached embeddings...")
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
else:
    print("⚙️ Creating embeddings for first time...")
    chunks = load_text("/Users/i540458/Desktop/AI/rag-chat/doc.txt")
    embeddings = embed_model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

print(f"✅ Loaded {len(chunks)} chunks into FAISS index")

# --- Ask function ---
def ask(question, top_k=2):
    q_embed = embed_model.encode([question])
    D, I = index.search(np.array(q_embed).astype("float32"), top_k)

    if D[0][0] > 1.0:
        return "❌ Sorry, I couldn't find that in the document."

    best_chunks = [chunks[i] for i in I[0]]
    context = "\n\n".join(best_chunks)
    prompt = f"Use ONLY the following text to answer:\n\n{context}\n\nQuestion: {question}"

    completion = client.chat.completions.create(
        model="deepseek/deepseek-chat-v3.1",
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message.content.strip()

# --- Example run ---
if __name__ == "__main__":
    print("📘 Document loaded. Ask me questions (type 'exit' to quit).")
    while True:
        q = input("\nAsk: ")
        if q.lower() in ("exit", "quit"):
            break
        print("\n🤖", ask(q))
