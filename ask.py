from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
client_db = chromadb.PersistentClient(path="./chroma_db")
collection = client_db.get_or_create_collection(name="income_tax")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def ask_question(query):
    query_embedding = embed_model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=8
    )

    retrieved_chunks = results['documents'][0]
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""You are a helpful Indian tax law assistant. Answer strictly based on 
the context provided below.

RULES (follow carefully):
1. Only state facts that are explicitly written in the context. Do not add assumptions 
   or extra claims not directly stated.
2. If the context fully answers the question, give a clear, complete answer without 
   any disclaimer.
3. If the context does NOT contain the answer, respond only with the fallback line 
   (in the same language as the question): "I don't have confirmed information on 
   this, please consult a CA."
4. Never mix a correct answer with the fallback disclaimer.

LANGUAGE RULE:
Match your response language to the user's question (Hindi, Hinglish, or English). 
If the user explicitly requests a specific language, follow that instead.

FORMATTING RULE:
Do not use markdown formatting such as **bold**, bullet symbols (-, *), or headers, 
even when explaining multi-step calculations. Write in plain text only, using 
numbered steps in sentence form (e.g., "First,... Second,... Finally,...") or 
simple line breaks instead of bullet points or dashes.

CITATION RULE:
Always include the section number in your answer when the context provides one — 
citing a section is required, not optional. Cite the current section number under 
the Income-tax Act, 2025 (e.g., "Section 123") as the primary reference. You may 
additionally mention the old section number in parentheses if it aids the user's 
understanding, e.g., "Section 123 (previously known as Section 80C)". Never cite 
only the old section number, and never omit the section number entirely.

Always mention the exact section number if available.

Context:
{context}

Question: {query}

Answer:"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    print("=" * 50)
    print("Kanoon-RAG — Income Tax Assistant (Section 80C)")
    print("Type 'quit' to exit")
    print("=" * 50)

    while True:
        question = input("\nYour question: ")

        if question.lower() in ["quit", "exit", "bye"]:
            print("Thanks!")
            break

        print("\nAnswer:")
        print(ask_question(question))