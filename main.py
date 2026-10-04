from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from dotenv import load_dotenv
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import time

load_dotenv()

app = FastAPI(title="Kanoon-RAG API")
app.mount("/static", StaticFiles(directory="static"), name="static")

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
client_db = chromadb.PersistentClient(path="./chroma_db")
collection = client_db.get_or_create_collection(name="income_tax")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class Question(BaseModel):
    query: str


def ask_question(query):
    query_embedding = embed_model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=15
    )

    retrieved_chunks = results['documents'][0]
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""You are a helpful Indian tax law assistant. Answer strictly based on 
the context provided below.

RULES (follow carefully):
1. Only state facts that are explicitly written in the context. Do not add assumptions, 
   extra claims, or conclusions that are not directly stated (e.g. do not say a rule 
   "applies to a specific year" unless the context explicitly says so).
2. If the context fully answers the question, give a clear, complete answer. 
   Do NOT add any disclaimer or fallback line in this case.
3. If the context does NOT contain the answer at all, respond with ONLY this exact line 
   in the same language as the question: "I don't have confirmed information on this, 
   please consult a CA." (translate naturally if the question is in Hindi/Hinglish).
4. Never mix a correct answer with the fallback disclaimer in the same response — 
   pick one or the other.
5. When a question involves tax liability or "how much tax" at a specific income 
   level, check the context for BOTH the slab rate AND any applicable rebate 
   (such as Section 87A). If both are present in the context, explain how they 
   work together, since the rebate can make tax payable effectively zero even 
   when the slab shows a non-zero rate at that income level.

LANGUAGE RULE:
Match the language of your response to the user's question:
- If the question is in Hindi, answer in Hindi.
- If the question is in Hinglish (Hindi+English mix), answer in Hinglish.
- If the question is in English, answer in English.
- If the user explicitly asks for a specific language (e.g. "answer in English", 
  "hindi mein batao"), follow that instruction instead of the above default.

FORMATTING RULE:
Do not use markdown formatting such as **bold**, bullet symbols (-, *), or headers. 
Write in plain text only, using simple line breaks to separate points if needed.

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

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            answer = response.choices[0].message.content
            if not answer or not answer.strip():
                raise ValueError("Empty response from model")
            return answer
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return "Sorry, could not connect to the server right now. Please try again later."


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/ask")
def ask(question: Question):
    answer = ask_question(question.query)
    return {
        "question": question.query,
        "answer": answer
    }
