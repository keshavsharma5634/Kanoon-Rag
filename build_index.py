from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import os
import hashlib
import json

DATA_DIR = "data"
HASH_FILE = "file_hashes.json"

splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="income_tax")


def get_file_hash(filepath):
    """Generate a unique fingerprint based on file content."""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_stored_hashes():
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            return json.load(f)
    return {}


def save_stored_hashes(hashes):
    with open(HASH_FILE, "w") as f:
        json.dump(hashes, f, indent=2)


def process_file(filename, filepath):
    """Chunk, embed, and add a single file's content to the database."""
    with open(filepath, "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks = splitter.split_text(full_text)
    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename} for _ in chunks]
    embeddings = model.encode(chunks)

    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=ids,
        metadatas=metadatas
    )
    print(f"  Added {len(chunks)} chunks")


def remove_file_chunks(filename):
    """Delete all existing chunks belonging to a file (used before re-adding updated content)."""
    collection.delete(where={"source": filename})


stored_hashes = load_stored_hashes()
current_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]

new_hashes = {}
added, updated, skipped = 0, 0, 0

for filename in current_files:
    filepath = os.path.join(DATA_DIR, filename)
    current_hash = get_file_hash(filepath)
    new_hashes[filename] = current_hash

    if filename not in stored_hashes:
        print(f"[NEW] {filename}")
        process_file(filename, filepath)
        added += 1

    elif stored_hashes[filename] != current_hash:
        print(f"[UPDATED] {filename}")
        remove_file_chunks(filename)
        process_file(filename, filepath)
        updated += 1

    else:
        print(f"[SKIPPED] {filename} (no changes)")
        skipped += 1

# Handle files that were deleted from the data folder
deleted_files = set(stored_hashes.keys()) - set(current_files)
for filename in deleted_files:
    print(f"[REMOVED] {filename} (file no longer exists)")
    remove_file_chunks(filename)

save_stored_hashes(new_hashes)

print(f"\nSummary: {added} added, {updated} updated, {skipped} skipped, {len(deleted_files)} removed")
print("Database is up to date.")
