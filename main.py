
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from pathlib import Path
import os
import re


# ============================================================
# 1. SETTINGS
# ============================================================

PDF_FOLDER = Path("documents")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Gemini model
LLM_MODEL = "gemini-3.6-flash"

CHUNK_SIZE = 700
TOP_K = 5


# ============================================================
# 2. CONNECT TO GEMINI
# ============================================================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    print("GEMINI_API_KEY is not set.")
    print("Set it in PowerShell before running the program.")
    exit()

client = genai.Client(api_key=api_key)


# ============================================================
# 3. READ PDF FILES
# ============================================================

pdf_files = list(PDF_FOLDER.glob("*.pdf"))

print("Number of PDF files:", len(pdf_files))

if not pdf_files:
    print("No PDF files found in the documents folder.")
    exit()


# ============================================================
# 4. CLEAN PDF TEXT
# ============================================================

def clean_text(text):

    # Fix words broken across lines
    # Example:
    # worldwid
    # e
    # becomes worldwide

    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)

    # Replace multiple spaces/newlines with one space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 5. CREATE CHUNKS
# ============================================================

chunks = []
sources = []


for pdf_file in pdf_files:

    print("Reading:", pdf_file.name)

    reader = PdfReader(pdf_file)

    for page in reader.pages:

        page_text = page.extract_text()

        if not page_text:
            continue

        text = clean_text(page_text)

        if not text:
            continue

        # Split into sentences
        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        current_chunk = ""

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            # Add sentence if it fits
            if len(current_chunk) + len(sentence) + 1 <= CHUNK_SIZE:

                current_chunk += sentence + " "

            else:

                # Save current chunk
                if current_chunk.strip():

                    chunks.append(
                        current_chunk.strip()
                    )

                    sources.append(
                        pdf_file.name
                    )

                # Start new chunk
                current_chunk = sentence + " "

        # Save final chunk
        if current_chunk.strip():

            chunks.append(
                current_chunk.strip()
            )

            sources.append(
                pdf_file.name
            )


print("\n--- Chunks Created ---\n")

for i in range(len(chunks)):

    print(
        f"Chunk {i} | "
        f"Source: {sources[i]} | "
        f"Length: {len(chunks[i])}"
    )

print("\nNumber of chunks:", len(chunks))


# ============================================================
# 6. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ============================================================
# 7. CREATE CHUNK EMBEDDINGS
# ============================================================

print("Creating embeddings...")

embeddings = model.encode(
    chunks,
    normalize_embeddings=True
)

print(
    "Number of embeddings:",
    len(embeddings)
)

print(
    "Size of each embedding:",
    len(embeddings[0])
)


# ============================================================
# 8. GET USER QUESTION
# ============================================================

question = input("\nYou: ").strip()


# ============================================================
# 9. CREATE QUESTION EMBEDDING
# ============================================================

question_embedding = model.encode(
    [question],
    normalize_embeddings=True
)


# ============================================================
# 10. SEMANTIC SEARCH
# ============================================================

semantic_scores = cosine_similarity(
    question_embedding,
    embeddings
)[0]


# ============================================================
# 11. KEYWORD SEARCH
# ============================================================

def get_keywords(text):

    words = re.findall(
        r"\b[a-zA-Z0-9₹%]+\b",
        text.lower()
    )

    stop_words = {
        "what",
        "is",
        "the",
        "a",
        "an",
        "for",
        "of",
        "to",
        "in",
        "on",
        "and",
        "or",
        "are",
        "was",
        "were",
        "with",
        "does",
        "do",
        "how",
        "can",
        "will",
        "this",
        "that",
        "required",
        "please",
        "tell",
        "me"
    }

    return {
        word
        for word in words
        if word not in stop_words
        and len(word) > 1
    }


question_keywords = get_keywords(
    question
)


keyword_scores = []


for chunk in chunks:

    chunk_words = get_keywords(
        chunk
    )

    if not question_keywords:

        score = 0

    else:

        matching_words = (
            question_keywords
            .intersection(chunk_words)
        )

        score = (
            len(matching_words)
            / len(question_keywords)
        )

    keyword_scores.append(score)


# ============================================================
# 12. COMBINE SEARCH SCORES
# ============================================================

combined_scores = []


for i in range(len(chunks)):

    semantic = semantic_scores[i]

    keyword = keyword_scores[i]

    combined = (
        0.70 * semantic
        +
        0.30 * keyword
    )

    combined_scores.append(
        combined
    )


# ============================================================
# 13. RETRIEVE TOP-K CHUNKS
# ============================================================

top_indices = sorted(
    range(len(chunks)),
    key=lambda i: combined_scores[i],
    reverse=True
)[:TOP_K]


print("\n--- Retrieved Chunks ---\n")


for i in top_indices:

    print(
        "Chunk:",
        i,
        "| Source:",
        sources[i],
        "| Semantic:",
        round(
            semantic_scores[i],
            4
        ),
        "| Keyword:",
        round(
            keyword_scores[i],
            4
        ),
        "| Combined:",
        round(
            combined_scores[i],
            4
        )
    )


# ============================================================
# 14. BUILD CONTEXT
# ============================================================

relevant_context_parts = []


for i in top_indices:

    relevant_context_parts.append(
        f"""
[Source: {sources[i]} | Chunk: {i}]

{chunks[i]}
"""
    )


relevant_context = "\n".join(
    relevant_context_parts
)


# ============================================================
# 15. DISPLAY CONTEXT
# ============================================================

print(
    "\n--- Context Sent to AI ---\n"
)

print(
    relevant_context
)


# ============================================================
# 16. CREATE RAG PROMPT
# ============================================================

prompt = f"""
You are an AI Placement Assistant.

Answer the user's question using ONLY the information
contained in the provided document context.

DOCUMENT CONTEXT:
{relevant_context}

USER QUESTION:
{question}

IMPORTANT RULES:

1. Use only the document context.
2. Do not use outside knowledge.
3. Do not guess.
4. Do not invent information.
5. Do not change numbers, percentages, dates,
   salaries, CGPAs, or other factual values.
6. If the answer is explicitly present in the context,
   give that answer directly.
7. If the answer is not present in the context,
   say exactly:
   Not mentioned in the document.
8. Keep the answer short and clear.

ANSWER:
"""


# ============================================================
# 17. SEND CONTEXT TO GEMINI
# ============================================================

print("\n--- AI Answer ---\n")


interaction = client.interactions.create(
    model=LLM_MODEL,
    input=prompt
)


# ============================================================
# 18. DISPLAY ANSWER
# ============================================================

print(
    interaction.output_text
)

