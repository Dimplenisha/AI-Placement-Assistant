from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from pathlib import Path
import os
import re


# ============================================================
# SETTINGS
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-3.6-flash"

CHUNK_SIZE = 700
TOP_K = 5


# ============================================================
# CONNECT TO GEMINI
# ============================================================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is not set. "
        "Set it in PowerShell before running the application."
    )

client = genai.Client(api_key=api_key)


# ============================================================
# CLEAN PDF TEXT
# ============================================================

def clean_text(text):

    # Fix words broken across lines
    text = re.sub(
        r"(\w)-\s+(\w)",
        r"\1\2",
        text
    )

    # Remove extra spaces/newlines
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# CREATE CHUNKS FROM PDF
# ============================================================

def create_chunks_from_pdf(pdf_path):

    chunks = []
    sources = []

    reader = PdfReader(pdf_path)

    for page in reader.pages:

        page_text = page.extract_text()

        if not page_text:
            continue

        text = clean_text(page_text)

        if not text:
            continue

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        current_chunk = ""

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            if (
                len(current_chunk)
                + len(sentence)
                + 1
                <= CHUNK_SIZE
            ):

                current_chunk += sentence + " "

            else:

                if current_chunk.strip():

                    chunks.append(
                        current_chunk.strip()
                    )

                    sources.append(
                        Path(pdf_path).name
                    )

                current_chunk = sentence + " "

        # Save final chunk
        if current_chunk.strip():

            chunks.append(
                current_chunk.strip()
            )

            sources.append(
                Path(pdf_path).name
            )

    return chunks, sources


# ============================================================
# BUILD RAG INDEX
# ============================================================

def build_rag_index(pdf_files):

    chunks = []
    sources = []

    # ----------------------------------------
    # Read all PDFs
    # ----------------------------------------

    for pdf_file in pdf_files:

        pdf_chunks, pdf_sources = (
            create_chunks_from_pdf(pdf_file)
        )

        chunks.extend(pdf_chunks)
        sources.extend(pdf_sources)

    if not chunks:

        raise ValueError(
            "No readable text was found in the PDF files."
        )

    # ----------------------------------------
    # Load embedding model
    # ----------------------------------------

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # ----------------------------------------
    # Create embeddings
    # ----------------------------------------

    embeddings = embedding_model.encode(
        chunks,
        normalize_embeddings=True
    )

    return {
        "chunks": chunks,
        "sources": sources,
        "embeddings": embeddings,
        "embedding_model": embedding_model
    }


# ============================================================
# KEYWORD EXTRACTION
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


# ============================================================
# ASK QUESTION
# ============================================================

def ask_question(question, rag_index):

    chunks = rag_index["chunks"]
    sources = rag_index["sources"]
    embeddings = rag_index["embeddings"]
    embedding_model = rag_index["embedding_model"]

    # ========================================================
    # QUESTION EMBEDDING
    # ========================================================

    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    # ========================================================
    # SEMANTIC SEARCH
    # ========================================================

    semantic_scores = cosine_similarity(
        question_embedding,
        embeddings
    )[0]

    # ========================================================
    # KEYWORD SEARCH
    # ========================================================

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

    # ========================================================
    # COMBINE SCORES
    # ========================================================

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

    # ========================================================
    # TOP-K RETRIEVAL
    # ========================================================

    top_indices = sorted(
        range(len(chunks)),
        key=lambda i: combined_scores[i],
        reverse=True
    )[:TOP_K]

    # ========================================================
    # BUILD CONTEXT
    # ========================================================

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

    # ========================================================
    # RAG PROMPT
    # ========================================================

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

    # ========================================================
    # SEND TO GEMINI
    # ========================================================

    interaction = client.interactions.create(
        model=LLM_MODEL,
        input=prompt
    )

    answer = interaction.output_text

    # ========================================================
    # SOURCE INFORMATION
    # ========================================================

    retrieved_sources = []

    for i in top_indices:

        retrieved_sources.append({
            "source": sources[i],
            "chunk": i,
            "score": round(
                combined_scores[i],
                4
            )
        })

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        "answer": answer,
        "sources": retrieved_sources
    }