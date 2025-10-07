import os
import re
import random
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_ollama import OllamaLLM
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
from transformers import pipeline
from rag import auto_ingest_data_folder


OLLAMA_URL = "http://127.0.0.1:11434"
SESSION_ID = "user1-session"

# Make langdetect deterministic
DetectorFactory.seed = 0

# Allowed languages
SUPPORTED_LANGS = ["en", "hi", "mr", "gu", "ta", "te"]

# -----------------------------
# Prompts
# -----------------------------
REWRITE_PROMPT = """You are a campus assistant whose job is to rewrite a user's follow-up question into a 
fully self-contained English question that can be answered without additional context.

Context: the assistant serves a college. It answers about fees, scholarships, admissions, timetables, exams, 
hostels, certificates, backlog rules, exam forms, and other student services.

Conversation history:
{history}

User follow-up:
{question}

Return exactly one rewritten standalone question in plain English.
"""

CLARIFY_PROMPT = """You are an assistant that checks if the user's rewritten question still lacks 
a key detail (semester, department, scholarship type, admission type, exam type, etc.).

Rewritten question: "{question}"
Retrieved snippets:
{snippets}

If a missing detail is detected, suggest a short clarifying question (≤12 words).
If no clarification is needed, return empty string.

Respond in English only.
"""

# -----------------------------
# Setup Memory & Redis
# -----------------------------
redis_history = RedisChatMessageHistory(
    url="redis://localhost:6379/0",
    session_id=SESSION_ID
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    chat_memory=redis_history,
    return_messages=True
)

# -----------------------------
# LLM and Retriever
# -----------------------------
llm = OllamaLLM(model="mistral", base_url=OLLAMA_URL)

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data")
retriever = auto_ingest_data_folder(data_path)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer"
)

# -----------------------------
# Sentiment Detection
# -----------------------------
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

def detect_sentiment(user_input: str) -> str:
    try:
        result = sentiment_analyzer(user_input[:512])[0]
        label = result["label"].lower()
        if "1" in label or "2" in label:
            return "negative"
        elif "3" in label:
            return "neutral"
        else:
            return "positive"
    except Exception:
        return "neutral"

# -----------------------------
# Empathy Responses
# -----------------------------
MILD_EMPATHY = [
    "I understand this is stressful. Please don’t worry, I’ll help you with the correct information.",
    "I can see this is causing you concern. Let’s go through it step by step together.",
    "I hear your worry. Don’t panic, I’ll guide you with the right details.",
    "It sounds like you’re stressed. I’m here to support you and provide clear answers.",
    "I know this must feel overwhelming, but you’re not alone—I’ll help you find the solution."
]

SEVERE_EMPATHY = [
    "This sounds very stressful. Please take a deep breath—I’ll guide you now.",
    "I understand you’re really worried. Don’t panic, I’ll provide the details you need.",
    "I hear your concern about finances. Please also reach out to the official scholarship helpline for assistance.",
    "I know this situation feels urgent. I’ll give you the correct info, and you can also contact the admin office for support.",
    "It must be tough to handle this. I’ll share the deadlines, but please also talk to student support if needed."
]

def get_empathy_message(user_input: str) -> str:
    severe_keywords = ["worried", "tension", "stress", "can’t afford", "panic", "afraid", "चिंता", "परेशान", "डर"]
    if any(kw in user_input.lower() for kw in severe_keywords):
        return random.choice(SEVERE_EMPATHY)
    return random.choice(MILD_EMPATHY)

# -----------------------------
# Slot Extraction
# -----------------------------
def extract_slots(text):
    slots = []
    date_pattern = r"(\d{1,2}\s?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s?\d{2,4})"
    slots += re.findall(date_pattern, text, flags=re.IGNORECASE)
    sem_pattern = r"\b(FE|SE|TE|BE)\b"
    slots += re.findall(sem_pattern, text)
    keywords = ["UG", "PG", "merit", "need-based", "backlog", "supplementary"]
    for kw in keywords:
        if re.search(rf"\b{kw}\b", text, flags=re.IGNORECASE):
            slots.append(kw)
    return slots

# -----------------------------
# Helpers
# -----------------------------
def llm_rewrite_followup(history_text: str, question: str) -> str:
    prompt = REWRITE_PROMPT.format(history=history_text, question=question)
    try:
        response = llm.invoke(prompt)
        return str(response).strip().splitlines()[0]
    except Exception:
        return question

def llm_check_need_clarify(rewritten_question: str, snippets: list) -> str:
    try:
        snippet_texts = [s.page_content[:400].replace("\n", " ") for s in snippets]
        all_slots = [extract_slots(txt) for txt in snippet_texts]
        flat_slots = [s for sub in all_slots for s in sub]

        if flat_slots and len(set(flat_slots)) == 1:
            return ""

        snippet_text = "\n---\n".join(snippet_texts)
        prompt = CLARIFY_PROMPT.format(question=rewritten_question, snippets=snippet_text)
        response = llm.invoke(prompt)
        resp_text = str(response).strip()
        return resp_text if resp_text and "?" in resp_text else ""
    except Exception:
        return ""

def get_history_as_text() -> str:
    try:
        messages = memory.chat_memory.messages
        lines = []
        for m in messages[-10:]:
            content = getattr(m, "content", str(m))
            speaker = "User" if "user" in str(m).lower() or "human" in str(m).lower() else "Assistant"
            lines.append(f"{speaker}: {content}")
        return "\n".join(lines)
    except Exception:
        return ""

# -----------------------------
# Chat Loop
# -----------------------------
print("🌍 Context-aware Multilingual Campus Chatbot ready. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit", "bye"):
        print("👋 Goodbye!")
        break
    if user_input.lower() in ("hii","hello"):
        print("Hello, How can I help you?")
    if user_input.lower() in ("Thanks","Thank you"):
        print("You’re very welcome! 🎉 Feel free to ask")

    try:
        # Detect language safely
        try:
            user_lang = detect(user_input)
        except Exception:
            user_lang = "en"

        if user_lang not in SUPPORTED_LANGS:
            user_lang = "en"

        # Sentiment detection
        sentiment = detect_sentiment(user_input)
        if sentiment == "negative":
            empathy_msg = get_empathy_message(user_input)
            reply = GoogleTranslator(source="en", target=user_lang).translate(empathy_msg)
            print(f"🤖 {reply}")

        # Translate query to English
        query_en = GoogleTranslator(source="auto", target="en").translate(user_input)

        # Rewrite follow-up
        hist_text = get_history_as_text()
        rewritten_q = llm_rewrite_followup(hist_text, query_en)

        # Run RAG chain
        result = qa_chain.invoke({"question": rewritten_q})
        answer_en = result.get("answer", "").strip()
        retrieved_docs = result.get("source_documents", [])

        if not retrieved_docs or not answer_en:
            fallback = "This information is not in the official documents."
            reply = GoogleTranslator(source="en", target=user_lang).translate(fallback)
            print(f"🤖 {reply}")
            continue

        # Clarify if needed
        clarify_q = llm_check_need_clarify(rewritten_q, retrieved_docs)

        # Translate answer back
        final_answer = GoogleTranslator(source="en", target=user_lang).translate(answer_en)
        if clarify_q:
            clarify_text = GoogleTranslator(source="en", target=user_lang).translate(clarify_q)
            final_answer = f"{final_answer}\nℹ️ {clarify_text}"

        sources = list({d.metadata.get("source", "unknown") for d in retrieved_docs})
        print(f"🤖 {final_answer}")
        if sources:
            print(f"📖 Sources: {sources}")

    except Exception as e:
        print(f"⚠️ Error: {e}")
