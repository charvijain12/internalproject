import streamlit as st
import pandas as pd
import os
import base64
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import logging

# LangChain imports for RAG
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- FORCE LIGHT MODE ----------
st.markdown("""
<style>
:root {
    color-scheme: light !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD ENV ----------
load_dotenv()

# ---------- DIRECTORIES ----------
POLICY_DIR = os.path.join(os.path.dirname(__file__), "policies")
FAISS_INDEX_PATH = "faiss_index/org_policy"
QUERY_FILE = "queries.csv"

os.makedirs(POLICY_DIR, exist_ok=True)
os.makedirs("faiss_index", exist_ok=True)

if not os.path.exists(QUERY_FILE):
    pd.DataFrame(columns=["timestamp", "context", "question", "answer", "mode"]).to_csv(QUERY_FILE, index=False)

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Escobar Consultancy - Policy Insights",
    page_icon="💼",
    layout="wide"
)

# ---------- CSS ----------
st.markdown("""
<style>
[data-testid="stSidebarContent"] {
    background: linear-gradient(180deg, #E6E6FA, #FFF8E1) !important;
    padding-top: 25px !important;
    border-right: 1px solid #e0e0e0;
}

.menu-item {
    padding: 10px 16px;
    border-radius: 10px;
    margin-bottom: 10px;
    cursor: pointer;
    display: block;
    font-size: 16px;
    font-weight: 500;
    color: black;
    background: rgba(255,255,255,0.45);
    text-decoration: none;
    transition: 0.2s ease;
}

.header {
    background: linear-gradient(90deg, #E6E6FA, #FFF8E1);
    padding: 20px;
    border-radius: 10px;
    color: black;
    text-align: center;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
    margin-bottom: 30px;
}

.card {
    background: white;
    border: 1px solid #e8e8e8;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

.chat-bubble-user {
    background-color: #E8DAEF;
    padding: 12px;
    border-radius: 10px;
    margin: 10px 0;
}

.chat-bubble-bot {
    background-color: #F3E5F5;
    padding: 12px;
    border-radius: 10px;
    border-left: 4px solid #C39BD3;
    margin: 10px 0;
}

.source-citation {
    background-color: #FFF9E6;
    border-left: 3px solid #FFD700;
    padding: 8px;
    margin-top: 10px;
    font-size: 0.9em;
    border-radius: 5px;
}

div.stButton > button {
    background: linear-gradient(90deg, #F3E5F5, #E6E6FA);
    border: 1px solid #C39BD3;
    border-radius: 8px;
    color: black;
    padding: 8px 20px;
    font-weight: 600;
}

body, [data-testid="stAppViewContainer"] {
    background-color: #FAFAFA;
    font-family: "Segoe UI", sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None


# ---------- SIMPLE Q&A (GROQ) ----------
def ask_ai_simple(prompt):
    """Simple Groq-based Q&A for temporary uploads"""
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a friendly HR policy assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error: {e}"


def smart_truncate(content, question, max_chars=8000):
    """Find most relevant section using keyword matching"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(content)

    # Simple keyword scoring
    keywords = set(question.lower().split())
    scored_chunks = []

    for i, chunk in enumerate(chunks):
        score = sum(1 for word in keywords if word in chunk.lower())
        scored_chunks.append((score, i, chunk))

    # Take top 3 most relevant chunks
    scored_chunks.sort(reverse=True)
    relevant = "\n\n---\n\n".join([chunk for _, _, chunk in scored_chunks[:3]])

    return relevant[:max_chars]


# ---------- RAG SYSTEM ----------
@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system with FAISS"""
    try:
        # Check if FAISS index exists
        if not os.path.exists(FAISS_INDEX_PATH):
            logger.warning(f"FAISS index not found at {FAISS_INDEX_PATH}")
            return None, None

        # Load OpenRouter API key
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error("GROQ_API_KEY not found")
            return None, None

        # Load embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Load FAISS
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

        # Configure retriever
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4,
                           "fetch_k": 20,
                           "lambda_mult": 0.5}
        )

        # Setup LLM with Groq (NO MODERATION!)
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.7,
            api_key=api_key,
            timeout=30,
            max_retries=2
        )

        # Prompt template
        prompt = ChatPromptTemplate.from_template("""You are an HR policy assistant for Escobar Consulting Pvt Ltd.

                Context from policy documents:
                {context}

                Question: {question}

                Rules:
                - Answer ONLY based on the provided context
                - If unsure, say "I don't have enough information in the policy documents"
                - Cite the policy section when possible
                - Use simple, employee-friendly language

                Answer:""")

        # Format documents
        def format_docs(docs):
            if not docs:
                return "No context found."
            formatted = []
            for i, doc in enumerate(docs):
                source = doc.metadata.get('source', 'Unknown')
                formatted.append(f"Document {i + 1} [{source}]:\n{doc.page_content}")
            return "\n\n---\n\n".join(formatted)

        # Build chain
        rag_chain = (
                RunnableParallel({
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough()
                })
                | prompt
                | llm
                | StrOutputParser()
        )

        logger.info("✅ RAG system initialized successfully")
        return rag_chain, retriever

    except Exception as e:
        logger.error(f"RAG initialization error: {e}")
        return None, None


def get_rag_answer(question):
    """Get answer from RAG system with source citations"""
    try:
        if st.session_state.rag_chain is None:
            return "RAG system not initialized.", []

        # Get relevant documents
        docs = st.session_state.retriever.invoke(question)

        # Get answer
        answer = st.session_state.rag_chain.invoke(question)

        # Extract sources
        sources = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            sources.append(f"{source} (Page {page})")

        return answer, sources

    except Exception as e:
        logger.error(f"RAG answer error: {e}")
        return f"Error: {str(e)}", []


# ---------- INDEX MANAGEMENT ----------
def rebuild_faiss_index():
    """Rebuild FAISS index from policies folder"""
    try:
        pdf_files = [f for f in os.listdir(POLICY_DIR) if f.endswith('.pdf')]

        if not pdf_files:
            return False, "No PDF files found in policies folder"

        # Load embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Load and chunk documents
        all_docs = []
        for pdf in pdf_files:
            loader = PyPDFLoader(os.path.join(POLICY_DIR, pdf))
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)
            all_docs.extend(chunks)

        # Create FAISS index
        vectorstore = FAISS.from_documents(all_docs, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)

        logger.info(f"✅ Indexed {len(all_docs)} chunks from {len(pdf_files)} documents")
        return True, f"Successfully indexed {len(pdf_files)} policies"

    except Exception as e:
        logger.error(f"Index rebuild error: {e}")
        return False, str(e)


# ---------- HELPER FUNCTIONS ----------
def save_query(context, question, answer, mode="simple"):
    """Save query to CSV"""
    df = pd.read_csv(QUERY_FILE)
    new = pd.DataFrame([[datetime.now(), context, question, answer, mode]],
                       columns=["timestamp", "context", "question", "answer", "mode"])
    df = pd.concat([df, new], ignore_index=True)
    df.to_csv(QUERY_FILE, index=False)


def category_of(question):
    """Categorize question"""
    q = question.lower()
    if any(x in q for x in ["leave", "holiday", "vacation"]): return "Leave & Attendance"
    if any(x in q for x in ["salary", "pay", "compensation"]): return "Payroll & Compensation"
    if any(x in q for x in ["policy", "rules"]): return "Policy Clarification"
    if any(x in q for x in ["approval", "form", "hr"]): return "HR Processes"
    return "General"


def show_policy_card(path):
    """Display policy card with download link"""
    name = os.path.basename(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"<div class='card'><b>📄 {name}</b><br>"
        f"<a style='color:#6C35A3;' href='data:application/octet-stream;base64,{b64}' download='{name}'>📥 Download</a></div>",
        unsafe_allow_html=True
    )


# ---------- INITIALIZE RAG ON STARTUP ----------
if st.session_state.rag_chain is None and os.path.exists(FAISS_INDEX_PATH):
    with st.spinner("🚀 Loading RAG system..."):
        rag_chain, retriever = initialize_rag_system()
        if rag_chain:
            st.session_state.rag_chain = rag_chain
            st.session_state.retriever = retriever
            st.session_state.rag_enabled = True

# ---------- SIDEBAR ----------
st.sidebar.markdown("### 🧭 Navigation")
page = st.sidebar.radio(
    "",
    [
        "Home",
        "Ask Policy AI (RAG)",
        "Upload & Ask",
        "All Policies",
        "Manage Knowledge Base",
        "My Analytics",
        "My FAQs",
        "Contact & Support"
    ],
    label_visibility="collapsed"
)

# RAG Status Indicator
if st.session_state.rag_enabled:
    st.sidebar.success("🟢 RAG System: Active")
else:
    st.sidebar.warning("🟡 RAG System: Not Available")

# ===========================================
#                HOME PAGE
# ===========================================
if page == "Home":
    st.markdown(
        '<div class="header"><h1>💼 Escobar Consultancy — Policy Insights Dashboard</h1><p>Your AI-powered company policy assistant.</p></div>',
        unsafe_allow_html=True)

    st.title("🏠 Welcome to the Policy Portal")

    st.markdown("""
    <div class='card'>
        <h3>About This Dashboard</h3>
        <p>This platform provides two ways to interact with company policies:</p>
        <ul style="margin-left:20px;">
            <li><b>🔍 Ask Policy AI (RAG)</b> - Query all indexed company policies using advanced semantic search</li>
            <li><b>📤 Upload & Ask</b> - Temporarily upload a document for quick questions</li>
            <li><b>📚 All Policies</b> - Browse and download official policies</li>
            <li><b>⚙️ Manage Knowledge Base</b> - Add new policies to the RAG system</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='card'>
            <h4>🔍 RAG System (Recommended)</h4>
            <p><b>Best for:</b></p>
            <ul>
                <li>Searching across all policies</li>
                <li>Complex multi-document questions</li>
                <li>Getting cited answers with sources</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='card'>
            <h4>📤 Upload & Ask (Quick)</h4>
            <p><b>Best for:</b></p>
            <ul>
                <li>One-time document questions</li>
                <li>External documents</li>
                <li>Quick temporary queries</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ===========================================
#           ASK POLICY AI (RAG)
# ===========================================
elif page == "Ask Policy AI (RAG)":
    st.title("🔍 Ask Policy AI (RAG System)")

    if not st.session_state.rag_enabled:
        st.error("⚠️ RAG system is not available. Please build the knowledge base first in 'Manage Knowledge Base'.")
    else:
        st.info("💡 This searches across ALL indexed policies using semantic understanding.")

        # Sample questions
        with st.expander("📝 Sample Questions"):
            samples = [
                "What is the leave policy?",
                "How do I submit a reimbursement?",
                "What are the working hours?",
                "What is the remote work policy?"
            ]
            for q in samples:
                if st.button(q, key=f"sample_{q}"):
                    st.session_state.temp_question = q

        # Chat interface
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    st.markdown(f"<div class='source-citation'>📚 <b>Sources:</b> {', '.join(msg['sources'])}</div>",
                                unsafe_allow_html=True)

        # Input
        question = st.chat_input("Ask about company policies...") or st.session_state.get("temp_question", None)

        if question:
            st.session_state.temp_question = None

            # User message
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            # AI response
            with st.chat_message("assistant"):
                with st.spinner("Searching policies..."):
                    answer, sources = get_rag_answer(question)
                    st.markdown(answer)
                    if sources:
                        st.markdown(
                            f"<div class='source-citation'>📚 <b>Sources:</b> {', '.join(set(sources[:3]))}</div>",
                            unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            save_query("RAG - All Policies", question, answer, "rag")

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

# ===========================================
#         UPLOAD & ASK (TEMPORARY)
# ===========================================
elif page == "Upload & Ask":
    st.title("📤 Upload & Ask")

    st.info("💡 This mode is for quick questions on documents NOT in the permanent knowledge base.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload Policy", type=["pdf"])
    with col2:
        files = [f for f in os.listdir(POLICY_DIR) if f.endswith(".pdf")]
        selected = st.selectbox("Or Choose Existing", ["None"] + files)

    chosen = uploaded if uploaded else (selected if selected != "None" else None)

    content = None
    if chosen:
        try:
            reader = PdfReader(chosen if uploaded else open(os.path.join(POLICY_DIR, selected), "rb"))
            content = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
            st.success(f"✅ Loaded document ({len(content)} characters)")
        except:
            st.error("❌ Could not read policy file.")

    if content:
        st.markdown("## 💬 Ask Questions About This Document")
        question = st.text_input("Your question:")

        col1, col2 = st.columns(2)
        with col1:
            ask_btn = st.button("Ask AI")
        with col2:
            sum_btn = st.button("Summarize Document")

        if ask_btn and question:
            relevant_content = smart_truncate(content, question)
            answer = ask_ai_simple(f"Policy:\n{relevant_content}\n\nQuestion: {question}")

            st.markdown(f"<div class='chat-bubble-user'><b>You:</b> {question}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-bubble-bot'><b>AI:</b> {answer}</div>", unsafe_allow_html=True)

            context = uploaded.name if uploaded else selected
            save_query(context, question, answer, "simple")

        if sum_btn:
            summary = ask_ai_simple(f"Summarize this policy in 5 key points:\n{content[:8000]}")
            st.markdown(f"<div class='chat-bubble-bot'><b>Summary:</b><br>{summary}</div>", unsafe_allow_html=True)

# ===========================================
#              ALL POLICIES
# ===========================================
elif page == "All Policies":
    st.title("📚 All Company Policies")

    files = [f for f in os.listdir(POLICY_DIR) if f.endswith(".pdf")]

    if not files:
        st.info("No policy files found in the policies folder.")
    else:
        st.write(f"**{len(files)} policies available**")
        selected = st.selectbox("Select a policy:", files)
        show_policy_card(os.path.join(POLICY_DIR, selected))

# ===========================================
#          MANAGE KNOWLEDGE BASE
# ===========================================
elif page == "Manage Knowledge Base":
    st.title("⚙️ Manage RAG Knowledge Base")

    st.markdown("""
    <div class='card'>
        <h4>How It Works</h4>
        <ol>
            <li>Add PDF files to the <code>policies/</code> folder</li>
            <li>Click "🔄 Rebuild Knowledge Base" to index them</li>
            <li>The RAG system will search these documents automatically</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # Show current policies
    files = [f for f in os.listdir(POLICY_DIR) if f.endswith(".pdf")]
    st.subheader(f"📂 Current Policies ({len(files)})")

    if files:
        for f in files:
            st.markdown(f"- {f}")
    else:
        st.warning("No policies found. Add PDF files to the 'policies' folder.")

    st.divider()

    # Rebuild button
    if st.button("🔄 Rebuild Knowledge Base", type="primary"):
        with st.spinner("Indexing policies... This may take a few minutes..."):
            success, message = rebuild_faiss_index()

            if success:
                st.success(f"✅ {message}")
                # Clear cached RAG system
                st.cache_resource.clear()
                st.session_state.rag_chain = None
                st.session_state.rag_enabled = False
                st.info("🔄 Please refresh the page to activate the new index.")
            else:
                st.error(f"❌ Error: {message}")

    # Upload new policy
    st.divider()
    st.subheader("📤 Upload New Policy")
    uploaded = st.file_uploader("Upload PDF to add to knowledge base", type=["pdf"])

    if uploaded:
        save_path = os.path.join(POLICY_DIR, uploaded.name)

        if os.path.exists(save_path):
            st.warning(f"⚠️ File '{uploaded.name}' already exists.")
        else:
            if st.button("💾 Save to Policies Folder"):
                with open(save_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                st.success(f"✅ Saved '{uploaded.name}'. Now rebuild the knowledge base to index it.")

# ===========================================
#               ANALYTICS
# ===========================================
elif page == "My Analytics":
    st.title("📊 Your Policy Insights")

    df = pd.read_csv(QUERY_FILE)

    if df.empty:
        st.info("You haven't asked any questions yet.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Questions", len(df))
        with col2:
            rag_count = len(df[df["mode"] == "rag"])
            st.metric("RAG Queries", rag_count)
        with col3:
            simple_count = len(df[df["mode"] == "simple"])
            st.metric("Simple Queries", simple_count)

        # Category breakdown
        df["category"] = df["question"].apply(category_of)
        st.markdown("### 📌 Question Categories")
        st.bar_chart(df["category"].value_counts())

        # Time analysis
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        st.markdown("### ⏰ When You Ask Questions")
        st.line_chart(df["hour"].value_counts().sort_index())

        # Word cloud
        st.markdown("### 🔥 Frequent Topics")
        wc = WordCloud(width=800, height=300, background_color="white", colormap="Purples") \
            .generate(" ".join(df["question"]))
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

# ===========================================
#                   FAQ
# ===========================================
elif page == "My FAQs":
    st.title("❓ Frequently Asked Questions")

    df = pd.read_csv(QUERY_FILE)

    if df.empty:
        st.info("Ask some questions first!")
    else:
        bundle = "\n".join(df["question"].tail(20))  # Last 20 questions
        faqs = ask_ai_simple(f"Create 5 short FAQs based on these questions:\n{bundle}")
        st.markdown(f"<div class='card'>{faqs}</div>", unsafe_allow_html=True)

# ===========================================
#           CONTACT & SUPPORT
# ===========================================
elif page == "Contact & Support":
    st.title("📞 Contact & Support")

    st.markdown("""
    <div class='card'>
        <h3>We're here to help!</h3>
        <p>If you need assistance regarding any policy or HR process, please reach out below.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🏢 Department Contacts")

    departments = {
        "Human Resources": {"email": "hr@escobarconsultancy.in", "phone": "98234xxxxx"},
        "Finance & Payroll": {"email": "finance@escobarconsultancy.in", "phone": "98188xxxxx"},
        "IT Support": {"email": "it@escobarconsultancy.in", "phone": "99777xxxxx"},
        "Compliance": {"email": "compliance@escobarconsultancy.in", "phone": "99876xxxxx"},
    }

    for dept, info in departments.items():
        st.markdown(
            f"""<div class='card'>
                <h4>{dept}</h4>
                📧 {info['email']}<br>
                📞 {info['phone']}
            </div>""",
            unsafe_allow_html=True
        )
