# app.py — LexAir | Asistente Experto en Derechos de Pasajeros Aéreos (UE)
# Ejecutar con: streamlit run app.py

import os
import uuid
import time
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence
from typing_extensions import TypedDict
import base64

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA — debe ser la primera llamada Streamlit
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LexAir — Derechos de Pasajeros Aéreos UE",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────────────────────

@st.cache_data
def get_logo_b64():
    with open("logo.png", "rb") as f:
        return base64.b64encode(f.read()).decode()


# ──────────────────────────────────────────────────────────────
# CSS PERSONALIZADO — interfaz de chatbot moderna con tonos pastel
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Serif+Display:ital@0;1&display=swap');

    /* ── Variables globales ── */
    :root {
        --bg-main:       #F8F5F0;
        --bg-sidebar:    #FFFFFF;
        --bg-chat:       #FFFFFF;
        --bg-user-msg:   #EDE7F6;
        --bg-bot-msg:    #FFFFFF;
        --accent:        #7B6FDE;
        --accent-light:  #E5DEFF;
        --accent-soft:   #F5F0FF;
        --text-primary:  #1A1A2E;
        --text-secondary:#6B7280;
        --text-muted:    #9CA3AF;
        --border:        #E5E7EB;
        --border-focus:  #C4B5FD;
        --shadow-sm:     0 1px 3px rgba(0,0,0,.07), 0 1px 2px rgba(0,0,0,.04);
        --shadow-md:     0 4px 12px rgba(0,0,0,.08);
        --radius-sm:     8px;
        --radius-md:     14px;
        --radius-lg:     20px;
        --font-body:     'DM Sans', sans-serif;
        --font-display:  'DM Serif Display', serif;
    }

    /* ── Reset y base ── */
    html, body, [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 10% 25%, rgba(123, 111, 222, 0.18) 0%, transparent 55%),
            radial-gradient(ellipse at 90% 15%, rgba(200, 150, 250, 0.15) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 85%, rgba(255, 200, 200, 0.12) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 40%, rgba(160, 200, 240, 0.1) 0%, transparent 50%),
            linear-gradient(160deg, #D4C5F0 0%, #F0D6D6 35%, #C5E0F0 70%, #F0E4D4 100%) !important;
        font-family: var(--font-body) !important;
        color: var(--text-primary) !important;
    }

    /* ── Forzar transparencia en todos los contenedores ── */
    [data-testid="stHeader"], header {
        background: transparent !important;
        backdrop-filter: none !important;
    }
    [data-testid="stApp"], .stApp {
        background: transparent !important;
    }
    .stApp > div, .main, .block-container,
    section[data-testid="stMain"],
    section[data-testid="stMain"] > div,
    div[data-testid="stVerticalBlock"],
    div[data-testid="column"] {
        background: transparent !important;
    }
    footer { display: none !important; }

    /* ── Eliminar fondo oscuro del área de input inferior ── */
    [data-testid="stBottom"],
    [data-testid="stChatInputContainer"] {
        background: transparent !important;
        border: none !important;
    }
    [data-testid="stBottom"] > div {
        background: transparent !important;
    }
    div[data-testid="stChatInput"] {
        box-shadow: var(--shadow-md) !important;
    }

    /* ── Ocultar elementos Streamlit innecesarios ── */
    #MainMenu, footer, [data-testid="stToolbar"],
    [data-testid="stDecoration"] { display: none !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 1.5rem 1.25rem !important;
    }

    /* ── Logo / título sidebar ── */
    .sidebar-logo {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 8px;
        margin-bottom: 1.5rem;
        padding-bottom: 1.25rem;
        border-bottom: 1px solid var(--border);
        text-align: center;
    }
    .logo-img {
        width: 120px;
        height: auto;
        border-radius: 12px;
        flex-shrink: 0;
    }
    .sidebar-logo .logo-sub {
        font-size: 0.68rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        line-height: 1.3;
    }


    /* ── Badges de reglamento en sidebar ── */
    .reg-badge {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        padding: 8px 10px;
        background: var(--accent-soft);
        border: 1px solid var(--accent-light);
        border-radius: var(--radius-sm);
        margin-bottom: 6px;
        font-size: 0.78rem;
        color: var(--text-secondary);
        line-height: 1.4;
    }
    .reg-badge .reg-num {
        font-weight: 600;
        color: var(--accent);
        white-space: nowrap;
        font-size: 0.72rem;
    }

    /* ── Área principal del chat ── */
    .main-header {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
        margin-bottom: 0.5rem;
    }
    .main-logo {
        width: 180px;
        height: auto;
        margin-bottom: 0.6rem;
    }
    .main-header p {
        color: var(--text-secondary);
        font-size: 1.1rem;
        margin: 0;
    }

    /* ── Contenedor del chat ── */
    .chat-container {
        max-width: 780px;
        margin: 0 auto;
        padding: 0 1rem;
    }

    /* ── Burbujas de mensaje ── */
    .message-row {
        display: flex;
        gap: 10px;
        margin-bottom: 1rem;
        animation: fadeSlideUp 0.3s ease;
    }
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .message-row.user { flex-direction: row-reverse; }

    .avatar {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        flex-shrink: 0;
        margin-top: 2px;
    }
    .avatar.bot {
        background: var(--accent);
        color: white;
        box-shadow: 0 2px 8px rgba(37,99,235,.3);
    }
    .avatar.user {
        background: var(--bg-user-msg);
        color: var(--accent);
        border: 1px solid var(--border);
    }

    .bubble {
        max-width: 82%;
        padding: 12px 16px;
        border-radius: var(--radius-md);
        font-size: 0.9rem;
        line-height: 1.65;
        box-shadow: var(--shadow-sm);
    }
    .bubble.bot {
        background: var(--bg-bot-msg);
        border: 1px solid var(--border);
        border-top-left-radius: 4px;
        color: var(--text-primary);
    }
    .bubble.user {
        background: var(--accent);
        color: #fff;
        border-top-right-radius: 4px;
    }
    .bubble strong { font-weight: 600; }
    .bubble.user strong { color: #fff; }

    /* ── Timestamp ── */
    .msg-time {
        font-size: 0.68rem;
        color: var(--text-muted);
        margin-top: 4px;
        padding: 0 4px;
        text-align: right;
    }
    .message-row.user .msg-time { text-align: right; }
    .message-row.bot  .msg-time { text-align: left;  }

    /* ── Welcome card ── */
    .welcome-card {
        background: var(--bg-chat);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 2rem 2.25rem;
        margin: 1.5rem auto 2rem;
        max-width: 780px;
        box-shadow: var(--shadow-md);
    }
    .welcome-card h3 {
        font-family: var(--font-display);
        font-size: 1.4rem;
        color: var(--text-primary);
        margin: 0 0 0.8rem;
    }
    .welcome-card p {
        font-size: 1rem;
        color: var(--text-secondary);
        margin: 0 0 1.2rem;
        line-height: 1.7;
    }
    .suggestion-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
    }
    .suggestion-chip {
        background: var(--accent-soft);
        border: 1px solid var(--accent-light);
        border-radius: var(--radius-sm);
        padding: 8px 12px;
        font-size: 0.8rem;
        color: var(--accent);
        cursor: pointer;
        transition: all .15s ease;
        line-height: 1.4;
    }
    .suggestion-chip:hover {
        background: var(--accent-light);
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(37,99,235,.15);
    }

    /* ── Spinner / typing indicator ── */
    .typing-indicator {
        display: flex;
        gap: 4px;
        align-items: center;
        padding: 12px 16px;
        background: var(--bg-bot-msg);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        border-top-left-radius: 4px;
        width: fit-content;
        box-shadow: var(--shadow-sm);
    }
    .typing-dot {
        width: 7px; height: 7px;
        background: var(--accent);
        border-radius: 50%;
        animation: typingBounce 1.2s infinite;
        opacity: 0.7;
    }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes typingBounce {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
        30%           { transform: translateY(-5px); opacity: 1; }
    }

    /* ── Input chat nativo de Streamlit ── */
    [data-testid="stChatInput"] {
        border-radius: var(--radius-lg) !important;
        border: 1.5px solid var(--border) !important;
        background: white !important;
        box-shadow: var(--shadow-md) !important;
        font-family: var(--font-body) !important;
        transition: border-color .2s ease, box-shadow .2s ease !important;
    }
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] input,
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] * {
        background: white !important;
        color: var(--text-primary) !important;
        caret-color: var(--accent) !important;
    }
    [data-testid="stChatInput"] input::placeholder,
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--text-muted) !important;
        opacity: 1 !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--border-focus) !important;
        box-shadow: 0 0 0 3px rgba(123, 111, 222, .12), var(--shadow-md) !important;
    }

    /* ── Botón nueva sesión ── */
    .stButton > button {
        background: white !important;
        color: var(--accent) !important;
        border: 1.5px solid var(--accent-light) !important;
        border-radius: var(--radius-sm) !important;
        font-family: var(--font-body) !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        padding: 0.4rem 1rem !important;
        transition: all .15s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background: var(--accent-soft) !important;
        border-color: var(--accent) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 8px rgba(37,99,235,.12) !important;
    }

    /* ── Info box sidebar ── */
    .info-box {
        background: var(--accent-soft);
        border: 1px solid var(--accent-light);
        border-radius: var(--radius-sm);
        padding: 10px 12px;
        font-size: 0.78rem;
        color: var(--text-secondary);
        line-height: 1.5;
        margin-top: 0.5rem;
    }
    .info-box strong { color: var(--accent); }

    /* ── Status badge ── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.72rem;
        color: #059669;
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-radius: 20px;
        padding: 3px 10px;
        margin-bottom: 1rem;
    }
    .status-dot {
        width: 6px; height: 6px;
        background: #10B981;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.4; }
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

    /* ── Sidebar section titles ── */
    .sidebar-section-title {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin: 1.2rem 0 0.6rem;
    }

    /* ── Separador ── */
    hr.custom-divider {
        border: none;
        border-top: 1px solid var(--border);
        margin: 1.25rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# CARGA DE RECURSOS (cacheado para no re-ejecutar en cada mensaje)
# ──────────────────────────────────────────────────────────────
load_dotenv()
API_KEY        = os.getenv("GOOGLE_API_KEY")
PDF_DIR        = "./pdfs"
CHROMA_DIR     = "./chroma_derechos_pasajeros"
COLLECTION     = "reglamentos_ue_pasajeros"

SYSTEM_PROMPT = """Eres **LexAir**, un asistente jurídico experto y especializado \
exclusivamente en los derechos de los pasajeros aéreos en la Unión Europea. \
Tu base de conocimiento está compuesta únicamente por los siguientes reglamentos \
y directrices oficiales de la UE:

• Reglamento (CE) n.º 261/2004: compensación y asistencia por denegación de embarque, \
cancelación o gran retraso de vuelos.
• Reglamento (CE) n.º 889/2002: responsabilidad de las compañías aéreas sobre \
pasajeros y equipaje (aplica el Convenio de Montreal).
• Reglamento (CE) n.º 1008/2008: normas para la explotación de servicios aéreos \
(licencias, transparencia de precios).
• Comunicación C/2024/5992: directrices interpretativas sobre derechos de personas \
con discapacidad o movilidad reducida en transporte aéreo (Reglamento 1107/2006).
• Directrices interpretativas 2016 sobre el Reglamento 261/2004.

================================================
REGLAS ABSOLUTAS DE COMPORTAMIENTO:
================================================

1. **CITACIÓN OBLIGATORIA**: SIEMPRE debes citar el número de Reglamento y el \
artículo exacto que sustenta tu respuesta.

2. **RESTRICCIÓN DE DOMINIO**: Si la pregunta NO puede responderse con el contexto \
proporcionado, responde: 'Lo siento, esa información no se encuentra en mi base de \
conocimiento legal. Le recomiendo consultar con un abogado especializado en derecho \
aéreo o con el organismo nacional de aplicación de su país.'

3. **NO INVENTAR CIFRAS NI PLAZOS**: Solo menciona cifras si están explícitamente \
en el contexto. Nunca extrapoles importes legales.

4. **ESTRUCTURA DE RESPUESTA**:
   a) **Respuesta directa**: La respuesta concisa a la pregunta.
   b) **Base legal**: El artículo y Reglamento exactos.
   c) **Condiciones y excepciones**: Matices importantes.
   d) **Recomendación práctica** (si procede): Cómo ejercer el derecho.

5. **TONO Y ESTILO**: Formal pero accesible. Explica términos técnicos. Usa negrita para conceptos clave.

6. **MEMORIA CONVERSACIONAL**: Usa el historial para evitar repeticiones y conectar preguntas.

7. **LÍMITES**: No eres abogado y no puedes dar asesoramiento vinculante.

================================================
CONTEXTO LEGAL RECUPERADO PARA ESTA CONSULTA:
================================================

{context}
"""


@st.cache_resource(show_spinner=False)
def load_agent():
    """Carga y cachea todos los recursos del agente RAG."""
    # 1. Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=API_KEY,
    )

    # 2. ChromaDB — intentar cargar la colección existente
    #    Si no existe, cargar los PDFs y re-indexar
    vectorstore = None
    try:
        import chromadb
        _c = chromadb.PersistentClient(path=CHROMA_DIR)
        col = _c.get_collection(COLLECTION)
        count = col.count()
        _c._impl.close()

        if count > 0:
            vectorstore = Chroma(
                collection_name=COLLECTION,
                embedding_function=embeddings,
                persist_directory=CHROMA_DIR,
            )
    except Exception:
        pass

    if vectorstore is None:
        # Cargar PDFs y crear la colección desde cero
        loader = PyPDFDirectoryLoader(path=PDF_DIR, glob="EU_*.pdf", silent_errors=True)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        BATCH = 50
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i:i + BATCH]
            if vectorstore is None:
                vectorstore = Chroma.from_documents(
                    documents=batch, embedding=embeddings,
                    collection_name=COLLECTION, persist_directory=CHROMA_DIR,
                )
            else:
                vectorstore.add_documents(batch)
            if i + BATCH < len(chunks):
                time.sleep(1)

    # 3. LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=API_KEY,
        temperature=0.1,
        max_output_tokens=2048,
    )

    # 4. Estado del grafo
    class AgentState(TypedDict):
        messages: Annotated[Sequence[object], add_messages]

    # 5. Nodo RAG
    def retrieve_and_generate(state):
        messages = state["messages"]
        last_human = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human = msg.content
                break
        if not last_human:
            return {"messages": [AIMessage(content="No he recibido ninguna pregunta.")]}

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        docs = retriever.invoke(last_human)

        if docs:
            parts = [
                f"[FRAGMENTO {i} | Fuente: {os.path.basename(d.metadata.get('source','?'))} | Página: {d.metadata.get('page','?')}]\n{d.page_content}"
                for i, d in enumerate(docs, 1)
            ]
            context = "\n\n" + "—" * 40 + "\n\n".join(parts)
        else:
            context = "No se han encontrado fragmentos relevantes en la base de conocimiento."

        llm_msgs = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]
        for msg in messages:
            if isinstance(msg, (HumanMessage, AIMessage)):
                llm_msgs.append(msg)

        response = llm.invoke(llm_msgs)
        return {"messages": [response]}

    # 6. Grafo
    wf = StateGraph(state_schema=AgentState)
    wf.add_node("retrieve_and_generate", retrieve_and_generate)
    wf.add_edge(START, "retrieve_and_generate")
    wf.add_edge("retrieve_and_generate", END)
    app = wf.compile(checkpointer=InMemorySaver())

    return app, vectorstore


def ask_lexair(app, question: str, thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )
    return result["messages"][-1].content


# ──────────────────────────────────────────────────────────────
# ESTADO DE SESIÓN
# ──────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = f"streamlit_{uuid.uuid4().hex[:8]}"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # lista de dicts {role, content, time}
if "session_start_index" not in st.session_state:
    st.session_state.session_start_index = 0
if "agent_loaded" not in st.session_state:
    st.session_state.agent_loaded = False


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    logo_b64 = get_logo_b64()
    st.markdown(f"""
    <div class="sidebar-logo">
        <img src="data:image/png;base64,{logo_b64}" class="logo-img">
        <div class="logo-sub">Derechos del pasajero UE</div>
    </div>
    """, unsafe_allow_html=True)

    # Status badge
    st.markdown('<div class="status-badge"><div class="status-dot"></div>Sistema activo</div>',
                unsafe_allow_html=True)

    # Nueva sesión (no borra el historial)
    if st.button("🔄  Nueva conversación", key="new_conv"):
        st.session_state.session_id = f"streamlit_{uuid.uuid4().hex[:8]}"
        st.session_state.session_start_index = len(st.session_state.chat_history)
        st.rerun()

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # --- Historial de conversación ---
    if st.session_state.chat_history:
        st.markdown('<div class="sidebar-section-title">📜 Historial</div>', unsafe_allow_html=True)
        for i, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                preview = msg["content"][:55] + ("..." if len(msg["content"]) > 55 else "")
                if st.button(f"💬 {preview}", key=f"hist_{i}", use_container_width=True):
                    st.session_state.scroll_to = i
                    st.rerun()
        if st.session_state.session_start_index > 0:
            if st.button("⬇️  Volver al final", key="goto_latest", use_container_width=True):
                st.session_state.session_start_index = 0
                st.rerun()
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>⚠️ Aviso legal</strong><br>
        LexAir proporciona información general sobre normativa europea. No sustituye el asesoramiento
        de un abogado especializado en derecho aéreo.
    </div>
    """, unsafe_allow_html=True)

    # Sesión actual (colapsado)
    with st.expander("ℹ️ Info de sesión"):
        st.caption(f"ID: `{st.session_state.session_id}`")
        st.caption(f"Mensajes: {len(st.session_state.chat_history)}")


# ──────────────────────────────────────────────────────────────
# ÁREA PRINCIPAL
# ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <img src="data:image/png;base64,{get_logo_b64()}" class="main-logo">
    <p>Consulta tus derechos como pasajero aéreo en la Unión Europea</p>
</div>
""", unsafe_allow_html=True)


# ── Cargar agente (con spinner) ──
with st.spinner("Cargando base de conocimiento legal..."):
    try:
        agent_app, vectorstore = load_agent()
        st.session_state.agent_loaded = True
    except Exception as e:
        st.error(f"❌ Error al cargar el agente: {e}")
        st.stop()


# ── Sugerencias de preguntas (solo si el chat está vacío) ──
SUGGESTIONS = [
    "✈️  Mi vuelo llegó con 4h de retraso. ¿Tengo derecho a compensación?",
    "🚫  Me denegaron el embarque por overbooking. ¿Qué derechos tengo?",
    "♿  Mi madre usa silla de ruedas. ¿Puede la aerolínea negarle el embarque?",
    "🧳  Mi maleta llegó dañada. ¿Cuánto tiempo tengo para reclamar?",
]

visible_msgs = st.session_state.chat_history[st.session_state.session_start_index:]
if not visible_msgs:
    st.markdown("""
    <div class="welcome-card">
        <h3>👋 ¡Hola! Soy LexAir</h3>
        <p>Estoy especializado en los reglamentos europeos de protección al pasajero aéreo.
        Puedo ayudarte con compensaciones por retrasos o cancelaciones, denegaciones de embarque,
        derechos con movilidad reducida, reclamaciones por equipaje y mucho más.</p>
        <div class="suggestion-grid">
    """, unsafe_allow_html=True)

    cols = st.columns(2)
    for idx, suggestion in enumerate(SUGGESTIONS):
        with cols[idx % 2]:
            if st.button(suggestion, key=f"sug_{idx}", use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()

    st.markdown("</div></div>", unsafe_allow_html=True)


# ── Historial de mensajes ──
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

start = st.session_state.session_start_index
for i, msg in enumerate(st.session_state.chat_history[start:], start=start):
    role  = msg["role"]
    content = msg["content"]
    ts    = msg.get("time", "")
    if role == "user":
        st.markdown(f"""
        <div class="message-row user" id="msg_{i}">
            <div class="avatar user">👤</div>
            <div>
                <div class="bubble user">{content}</div>
                <div class="msg-time">{ts}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Convertir **negrita** a <strong> para HTML
        import re
        html_content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
        html_content = html_content.replace("\n", "<br>")
        st.markdown(f"""
        <div class="message-row bot" id="msg_{i}">
            <div class="avatar bot">✈</div>
            <div>
                <div class="bubble bot">{html_content}</div>
                <div class="msg-time">{ts}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ── Procesar pregunta pendiente (desde botones de sugerencia) ──
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")
    ts_now = time.strftime("%H:%M")
    st.session_state.chat_history.append({"role": "user", "content": question, "time": ts_now})

    with st.spinner(""):
        st.markdown("""
        <div class="chat-container">
            <div class="message-row bot">
                <div class="avatar bot">✈</div>
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        answer = ask_lexair(agent_app, question, st.session_state.session_id)

    st.session_state.chat_history.append({"role": "bot", "content": answer, "time": time.strftime("%H:%M")})
    st.rerun()


# ── Scroll a mensaje específico ──
if "scroll_to" in st.session_state:
    idx = st.session_state["scroll_to"]
    if idx < st.session_state.session_start_index:
        st.session_state.session_start_index = idx
        st.rerun()
    st.session_state.pop("scroll_to")
    st.markdown(f"""
    <style>
        #msg_{idx} {{
            background: rgba(123, 111, 222, 0.12) !important;
            border-radius: 14px !important;
            padding: 2px !important;
            animation: msgPulse 2s ease;
        }}
        @keyframes msgPulse {{
            0%   {{ background: rgba(123, 111, 222, 0.18) !important; }}
            100% {{ background: transparent !important; }}
        }}
    </style>
    """, unsafe_allow_html=True)


# ── Input del usuario ──
if prompt := st.chat_input("Escribe tu consulta sobre derechos de pasajeros..."):
    ts_now = time.strftime("%H:%M")
    st.session_state.chat_history.append({"role": "user", "content": prompt, "time": ts_now})

    with st.spinner(""):
        st.markdown("""
        <div class="chat-container">
            <div class="message-row bot">
                <div class="avatar bot">✈</div>
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        answer = ask_lexair(agent_app, prompt, st.session_state.session_id)

    st.session_state.chat_history.append({"role": "bot", "content": answer, "time": time.strftime("%H:%M")})
    st.rerun()