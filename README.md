# ✈️ LexAir — Asistente Experto en Derechos de los Pasajeros Aéreos (UE)

> Agente de inteligencia artificial especializado en el marco legal europeo de protección de los pasajeros aéreos. Responde consultas citando siempre el Reglamento y artículo exactos. Desplegado como aplicación web con Streamlit.

---

## 📋 Descripción

LexAir es un agente RAG (*Retrieval-Augmented Generation*) construido con LangGraph y Gemini que actúa como asesor jurídico virtual acotado exclusivamente a la normativa aérea de la Unión Europea. Su base de conocimiento está formada por los reglamentos oficiales publicados en el Diario Oficial de la UE.

El agente nunca inventa cifras ni plazos: si la respuesta no está en los documentos, lo dice explícitamente y recomienda consultar a un abogado especializado.

---

## ⚖️ Base legal cubierta

| Documento | Descripción |
|-----------|-------------|
| **Reglamento (CE) n.º 261/2004** | Compensación y asistencia por denegación de embarque, cancelación o gran retraso |
| **Reglamento (CE) n.º 889/2002** | Responsabilidad de las compañías aéreas respecto a los pasajeros y su equipaje |
| **Reglamento (CE) n.º 1008/2008** | Normas comunes para la explotación de servicios aéreos (licencias, precios) |
| **C/2024/5992** | Directrices interpretativas sobre derechos de personas con discapacidad o movilidad reducida |
| **Directrices 261/2004 (2016)** | Directrices interpretativas de la Comisión sobre el Reglamento 261/2004 |

---

## 🌐 Fuente de los documentos

Los documentos legales utilizados como base de conocimiento se obtuvieron desde el portal oficial de la Unión Europea:

**[Derechos de los pasajeros aéreos — Your Europe](https://europa.eu/youreurope/citizens/travel/passenger-rights/air/index_es.htm)**

> Portal oficial de la UE con información sobre los derechos de los ciudadanos europeos en materia de transporte aéreo, incluyendo los reglamentos vigentes y las directrices interpretativas de la Comisión Europea.

---

## 🏗️ Arquitectura

```
Pregunta del usuario
        │
        ▼
 ChromaDB Retriever (k=6)
 — busca los 6 fragmentos más relevantes —
        │
        ▼
 System Prompt + Historial + Contexto legal
        │
        ▼
 Gemini 2.5 Flash (LLM)
        │
        ▼
 Respuesta estructurada con cita legal
```

El agente está implementado como un `StateGraph` de LangGraph con un único nodo `retrieve_and_generate` y memoria conversacional persistente por sesión mediante `InMemorySaver` y `thread_id`.

---

## 🛠️ Stack tecnológico

| Componente | Tecnología |
|------------|------------|
| **LLM** | `gemini-2.5-flash` (Google) |
| **Embeddings** | `gemini-embedding-001` (Google) |
| **Vector Store** | ChromaDB (persistencia local) |
| **Framework** | LangGraph + LangChain |
| **Memoria** | `InMemorySaver` con `thread_id` |
| **Frontend** | Streamlit |
| **Despliegue** | Streamlit Cloud |

---

## 📁 Estructura del proyecto

```
lexair/
│
├── pdfs/                          # Documentos oficiales UE en PDF
│   ├── EU_CELEX_32002R0889_ES_TXT.pdf
│   ├── EU_CELEX_32008R1008_ES_TXT.pdf
│   ├── EU_CELEX_52016XC0615(01)_ES_TXT.pdf
│   ├── EU_C_202405992ES.000101.fmx.xml.pdf
│   └── EU_cellar_439cd3a7-fd3c-4da7-8bf4-b0f60600c1d6.pdf
│
├── chroma_derechos_pasajeros/     # Base vectorial persistida (generada automáticamente)
│
├── app.py                         # Frontend Streamlit
├── asistente_lexair.ipynb         # Notebook de desarrollo y demostración
├── requirements.txt               # Dependencias del proyecto
└── README.md
```

---

## ⚙️ Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/troiti1/lexair.git
cd lexair
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

O manualmente:

```bash
pip install langchain langchain-google-genai langchain-chroma chromadb \
            langgraph pypdf python-dotenv langchain-community \
            langchain-text-splitters streamlit
```

### 3. Configurar la API key

Crear un archivo `.env` en la raíz del proyecto:

```
GOOGLE_API_KEY=tu_clave_api_aqui
```

> Puedes obtener una clave gratuita en [Google AI Studio](https://aistudio.google.com/).

### 4. Ejecutar la app localmente

```bash
streamlit run app.py
```

---

## ☁️ Despliegue en Streamlit Cloud

1. Sube el repositorio a GitHub (asegúrate de incluir `chroma_derechos_pasajeros/` y `pdfs/`).
2. Entra en [share.streamlit.io](https://share.streamlit.io/) con tu cuenta.
3. Pulsa **New app** y selecciona el repositorio y el archivo `app.py`.
4. En **Settings → Secrets**, añade:
   ```
   GOOGLE_API_KEY = "tu_clave_api_aqui"
   ```
5. Pulsa **Deploy**. LexAir quedará online con una URL pública.

---

## 🔍 Cómo funciona el pipeline RAG

### Paso 1 — Carga de documentos
Los PDFs se cargan con `PyPDFDirectoryLoader` usando `glob='*.pdf'` para cargar todos los archivos de la carpeta, independientemente de su nombre.

### Paso 2 — Chunking
Los documentos se segmentan con `RecursiveCharacterTextSplitter`:
- `chunk_size=1500` — suficiente para contener artículos completos con su contexto
- `chunk_overlap=200` — garantiza que las cláusulas que cruzan el límite entre chunks aparezcan completas en al menos uno de ellos

Resultado: **323 chunks** generados a partir de 78 páginas.

### Paso 3 — Indexación vectorial
Cada chunk se convierte en un vector numérico con `gemini-embedding-001` y se almacena en ChromaDB de forma persistente. Se valida que el número de vectores indexados coincida con el número de chunks antes de continuar.

Resultado: **686 vectores** indexados en la colección `reglamentos_ue_pasajeros`.

### Paso 4 — Recuperación y generación
Para cada consulta, el retriever recupera los **6 fragmentos más relevantes** (k=6) por similitud semántica. Estos fragmentos, junto con el historial de conversación y el system prompt, se envían a Gemini para generar la respuesta.

---

## 📝 System Prompt — Reglas absolutas

El agente opera bajo 7 reglas de comportamiento:

1. **Citación obligatoria** — Siempre cita el número de Reglamento y el artículo exacto que sustenta la respuesta.
2. **Restricción de dominio** — Si la pregunta no puede responderse con la base legal disponible, responde exactamente: *"Lo siento, esa información no se encuentra en mi base de conocimiento legal."*
3. **No inventar cifras ni plazos** — Solo menciona importes o plazos si están explícitamente en el contexto recuperado.
4. **Estructura fija** — Respuesta directa → Base legal → Condiciones y excepciones → Recomendación práctica.
5. **Tono formal y accesible** — Explica los términos técnicos cuando los usa.
6. **Memoria conversacional** — Usa el historial completo para evitar repetir información y mantener coherencia.
7. **Límites del rol** — No es un abogado y no puede dar asesoramiento legal vinculante.

---

## 💬 Ejemplos de uso

| Consulta | Respuesta de LexAir |
|----------|---------------------|
| *"Mi vuelo Madrid–Nueva York llegó 4 horas tarde. ¿Tengo derecho a compensación?"* | Sí, 600 € según el Art. 7 del Reg. 261/2004 |
| *"Me denegaron el embarque por overbooking en un vuelo Barcelona–París"* | 250 € + asistencia, Arts. 4, 7, 8 y 9 del Reg. 261/2004 |
| *"Mi madre usa silla de ruedas. ¿Pueden negarle el embarque?"* | No. Reg. 1107/2006 y Comunicación C/2024/5992 |
| *"Vuelo cancelado, el siguiente sale mañana. ¿Me pagan el hotel?"* | Sí, incluso si fue por tormenta. Art. 9 del Reg. 261/2004 |
| *"Mi maleta llegó dañada hace 3 semanas. ¿Qué plazo tengo?"* | 7 días para queja escrita / 2 años para demanda judicial. Reg. 889/2002 |

---

## 🐛 Bugs corregidos respecto al código original

| Bug | Causa | Solución |
|-----|-------|----------|
| El agente siempre respondía "no tengo información" | `glob='EU_*.pdf'` solo cargaba archivos con ese prefijo exacto | Cambiado a `glob='*.pdf'` |
| ChromaDB reportaba solo 50 vectores | Indexación parcial sin validación | Validación cruzada vectores vs. chunks + re-indexación automática |
| Sin validación de la base de conocimiento | No se comprobaba que los documentos se cargaron antes de continuar | Comprobación explícita con mensaje de error descriptivo |
| Recuperación insuficiente | `k=4` chunks recuperados | Aumentado a `k=6` para mejor cobertura de artículos cruzados |
| Sin manejo de errores en el nodo RAG | Cualquier fallo silencioso hacía que el LLM respondiera sin contexto | `try/except` explícito con logging del error |
| Modelo no validado | `gemini-2.5-flash` puede no estar disponible en todas las cuentas | Verificado y confirmado como modelo estable |

---

## 📊 Estadísticas del sistema

```
Documentos PDF procesados : 6
Páginas totales indexadas  : 78
Chunks generados           : 323
Vectores en ChromaDB       : 686
Chunks recuperados/consulta: 6
Modelo LLM                 : gemini-2.5-flash (temperatura 0.1)
Modelo embeddings          : gemini-embedding-001
```

---

## 📄 Licencia

Este proyecto es de carácter académico y educativo. Los documentos legales utilizados son de acceso público y están publicados en el Diario Oficial de la Unión Europea.

---

*Proyecto Final — IA Generativa | Módulo: RAG + LangGraph*


## 🌐 Demo en vivo

Prueba LexAir directamente en el navegador, sin instalar nada:

**[👉 lexair-2ftmxrc3zn7cpkprba6jbk.streamlit.app](https://lexair-2ftmxrc3zn7cpkprba6jbk.streamlit.app/)**
