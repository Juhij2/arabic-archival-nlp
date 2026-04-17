import os
import re
import subprocess
import warnings
from pathlib import Path

import chromadb
import pytesseract
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
from sentence_transformers import SentenceTransformer
from transformers import MarianMTModel, MarianTokenizer

from db import (
    get_connection,
    init_db,
    upsert_document,
    replace_entities,
    upsert_summary,
    get_document_details,
    search_documents_fts,
)

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Arabic Archival NLP Pipeline",
    page_icon="🕌",
    layout="wide"
)

DB_PATH = Path('data/archive.db')


@st.cache_resource
def load_models():
    models = {}

    with st.spinner("Loading embedder..."):
        models['embedder'] = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )

    with st.spinner("Loading ChromaDB..."):
        client = chromadb.PersistentClient(path="data/chromadb")
        try:
            models['collection'] = client.get_collection("arabic_documents")
        except Exception:
            models['collection'] = None

    with st.spinner("Loading Arabic NER model..."):
        try:
            camel_path = os.path.expanduser("~/.camel_tools/data/ner/arabert")
            if not os.path.exists(camel_path):
                st.info("Downloading Arabic NER model (542MB, one time only)...")
                subprocess.run(["camel_data", "-i", "ner-arabert"], capture_output=False, timeout=300)
            from camel_tools.ner import NERecognizer
            models['ner'] = NERecognizer.pretrained()
            models['ner_available'] = True
        except Exception:
            models['ner'] = None
            models['ner_available'] = False

    with st.spinner("Loading translator..."):
        try:
            model_name = "Helsinki-NLP/opus-mt-ar-en"
            models['tokenizer'] = MarianTokenizer.from_pretrained(model_name)
            models['translator'] = MarianMTModel.from_pretrained(model_name)
        except Exception:
            models['tokenizer'] = None
            models['translator'] = None

    return models


def get_db():
    conn = get_connection(DB_PATH)
    init_db(conn)
    return conn


def preprocess_image(img):
    img = img.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.5)
    img = img.filter(ImageFilter.SHARPEN)
    w, h = img.size
    if w < 1200:
        scale = 1200 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def clean_arabic(text):
    text = re.sub('[إأآا]', 'ا', text)
    text = re.sub('[يى]', 'ي', text)
    arabic_pattern = re.compile(r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s\.\،\,\:\!\?]')
    text = arabic_pattern.sub(' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 3]
    return '\n'.join(lines).strip()


def prepare_for_translation(text):
    clean = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    words = clean.split()
    words = [w for w in words if len(w) >= 3]
    deduped = []
    prev = None
    for w in words:
        if w != prev:
            deduped.append(w)
        prev = w
    clean = ' '.join(deduped)
    clean = re.sub(r' +', ' ', clean)
    return clean.strip()


def translate_arabic(text, models):
    if not models.get('translator'):
        return "Translation not available"
    try:
        words = text.split()
        chunks = []
        current = []
        for word in words:
            current.append(word)
            if len(' '.join(current)) > 300:
                chunks.append(' '.join(current))
                current = []
        if current:
            chunks.append(' '.join(current))

        results = []
        for chunk in chunks[:5]:
            if not chunk.strip():
                continue
            inputs = models['tokenizer'](
                chunk, return_tensors="pt", padding=True,
                truncation=True, max_length=512
            )
            translated = models['translator'].generate(**inputs)
            result = models['tokenizer'].decode(translated[0], skip_special_tokens=True)
            results.append(result)
        return ' '.join(results)
    except Exception as e:
        return f"Translation error: {e}"


def extract_entities(text, models):
    result = {"PERSON": [], "LOCATION": [], "ORGANIZATION": []}
    if not models.get('ner') or not text.strip():
        return result
    try:
        from camel_tools.tokenizers.word import simple_word_tokenize
        tokens = simple_word_tokenize(text)
        if not tokens:
            return result
        labels = models['ner'].predict_sentence(tokens)
        type_map = {"PERS": "PERSON", "LOC": "LOCATION", "ORG": "ORGANIZATION"}
        current = []
        current_type = None
        for token, label in zip(tokens, labels):
            if label.startswith('B-'):
                if current and current_type:
                    result.setdefault(current_type, []).append(' '.join(current))
                current_type = type_map.get(label[2:], label[2:])
                current = [token]
            elif label.startswith('I-') and current:
                current.append(token)
            else:
                if current and current_type:
                    result.setdefault(current_type, []).append(' '.join(current))
                current = []
                current_type = None
        if current and current_type:
            result.setdefault(current_type, []).append(' '.join(current))
    except Exception as e:
        st.error(f"NER error: {e}")
    return result


def search_docs_semantic(query, models):
    if not models.get('collection'):
        return []
    try:
        qe = models['embedder'].encode(query).tolist()
        results = models['collection'].query(query_embeddings=[qe], n_results=3)
        distances = results['distances'][0]
        min_d = min(distances)
        max_d = max(distances)
        output = []
        for meta, dist, doc in zip(results['metadatas'][0], distances, results['documents'][0]):
            sim = round((1 - (dist - min_d) / (max_d - min_d + 0.001)) * 100, 1)
            output.append({
                "source": meta['source'],
                "similarity": sim,
                "description": meta['description'],
                "persons": meta.get('persons', ''),
                "locations": meta.get('locations', ''),
                "preview": doc[:150]
            })
        return output
    except Exception as e:
        st.error(f"Search error: {e}")
        return []


st.title("🕌 Arabic Archival NLP Pipeline")
st.markdown("""
**SQLite-backed proof of concept for Arabic historical document analysis**

**Layer 1:** OCR → Text Cleaning → Named Entity Recognition → Semantic Search

**Layer 2:** Arabic → English draft summary stored in SQLite
""")
st.divider()

models = load_models()
conn = get_db()

if models.get('ner_available'):
    st.success(f"✓ Models loaded. Archive DB ready at `{DB_PATH}`")
else:
    st.warning(f"✓ Archive DB ready at `{DB_PATH}`. NER model may still need setup.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Process Document", "🔍 Search Documents", "🗄 Archive Browser", "🕸️ Graph Explorer", "🤖 AI Analysis"])

with tab1:
    st.markdown("### Upload a scanned Arabic document")
    uploaded_file = st.file_uploader("Choose an Arabic document image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Document", use_container_width=True)
            run_btn = st.button("🔍 Run Full Pipeline", type="primary", use_container_width=True)

        with col2:
            if run_btn:
                with st.spinner("Running OCR..."):
                    processed = preprocess_image(image)
                    raw_text = pytesseract.image_to_string(processed, lang='ara', config='--oem 3 --psm 6')
                    clean_text = clean_arabic(raw_text)

                if not clean_text.strip():
                    st.error("No Arabic text detected in this image")
                else:
                    translation_input = prepare_for_translation(clean_text)
                    with st.spinner("Translating..."):
                        translation = translate_arabic(translation_input, models)
                    with st.spinner("Extracting entities..."):
                        entities = extract_entities(clean_text, models)

                    doc_id = upsert_document(
                        conn,
                        source=uploaded_file.name,
                        title=uploaded_file.name,
                        image_path=uploaded_file.name,
                        description="Uploaded via Streamlit",
                        ocr_text_raw=raw_text,
                        ocr_text_clean=clean_text,
                        translation_input=translation_input,
                        translation_status='draft_generated'
                    )
                    replace_entities(conn, doc_id, entities, source_method='camel_pipeline')
                    upsert_summary(conn, doc_id, 'english_draft', translation, method='opus_mt', quality_note='experimental draft translation')

                    st.markdown("#### 📝 Extracted Arabic Text")
                    st.text_area("Arabic OCR Output (cleaned)", clean_text, height=150)

                    st.markdown("#### 🌐 English Draft")
                    st.caption("Stored in SQLite as an experimental draft summary/translation")
                    st.text_area("English Draft", translation, height=150)

                    with st.expander("🔍 View cleaned input sent to translator"):
                        st.text(translation_input)

                    st.markdown("#### 🏷 Named Entities")
                    ner_col1, ner_col2, ner_col3 = st.columns(3)
                    with ner_col1:
                        st.markdown("**👤 People**")
                        for e in entities.get('PERSON', []) or ["None found"]:
                            st.markdown(f"• {e}")
                    with ner_col2:
                        st.markdown("**📍 Locations**")
                        for e in entities.get('LOCATION', []) or ["None found"]:
                            st.markdown(f"• {e}")
                    with ner_col3:
                        st.markdown("**🏛 Organizations**")
                        for e in entities.get('ORGANIZATION', []) or ["None found"]:
                            st.markdown(f"• {e}")

                    st.success(f"Saved document and derived data to SQLite with document_id={doc_id}")
                    st.session_state["last_processed_doc_id"] = doc_id

with tab2:
    st.markdown("### Search across all indexed documents")
    query = st.text_input("Search query", placeholder="Try: names of people, official document, or أسماء الأشخاص")
    mode = st.radio("Search mode", ["Semantic (Chroma)", "Lexical (SQLite FTS)"], horizontal=True)

    if query:
        if mode == "Semantic (Chroma)":
            results = search_docs_semantic(query, models)
            st.markdown(f"### Semantic results for: *{query}*")
            medals = ["🥇", "🥈", "🥉"]
            for i, r in enumerate(results):
                with st.expander(f"{medals[i]} {r['source']} ({r['similarity']}% match)", expanded=(i == 0)):
                    st.markdown(f"**Type:** {r['description']}")
                    if r['persons']:
                        st.markdown(f"**👤 People:** {r['persons']}")
                    if r['locations']:
                        st.markdown(f"**📍 Places:** {r['locations']}")
                    st.markdown(f"**Text preview:** {r['preview']}...")
        else:
            safe_query = ' '.join(query.split())
            try:
                rows = search_documents_fts(conn, safe_query)
                st.markdown(f"### SQLite FTS results for: *{query}*")
                if not rows:
                    st.info("No lexical matches yet in SQLite. Process or import documents first.")
                for row in rows:
                    with st.expander(row['source'], expanded=True):
                        st.markdown(f"**Description:** {row['description'] or '—'}")
                        st.markdown(f"**Preview:** {(row['ocr_text_clean'] or '')[:250]}...")
            except Exception as e:
                st.error(f"FTS query issue: {e}")

with tab3:
    st.markdown("### Browse archive records")
    source_name = st.text_input("Load a document by source filename", placeholder="arabic_doc_004.jpg")
    if source_name:
        doc = get_document_details(conn, source_name)
        if not doc:
            st.info("Document not found in SQLite yet.")
        else:
            st.json(doc)

with tab4:
    import sqlite3 as _sqlite3
    import pandas as pd
    import networkx as nx
    from pyvis.network import Network
    import tempfile, os

    st.markdown("### Entity Knowledge Graph")
    st.markdown("Explore connections between people, locations, and organizations across all archived documents.")

    @st.cache_data
    def load_graph_data():
        _conn = _sqlite3.connect(DB_PATH)
        docs = pd.read_sql("""
            SELECT d.id, d.source, s.summary_text as english_draft
            FROM documents d
            LEFT JOIN summaries s ON s.document_id = d.id
            AND s.summary_type = 'english_draft'
        """, _conn)
        ents = pd.read_sql("""
            SELECT e.id, e.document_id, e.entity_type,
                   e.entity_text, d.source as document_source
            FROM entities e
            JOIN documents d ON d.id = e.document_id
        """, _conn)
        _conn.close()
        return docs, ents

    g_docs, g_ents = load_graph_data()

    if g_ents.empty:
        st.warning("No entities found. Process some documents first.")
    else:
        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Documents", len(g_docs))
        m2.metric("Total Entities", len(g_ents))
        m3.metric("People", len(g_ents[g_ents['entity_type'] == 'PERSON']))
        m4.metric("Locations", len(g_ents[g_ents['entity_type'] == 'LOCATION']))

        # Sidebar filters inside tab
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            type_filter = st.selectbox(
                "Filter by entity type",
                ["ALL"] + sorted(g_ents['entity_type'].unique().tolist()),
                key="graph_type_filter"
            )
        with col_filter2:
            entity_filter = st.selectbox(
                "Focus on entity",
                ["ALL"] + sorted(g_ents['entity_text'].unique().tolist()),
                key="graph_entity_filter"
            )

        # Filter data
        filtered = g_ents.copy()
        if type_filter != "ALL":
            filtered = filtered[filtered['entity_type'] == type_filter]
        if entity_filter != "ALL":
            filtered = filtered[filtered['entity_text'] == entity_filter]

        # Build graph
        G = nx.Graph()
        color_map = {
            "PERSON": "#e74c3c",
            "LOCATION": "#3498db",
            "ORGANIZATION": "#2ecc71",
            "DATE": "#f39c12",
            "EVENT": "#9b59b6"
        }

        for _, doc in g_docs.iterrows():
            G.add_node(f"doc_{doc['id']}", label=doc['source'],
                      node_type="document",
                      title=f"Document: {doc['source']}")

        for _, ent in filtered.iterrows():
            eid = f"ent_{ent['entity_text']}"
            if not G.has_node(eid):
                G.add_node(eid,
                    label=ent['entity_text'],
                    node_type=ent['entity_type'],
                    title=f"{ent['entity_type']}: {ent['entity_text']}",
                    color=color_map.get(ent['entity_type'], "#95a5a6"))
            G.add_edge(f"doc_{ent['document_id']}", eid)

        # Render pyvis
        net = Network(height="600px", width="100%",
                     bgcolor="#1a1a2e", font_color="white")
        for node_id, attrs in G.nodes(data=True):
            if attrs.get("node_type") == "document":
                net.add_node(node_id, label=attrs["label"],
                           title=attrs["title"],
                           color="#1abc9c", size=30, shape="box")
            else:
                net.add_node(node_id, label=attrs["label"],
                           title=attrs["title"],
                           color=attrs.get("color", "#95a5a6"),
                           size=20, shape="dot")
        for s, t in G.edges():
            net.add_edge(s, t, color="#555555")

        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "forceAtlas2Based": {
              "gravitationalConstant": -80,
              "centralGravity": 0.015,
              "springLength": 180,
              "springConstant": 0.08,
              "damping": 0.6,
              "avoidOverlap": 0.5
            },
            "solver": "forceAtlas2Based",
            "stabilization": {
              "enabled": true,
              "iterations": 400,
              "fit": false
            },
            "minVelocity": 0.1
          },
          "interaction": {"hover": true, "tooltipDelay": 100},
          "layout": {"randomSeed": 42, "improvedLayout": true}
        }
        """)

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".html", mode="w"
        ) as f:
            net.save_graph(f.name)
            html_path = f.name

        with open(html_path, "r") as f:
            html_content = f.read()

        # Fix canvas sizing + reliable auto-fit so all nodes stay visible.
        # Single fit() calls are unreliable in Streamlit iframes because the
        # event timing varies. We fit multiple times with increasing delays
        # and use padding so nodes do not get clipped at canvas edges.
        fit_script = """
        <script type="text/javascript">
          (function() {
            function doFit() {
              if (typeof network !== 'undefined' && network) {
                try {
                  network.fit({
                    animation: false,
                    minZoomLevel: 0.3,
                    maxZoomLevel: 2.0
                  });
                } catch(e) {}
              }
            }
            function applyFix() {
              var card = document.getElementById('mynetwork');
              if (card) {
                card.style.width = '100%';
                card.style.height = '560px';
              }
              if (typeof network !== 'undefined' && network) {
                // Fit when physics settles
                network.once('stabilizationIterationsDone', doFit);
                // Fallback fits at increasing delays to catch late-drifting nodes
                setTimeout(doFit, 300);
                setTimeout(doFit, 900);
                setTimeout(doFit, 1800);
                setTimeout(doFit, 3000);
              } else {
                setTimeout(applyFix, 150);
              }
            }
            if (document.readyState === 'complete') { applyFix(); }
            else { window.addEventListener('load', applyFix); }
          })();
        </script>
        </body>
        """
        html_content = html_content.replace("</body>", fit_script)

        st.components.v1.html(html_content, height=600, scrolling=False)
        os.unlink(html_path)

        st.markdown("""
        **Legend:** 🟦 Document &nbsp; 🔴 Person &nbsp;
        🔵 Location &nbsp; 🟢 Organization &nbsp;
        🟡 Date &nbsp; 🟣 Event
        """)

        # Example queries
        st.markdown("---")
        st.markdown("### Example Research Queries")

        with st.expander("📌 People appearing in more than one document"):
            multi = g_ents[g_ents['entity_type'] == 'PERSON']\
                .groupby('entity_text')['document_id']\
                .nunique().reset_index()
            multi.columns = ['Person', 'Document Count']
            multi = multi[multi['Document Count'] > 1]
            st.dataframe(multi if not multi.empty
                        else pd.DataFrame([{"Result": "No person appears in multiple documents yet"}]),
                        use_container_width=True)

        with st.expander("📌 All locations mentioned across the archive"):
            locs = g_ents[g_ents['entity_type'] == 'LOCATION']\
                [['entity_text', 'document_source']]\
                .drop_duplicates()\
                .rename(columns={'entity_text': 'Location',
                                'document_source': 'Found In'})
            st.dataframe(locs if not locs.empty
                        else pd.DataFrame([{"Result": "No locations found"}]),
                        use_container_width=True)

        with st.expander("📌 Documents where a person and location appear together"):
            _conn2 = _sqlite3.connect(DB_PATH)
            together = pd.read_sql("""
                SELECT DISTINCT d.source as Document,
                    p.entity_text as Person,
                    l.entity_text as Location
                FROM documents d
                JOIN entities p ON p.document_id = d.id
                    AND p.entity_type = 'PERSON'
                JOIN entities l ON l.document_id = d.id
                    AND l.entity_type = 'LOCATION'
            """, _conn2)
            _conn2.close()
            st.dataframe(together if not together.empty
                        else pd.DataFrame([{"Result": "No matches found"}]),
                        use_container_width=True)
with tab5:
    import anthropic
    import json as _json

    st.markdown("### 🤖 AI Manuscript Analysis")
    st.markdown(
        "Claude analyzes the raw OCR output of a newly uploaded manuscript: "
        "reconstructing the original Arabic text, producing a scholarly English translation, "
        "and flagging what was recoverable versus too damaged to interpret."
    )

    # ── Helper: chunk text ───────────────────────────────────
    def chunk_text(text, size=400):
        """Split text into chunks of roughly `size` characters at word boundaries."""
        words = text.split()
        chunks, current = [], []
        length = 0
        for word in words:
            current.append(word)
            length += len(word) + 1
            if length >= size:
                chunks.append(" ".join(current))
                current, length = [], 0
        if current:
            chunks.append(" ".join(current))
        return chunks

    # ── Helper: call Claude on one chunk ────────────────────
    def analyze_chunk(client, chunk_text_str, chunk_num, total):
        prompt = f"""You are a specialist in historical Arabic manuscripts, particularly administrative, legal, and archival documents from the medieval and early modern Islamic world (roughly 9th–19th century).

You are given a chunk of Arabic text that was extracted via OCR from a scanned manuscript image. The OCR quality is poor: words may be fused, characters broken, diacritics missing, and some tokens may be pure noise.

Your task has three parts:

1. RECONSTRUCT: Attempt to reconstruct what the original Arabic text most likely says. Correct obvious OCR errors. If a word is unrecoverable noise, replace it with [unclear]. Preserve the structure and meaning as best you can.

2. TRANSLATE: Provide a careful English translation of your reconstructed Arabic. Use scholarly register appropriate for historical documents. Where you are uncertain, add a note in parentheses.

3. ASSESSMENT: Briefly note the overall quality of this chunk. How much was recoverable? What type of document does this appear to be (administrative, legal, personal, religious, etc.)? What time period or context does the language suggest, if discernible?

Format your response EXACTLY as follows with these three headers:

RECONSTRUCTED ARABIC:
[your reconstructed Arabic text]

ENGLISH TRANSLATION:
[your English translation]

ASSESSMENT:
[your brief scholarly assessment]

Here is chunk {chunk_num} of {total}:

{chunk_text_str}"""

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    # ── Helper: parse Claude response ───────────────────────
    def parse_claude_response(response_text):
        result = {
            "reconstructed": "",
            "translation": "",
            "assessment": ""
        }
        sections = {
            "RECONSTRUCTED ARABIC:": "reconstructed",
            "ENGLISH TRANSLATION:": "translation",
            "ASSESSMENT:": "assessment"
        }
        current_key = None
        lines = response_text.split("\n")
        for line in lines:
            stripped = line.strip()
            matched = False
            for header, key in sections.items():
                if stripped.startswith(header):
                    current_key = key
                    remainder = stripped[len(header):].strip()
                    if remainder:
                        result[current_key] = remainder
                    matched = True
                    break
            if not matched and current_key:
                result[current_key] = (result[current_key] + "\n" + line).strip()
        return result

    # ── Check for a recently processed document ─────────────
    last_doc_id = st.session_state.get("last_processed_doc_id", None)

    if last_doc_id is None:
        st.info("Upload and process a document in the **📄 Process Document** tab first. The AI analysis will appear here automatically.")
    else:
        # Check if we already have a saved claude_analysis for this doc
        import sqlite3 as _sq
        _c = _sq.connect(DB_PATH)
        existing = _c.execute(
            "SELECT summary_text FROM summaries WHERE document_id=? AND summary_type='claude_analysis'",
            (last_doc_id,)
        ).fetchone()
        doc_row = _c.execute(
            "SELECT source, ocr_text_raw FROM documents WHERE id=?",
            (last_doc_id,)
        ).fetchone()
        _c.close()

        if doc_row is None:
            st.error("Could not find the processed document in the database.")
        else:
            doc_source, ocr_raw = doc_row
            st.markdown(f"**Analyzing:** `{doc_source}`")
            st.divider()

            if existing:
                # Load saved analysis
                st.success("✓ Analysis loaded from archive.")
                try:
                    saved = _json.loads(existing[0])
                    chunks_data = saved.get("chunks", [])
                except Exception:
                    chunks_data = []
                    st.warning("Could not parse saved analysis. Re-run to regenerate.")

                for i, chunk in enumerate(chunks_data):
                    with st.expander(f"Segment {i+1} of {len(chunks_data)}", expanded=(i == 0)):
                        col_a, col_b, col_c = st.columns([1, 1, 1])
                        with col_a:
                            st.markdown("**🔤 Original OCR Text**")
                            st.markdown(
                                f"<div style='background:#1a1a2e;padding:12px;border-radius:8px;"
                                f"font-family:serif;direction:rtl;text-align:right;"
                                f"font-size:0.9em;color:#ccc;min-height:120px'>{chunk.get('original','')}</div>",
                                unsafe_allow_html=True
                            )
                        with col_b:
                            st.markdown("**🔧 Claude's Reconstruction + Translation**")
                            st.markdown(
                                f"<div style='background:#0d2137;padding:12px;border-radius:8px;"
                                f"font-size:0.9em;color:#e8e8e8;min-height:120px'>"
                                f"<strong>Arabic:</strong><br>"
                                f"<span style='direction:rtl;display:block;text-align:right;font-family:serif'>"
                                f"{chunk.get('reconstructed','')}</span><br>"
                                f"<strong>English:</strong><br>{chunk.get('translation','')}</div>",
                                unsafe_allow_html=True
                            )
                        with col_c:
                            st.markdown("**📋 Scholarly Assessment**")
                            st.markdown(
                                f"<div style='background:#0d3320;padding:12px;border-radius:8px;"
                                f"font-size:0.9em;color:#c8e6c9;min-height:120px'>"
                                f"{chunk.get('assessment','')}</div>",
                                unsafe_allow_html=True
                            )

            else:
                # Run fresh analysis
                # Quick Claude connection test
                if st.button("Test Claude connection", key="test_claude_connection"):
                    try:
                        api_key = st.secrets.get("ANTHROPIC_API_KEY", "").strip()
                        if not api_key:
                            st.error("Missing ANTHROPIC_API_KEY in .streamlit/secrets.toml")
                        elif not api_key.startswith("sk-ant-"):
                            st.error("Anthropic API key format looks wrong.")
                        else:
                            client = anthropic.Anthropic(api_key=api_key)
                            resp = client.messages.create(
                                model="claude-sonnet-4-6",
                                max_tokens=20,
                                messages=[{"role": "user", "content": "Reply only with OK"}]
                            )
                            st.success(resp.content[0].text)
                    except Exception as e:
                        st.error(repr(e))

                run_analysis = st.button("🤖 Run AI Analysis", type="primary", use_container_width=True)

                if run_analysis:
                    try:
                        api_key = st.secrets["ANTHROPIC_API_KEY"]
                    except Exception:
                        st.error("API key not found. Make sure .streamlit/secrets.toml contains ANTHROPIC_API_KEY.")
                        st.stop()

                    client = anthropic.Anthropic(api_key=api_key)

                    if not ocr_raw or not ocr_raw.strip():
                        st.error("No OCR text found for this document.")
                        st.stop()

                    chunks = chunk_text(ocr_raw, size=400)
                    total = len(chunks)
                    st.info(f"Analyzing {total} segment(s)...")

                    progress = st.progress(0)
                    chunks_data = []
                    failed_chunks = []

                    for i, chunk in enumerate(chunks):
                        with st.spinner(f"Claude is analyzing segment {i+1} of {total}..."):
                            try:
                                response = analyze_chunk(client, chunk, i+1, total)
                                parsed = parse_claude_response(response)
                                parsed["original"] = chunk
                                chunks_data.append(parsed)
                            except Exception as e:
                                failed_chunks.append(
                                    {"segment": i + 1, "error": str(e), "original": chunk}
                                )
                                chunks_data.append({
                                    "original": chunk,
                                    "reconstructed": "",
                                    "translation": "",
                                    "assessment": f"Analysis failed for segment {i+1}: {e}"
                                })
                        progress.progress((i + 1) / total)

                    progress.empty()

                    if failed_chunks:
                        st.error(
                            f"Analysis failed for {len(failed_chunks)} of {total} segment(s). "
                            "Results were NOT saved to the archive. Please retry."
                        )

                        with st.expander("Show failed segments"):
                            for f in failed_chunks:
                                st.markdown(f"**Segment {f['segment']}**")
                                st.code(f["error"])
                        
                        for i, chunk in enumerate(chunks_data):
                            with st.expander(f"Segment {i+1} of {total}", expanded=(i == 0)):
                                col_a, col_b, col_c = st.columns([1, 1, 1])
                                with col_a:
                                    st.markdown("**🔤 Original OCR Text**")
                                    st.markdown(
                                        f"<div style='background:#1a1a2e;padding:12px;border-radius:8px;"
                                        f"font-family:serif;direction:rtl;text-align:right;"
                                        f"font-size:0.9em;color:#ccc;min-height:120px'>{chunk.get('original','')}</div>",
                                        unsafe_allow_html=True
                                    )
                                with col_b:
                                    st.markdown("**🔧 Claude's Reconstruction + Translation**")
                                    st.markdown(
                                        f"<div style='background:#0d2137;padding:12px;border-radius:8px;"
                                        f"font-size:0.9em;color:#e8e8e8;min-height:120px'>"
                                        f"<strong>Arabic:</strong><br>"
                                        f"<span style='direction:rtl;display:block;text-align:right;font-family:serif'>"
                                        f"{chunk.get('reconstructed','')}</span><br>"
                                        f"<strong>English:</strong><br>{chunk.get('translation','')}</div>",
                                        unsafe_allow_html=True
                                    )
                                with col_c:
                                    st.markdown("**📋 Scholarly Assessment**")
                                    st.markdown(
                                        f"<div style='background:#0d3320;padding:12px;border-radius:8px;"
                                        f"font-size:0.9em;color:#c8e6c9;min-height:120px'>"
                                        f"{chunk.get('assessment','')}</div>",
                                        unsafe_allow_html=True
                                    )
                    else:
                        upsert_summary(
                            conn,
                            last_doc_id,
                            'claude_analysis',
                            _json.dumps({"chunks": chunks_data}),
                            method='claude-sonnet-4-6',
                            quality_note='AI reconstruction, translation, and assessment'
                        )

                        st.success("✓ Analysis complete and saved to archive.")

                        for i, chunk in enumerate(chunks_data):
                            with st.expander(f"Segment {i+1} of {total}", expanded=(i == 0)):
                                col_a, col_b, col_c = st.columns([1, 1, 1])
                                with col_a:
                                    st.markdown("**🔤 Original OCR Text**")
                                    st.markdown(
                                        f"<div style='background:#1a1a2e;padding:12px;border-radius:8px;"
                                        f"font-family:serif;direction:rtl;text-align:right;"
                                        f"font-size:0.9em;color:#ccc;min-height:120px'>{chunk.get('original','')}</div>",
                                        unsafe_allow_html=True
                                    )
                                with col_b:
                                    st.markdown("**🔧 Claude's Reconstruction + Translation**")
                                    st.markdown(
                                        f"<div style='background:#0d2137;padding:12px;border-radius:8px;"
                                        f"font-size:0.9em;color:#e8e8e8;min-height:120px'>"
                                        f"<strong>Arabic:</strong><br>"
                                        f"<span style='direction:rtl;display:block;text-align:right;font-family:serif'>"
                                        f"{chunk.get('reconstructed','')}</span><br>"
                                        f"<strong>English:</strong><br>{chunk.get('translation','')}</div>",
                                        unsafe_allow_html=True
                                    )
                                with col_c:
                                    st.markdown("**📋 Scholarly Assessment**")
                                    st.markdown(
                                        f"<div style='background:#0d3320;padding:12px;border-radius:8px;"
                                        f"font-size:0.9em;color:#c8e6c9;min-height:120px'>"
                                        f"{chunk.get('assessment','')}</div>",
                                        unsafe_allow_html=True
                                    )
