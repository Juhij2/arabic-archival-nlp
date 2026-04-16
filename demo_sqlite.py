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

tab1, tab2, tab3 = st.tabs(["📄 Process Document", "🔍 Search Documents", "🗄 Archive Browser"])

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
