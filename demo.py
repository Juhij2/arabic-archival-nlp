import streamlit as st
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import chromadb
from sentence_transformers import SentenceTransformer
from camel_tools.ner import NERecognizer
from camel_tools.tokenizers.word import simple_word_tokenize
from transformers import MarianMTModel, MarianTokenizer
import re
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Arabic Archival NLP Pipeline",
    page_icon="🕌",
    layout="wide"
)

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
        except:
            models['collection'] = None
    with st.spinner("Loading NER model..."):
        try:
            models['ner'] = NERecognizer.pretrained()
        except:
            models['ner'] = None
    with st.spinner("Loading translator..."):
        try:
            model_name = "Helsinki-NLP/opus-mt-ar-en"
            models['tokenizer'] = MarianTokenizer.from_pretrained(model_name)
            models['translator'] = MarianMTModel.from_pretrained(model_name)
        except:
            models['tokenizer'] = None
            models['translator'] = None
    return models

def preprocess_image(img):
    img = img.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.5)
    img = img.filter(ImageFilter.SHARPEN)
    w, h = img.size
    if w < 1200:
        scale = 1200 / w
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return img

def clean_arabic(text):
    text = re.sub('[إأآا]', 'ا', text)
    text = re.sub('[يى]', 'ي', text)
    arabic_pattern = re.compile(
        r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s\.\،\,\:\!\?]'
    )
    text = arabic_pattern.sub(' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 3]
    return '\n'.join(lines).strip()

def prepare_for_translation(text):
    """
    Extra cleaning specifically before translation.
    Removes all non-Arabic characters that break Helsinki model.
    """
    # Keep only Arabic letters, spaces, and basic punctuation
    clean = re.sub(
        r'[^\u0600-\u06FF\s\،\.\,]',
        ' ',
        text
    )
    # Remove very short tokens that are OCR noise
    words = clean.split()
    words = [w for w in words if len(w) >= 2]
    clean = ' '.join(words)
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
                chunk, return_tensors="pt",
                padding=True, truncation=True, max_length=512
            )
            translated = models['translator'].generate(**inputs)
            result = models['tokenizer'].decode(
                translated[0], skip_special_tokens=True)
            results.append(result)
        return ' '.join(results)
    except Exception as e:
        return f"Translation error: {e}"

def extract_entities(text, models):
    result = {"PERSON": [], "LOCATION": [], "ORGANIZATION": []}
    if not models.get('ner') or not text.strip():
        return result
    try:
        tokens = simple_word_tokenize(text)
        if not tokens:
            return result
        labels = models['ner'].predict_sentence(tokens)
        type_map = {
            "PERS": "PERSON", "LOC": "LOCATION", "ORG": "ORGANIZATION"
        }
        current = []
        current_type = None
        for token, label in zip(tokens, labels):
            if label.startswith('B-'):
                if current and current_type:
                    result.setdefault(
                        current_type, []).append(' '.join(current))
                current_type = type_map.get(label[2:], label[2:])
                current = [token]
            elif label.startswith('I-') and current:
                current.append(token)
            else:
                if current and current_type:
                    result.setdefault(
                        current_type, []).append(' '.join(current))
                current = []
                current_type = None
        if current and current_type:
            result.setdefault(
                current_type, []).append(' '.join(current))
    except Exception as e:
        st.error(f"NER error: {e}")
    return result

def search_docs(query, models):
    if not models.get('collection'):
        return []
    try:
        qe = models['embedder'].encode(query).tolist()
        results = models['collection'].query(
            query_embeddings=[qe], n_results=3)
        distances = results['distances'][0]
        min_d = min(distances)
        max_d = max(distances)
        output = []
        for meta, dist, doc in zip(
            results['metadatas'][0],
            distances,
            results['documents'][0]
        ):
            sim = round(
                (1-(dist-min_d)/(max_d-min_d+0.001))*100, 1)
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

# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────

st.title("🕌 Arabic Archival NLP Pipeline")
st.markdown("""
**Proof of concept for Arabic historical document analysis**

**Layer 1:** OCR → Text Cleaning → Named Entity Recognition → Semantic Search

**Layer 2:** Arabic → English Translation (applied to cleaned Layer 1 output)
""")
st.divider()

models = load_models()
st.success("✓ All models loaded and ready!")

tab1, tab2 = st.tabs(["📄 Process Document", "🔍 Search Documents"])

with tab1:
    st.markdown("### Upload a scanned Arabic document")
    st.markdown("""
    The pipeline extracts text, identifies named entities,
    translates to English, and finds similar documents.
    """)

    uploaded_file = st.file_uploader(
        "Choose an Arabic document image",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])

        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Document",
                     use_container_width=True)
            run_btn = st.button(
                "🔍 Run Full Pipeline",
                type="primary",
                use_container_width=True
            )

        with col2:
            if run_btn:

                # LAYER 1: OCR
                with st.spinner("Running OCR..."):
                    processed = preprocess_image(image)
                    raw_text = pytesseract.image_to_string(
                        processed, lang='ara',
                        config='--oem 3 --psm 6'
                    )
                    clean_text = clean_arabic(raw_text)

                if not clean_text.strip():
                    st.error("No Arabic text detected in this image")
                else:
                    st.markdown("#### 📝 Layer 1: Extracted Arabic Text")
                    st.text_area(
                        "Arabic OCR Output (cleaned)",
                        clean_text,
                        height=150
                    )

                    # LAYER 2: Translation
                    st.markdown("#### 🌐 Layer 2: English Translation")
                    st.caption(
                        "Translation applied to cleaned Arabic text only"
                    )
                    with st.spinner("Translating..."):
                        translation_input = prepare_for_translation(
                            clean_text)
                        translation = translate_arabic(
                            translation_input, models)
                    st.text_area(
                        "English Translation",
                        translation,
                        height=150
                    )

                    # Show what was sent to translator
                    with st.expander(
                        "🔍 View cleaned input sent to translator"
                    ):
                        st.text(translation_input)

                    # LAYER 1: NER
                    st.markdown("#### 🏷 Layer 1: Named Entities")
                    with st.spinner("Extracting entities..."):
                        entities = extract_entities(clean_text, models)

                    ner_col1, ner_col2, ner_col3 = st.columns(3)
                    with ner_col1:
                        st.markdown("**👤 People**")
                        if entities['PERSON']:
                            for e in entities['PERSON']:
                                st.markdown(f"• {e}")
                        else:
                            st.markdown("*None found*")
                    with ner_col2:
                        st.markdown("**📍 Locations**")
                        if entities['LOCATION']:
                            for e in entities['LOCATION']:
                                st.markdown(f"• {e}")
                        else:
                            st.markdown("*None found*")
                    with ner_col3:
                        st.markdown("**🏛 Organizations**")
                        if entities['ORGANIZATION']:
                            for e in entities['ORGANIZATION']:
                                st.markdown(f"• {e}")
                        else:
                            st.markdown("*None found*")

                    # LAYER 1: Search
                    st.markdown("#### 🔎 Layer 1: Similar Documents")
                    with st.spinner("Searching..."):
                        results = search_docs(clean_text, models)
                    medals = ["🥇", "🥈", "🥉"]
                    for i, r in enumerate(results):
                        with st.expander(
                            f"{medals[i]} {r['source']} "
                            f"({r['similarity']}% match)"
                        ):
                            st.markdown(f"**Type:** {r['description']}")
                            if r['persons']:
                                st.markdown(
                                    f"**👤 People:** {r['persons']}")
                            if r['locations']:
                                st.markdown(
                                    f"**📍 Places:** {r['locations']}")
                            st.markdown(f"**Preview:** {r['preview']}...")

with tab2:
    st.markdown("### Search across all indexed documents")
    st.markdown("""
    Search in **English or Arabic**.
    The system finds relevant documents by meaning, not just keywords.
    """)

    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Search query",
            placeholder="Try: 'names of people' or 'أسماء الأشخاص'"
        )
    with col2:
        search_btn = st.button(
            "Search", type="primary", use_container_width=True)

    st.markdown("**Try these example queries:**")
    ex1, ex2, ex3, ex4 = st.columns(4)
    with ex1:
        if st.button("👤 names of people"):
            query = "names of people and family records"
    with ex2:
        if st.button("🏛 official document"):
            query = "official government document"
    with ex3:
        if st.button("📜 historical manuscript"):
            query = "historical manuscript dense text"
    with ex4:
        if st.button("🔤 أسماء الأشخاص"):
            query = "أسماء الأشخاص"

    if query and (search_btn or query):
        with st.spinner("Searching..."):
            results = search_docs(query, models)
        medals = ["🥇", "🥈", "🥉"]
        st.markdown(f"### Results for: *{query}*")
        st.divider()
        for i, r in enumerate(results):
            with st.expander(
                f"{medals[i]} {r['source']} ({r['similarity']}% match)",
                expanded=(i == 0)
            ):
                st.markdown(f"**Type:** {r['description']}")
                if r['persons']:
                    st.markdown(f"**👤 People:** {r['persons']}")
                if r['locations']:
                    st.markdown(f"**📍 Places:** {r['locations']}")
                st.markdown(f"**Text preview:** {r['preview']}...")