import gradio as gr
import json
import os
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import chromadb
from sentence_transformers import SentenceTransformer
from camel_tools.ner import NERecognizer
from camel_tools.tokenizers.word import simple_word_tokenize
import re
import warnings
warnings.filterwarnings('ignore')

print("Loading models...")

embedder = SentenceTransformer(
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
)
print("  ✓ Embedder loaded")

chroma_client = chromadb.PersistentClient(path="data/chromadb")
try:
    collection = chroma_client.get_collection("arabic_documents")
    print(f"  ✓ ChromaDB loaded: {collection.count()} docs")
except:
    collection = None
    print("  ✗ No ChromaDB collection")

try:
    ner = NERecognizer.pretrained()
    print("  ✓ NER loaded")
except:
    ner = None
    print("  ✗ NER failed")

try:
    from transformers import MarianMTModel, MarianTokenizer
    model_name = "Helsinki-NLP/opus-mt-ar-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    translation_model = MarianMTModel.from_pretrained(model_name)
    print("  ✓ Translator loaded")
    HAS_TRANSLATOR = True
except Exception as e:
    print(f"  ✗ Translator failed: {e}")
    HAS_TRANSLATOR = False

print("✓ All models ready!")

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

def translate_arabic(text):
    if not HAS_TRANSLATOR or not text.strip():
        return "Translation not available"
    try:
        # Split into small chunks
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
        for chunk in chunks[:5]:  # max 5 chunks
            inputs = tokenizer(
                chunk, return_tensors="pt",
                padding=True, truncation=True,
                max_length=512
            )
            translated = translation_model.generate(**inputs)
            result = tokenizer.decode(
                translated[0], skip_special_tokens=True)
            results.append(result)
        
        return ' '.join(results)
    except Exception as e:
        return f"Translation error: {e}"

def extract_entities(text):
    result = {"PERSON": [], "LOCATION": [], "ORGANIZATION": []}
    if ner is None or not text.strip():
        return result
    try:
        tokens = simple_word_tokenize(text)
        if not tokens:
            return result
        labels = ner.predict_sentence(tokens)
        type_map = {
            "PERS": "PERSON", "LOC": "LOCATION", "ORG": "ORGANIZATION"
        }
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
        print(f"NER error: {e}")
    return result

def process_document(image):
    if image is None:
        return "Please upload an image", "", "", ""
    
    try:
        # LAYER 1: OCR
        processed = preprocess_image(image)
        raw_text = pytesseract.image_to_string(
            processed, lang='ara', config='--oem 3 --psm 6')
        clean_text = clean_arabic(raw_text)
        
        if not clean_text.strip():
            return "No Arabic text detected", "", "", ""
        
        # LAYER 1: NER
        entities = extract_entities(clean_text)
        ent_display = ""
        icons = {"PERSON": "👤", "LOCATION": "📍", "ORGANIZATION": "🏛"}
        for etype, elist in entities.items():
            if elist:
                ent_display += f"{icons.get(etype, '•')} {etype}\n"
                for e in elist[:6]:
                    ent_display += f"  • {e}\n"
                ent_display += "\n"
        if not ent_display:
            ent_display = "No entities found"
        
        # LAYER 2: Translation
        translation = translate_arabic(clean_text)
        
        # LAYER 1: Search
        search_display = ""
        if collection is not None:
            try:
                qe = embedder.encode(clean_text).tolist()
                results = collection.query(
                    query_embeddings=[qe], n_results=3)
                distances = results['distances'][0]
                min_d = min(distances)
                max_d = max(distances)
                medals = ["🥇", "🥈", "🥉"]
                search_display = "Similar documents:\n\n"
                for i, (meta, dist) in enumerate(
                    zip(results['metadatas'][0], distances)
                ):
                    sim = round(
                        (1-(dist-min_d)/(max_d-min_d+0.001))*100, 1)
                    search_display += f"{medals[i]} {meta['source']} ({sim}%)\n"
                    if meta.get('persons'):
                        search_display += f"  👤 {meta['persons'][:60]}\n"
                    if meta.get('locations'):
                        search_display += f"  📍 {meta['locations'][:60]}\n"
                    search_display += "\n"
            except Exception as e:
                search_display = f"Search error: {e}"
        
        return clean_text, translation, ent_display, search_display
    
    except Exception as e:
        return f"Error: {e}", "", "", ""

def search_only(query):
    if not query.strip():
        return "Please enter a search query"
    if collection is None:
        return "No documents indexed"
    try:
        qe = embedder.encode(query).tolist()
        results = collection.query(query_embeddings=[qe], n_results=3)
        distances = results['distances'][0]
        min_d = min(distances)
        max_d = max(distances)
        medals = ["🥇", "🥈", "🥉"]
        output = f"Results for: '{query}'\n\n"
        for i, (doc_id, doc, meta, dist) in enumerate(zip(
            results['ids'][0], results['documents'][0],
            results['metadatas'][0], distances
        )):
            sim = round((1-(dist-min_d)/(max_d-min_d+0.001))*100, 1)
            output += f"{medals[i]} {meta['source']} ({sim}%)\n"
            output += f"  {meta['description']}\n"
            if meta.get('persons'):
                output += f"  👤 {meta['persons'][:70]}\n"
            if meta.get('locations'):
                output += f"  📍 {meta['locations'][:70]}\n"
            output += f"  📄 {doc[:120]}...\n\n"
        return output
    except Exception as e:
        return f"Search error: {e}"

with gr.Blocks(title="Arabic Archival NLP", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🕌 Arabic Archival NLP Pipeline
    **Layer 1:** OCR → Cleaning → NER → Semantic Search  
    **Layer 2:** Arabic → English Translation
    """)
    
    with gr.Tabs():
        with gr.Tab("📄 Process Document"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(type="pil", label="Upload Arabic Document")
                    btn = gr.Button("🔍 Run Pipeline", variant="primary")
                with gr.Column(scale=2):
                    arabic_out = gr.Textbox(label="Arabic Text (Layer 1)", lines=6)
                    english_out = gr.Textbox(label="English Translation (Layer 2)", lines=6)
                    entities_out = gr.Textbox(label="Named Entities (Layer 1)", lines=6)
                    search_out = gr.Textbox(label="Similar Documents (Layer 1)", lines=6)
            
            btn.click(
                fn=process_document,
                inputs=[img_input],
                outputs=[arabic_out, english_out, entities_out, search_out]
            )
        
        with gr.Tab("🔍 Search"):
            query_input = gr.Textbox(
                placeholder="Search in English or Arabic...",
                label="Query"
            )
            search_btn = gr.Button("Search", variant="primary")
            search_output = gr.Textbox(label="Results", lines=15)
            
            search_btn.click(
                fn=search_only,
                inputs=[query_input],
                outputs=[search_output]
            )
            
            gr.Examples(
                examples=[
                    ["names of people and family records"],
                    ["official government document"],
                    ["أسماء الأشخاص"],
                ],
                inputs=query_input
            )

if __name__ == "__main__":
    demo.launch(share=True, show_error=True)