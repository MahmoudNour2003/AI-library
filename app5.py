import streamlit as st
from pymarc import Record, Field, Subfield, MARCWriter, XMLWriter, MARCReader, parse_xml_to_array
import io
from io import BytesIO, StringIO
import requests
import os
import datetime
from lxml import etree
import tempfile
import json
import fitz  # PyMuPDF
from langdetect import detect
import re
from pathlib import Path
import shutil
import hashlib
from functools import lru_cache
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle

# --- Directory and Path Configuration ---
output_dir = "output"
XML_DB_DIR = "xml_database"

Path(output_dir).mkdir(parents=True, exist_ok=True)
Path(XML_DB_DIR).mkdir(parents=True, exist_ok=True)

# --- Q&A Bot Constants and Model Loading ---
FAISS_INDEX_PATH = "faiss_index.bin"
DOCUMENTS_PATH = "documents.pkl"
METADATA_PATH = "metadata.pkl"
# Load the embedding model once
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

EMBEDDING_MODEL = load_embedding_model()


# --- Session State Initialization ---
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = {}
if 'current_record' not in st.session_state:
    st.session_state.current_record = None
if 'llm_metadata' not in st.session_state:
    st.session_state.llm_metadata = None
if 'processed_file_hash' not in st.session_state:
    st.session_state.processed_file_hash = None


# --- Helper Functions (Existing) ---
def export_xml(record):
    xml_file = io.BytesIO()
    writer = XMLWriter(xml_file)
    writer.write(record)
    writer.close()
    return xml_file.getvalue()

def generate_control_number():
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S')

def export_txt(record):
    return record.as_marc21().encode('utf-8')

def sf(*args):
    return [Subfield(code=code, value=value) for code, value in zip(args[::2], args[1::2]) if value.strip()]

GROBID_URL = "https://kermitt2-grobid.hf.space/api/processFulltextDocument"

def send_pdf_to_grobid_header(pdf_path):
    with open(pdf_path, 'rb') as f:
        files = {'input': (os.path.basename(pdf_path), f, 'application/pdf')}
        headers = {'Accept': 'application/tei+xml, text/xml, application/xml;q=0.9, */*;q=0.8'}
        params = {
            'consolidateHeader': '1', 'consolidateCitations': '0',
            'teiCoordinates': 'persName,figure,ref,biblStruct', 'generateIDs': '1',
            'segmentSentences': '1', 'includeRawAffiliations': '1',
            'includeRawAuthors': '1', 'includeRawCitations': '0'
        }
        response = requests.post(GROBID_URL, files=files, headers=headers, params=params, timeout=120)
        print("[*] Status code:", response.status_code)
        if response.status_code == 200 and response.text.strip().startswith("<"):
            return response.text
        else:
            raise Exception("GROBID did not return valid TEI XML")

def extract_text_from_pdf(pdf_path, max_chars=1500):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
        if len(text) >= max_chars:
            break
    return text[:max_chars]

def ask_groq_model(prompt, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "gemma2-9b-it",
        "messages": [
            {"role": "system", "content": "You are an academic assistant that extracts metadata from papers."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        raise Exception(f"Groq Error {response.status_code}: {response.text}")

def extract_json_block(text):
    try:
        json_match = re.search(r'{[\s\S]*?}', text)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            raise ValueError("No valid JSON block found.")
    except json.JSONDecodeError as e:
        print("[!] JSON decoding error:", str(e))
        print("Raw content:\n", text)
        raise

def generate_008_field(pub_year: str, lang_code: str) -> str:
    today = datetime.datetime.utcnow().strftime('%y%m%d')
    pub_year = pub_year if pub_year.isdigit() else '2024'
    lang_code = lang_code[:3].lower().ljust(3, '#')
    return f"{today}s{pub_year}####xx###########{lang_code}#d"

def extract_physical_description(pdf_path):
    doc = fitz.open(pdf_path)
    return f"{len(doc)} pages ; 1 online resource."

def extract_publisher_and_pubyear(root, ns):
    print("[📍DEBUG] Extracting <publisher> and <date type='published'> from <publicationStmt>...")
    publication_stmt = root.find('./tei:teiHeader/tei:fileDesc/tei:publicationStmt', ns)
    publisher = "[Unknown Publisher]"
    pub_year = "NONE"
    if publication_stmt is not None:
        pub_node = publication_stmt.find('{http://www.tei-c.org/ns/1.0}publisher')
        if pub_node is not None and pub_node.text and pub_node.text.strip():
            publisher = pub_node.text.strip()
            print(f"[✅] Publisher found: {publisher}")
        else:
            print("[❌] Publisher not found inside <publicationStmt>")
        date_node = publication_stmt.find('{http://www.tei-c.org/ns/1.0}date')
        if date_node is not None and date_node.get("when"):
            pub_year = date_node.get("when")[:4]
            print(f"[✅] Publication year extracted: {pub_year}")
        else:
            print("[❌] Date not found or missing 'when' attribute")
    else:
        print("[❌] <publicationStmt> not found")
    return publisher, pub_year

def fetch_metadata_from_crossref(doi):
    print(f"[🌐] Querying Crossref for DOI: {doi}")
    url = f"https://api.crossref.org/works/{doi}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json().get("message", {})
            publisher = data.get("publisher")
            pub_year = "NONE"
            isbn_list = data.get("ISBN", [])
            if isbn_list:
                print("ISBN from Crossref:", isbn_list[0])
            if "published-print" in data and "date-parts" in data["published-print"]:
                pub_year = str(data["published-print"]["date-parts"][0][0])
            elif "published-online" in data and "date-parts" in data["published-online"]:
                pub_year = str(data["published-online"]["date-parts"][0][0])
            elif "created" in data and "date-parts" in data["created"]:
                pub_year = str(data["created"]["date-parts"][0][0])
            print(f"[✅] Crossref publisher: {publisher}, year: {pub_year}")
            return publisher, pub_year
        else:
            print(f"[❌] Crossref error {resp.status_code}")
    except Exception as e:
        print("[❌] Crossref exception:", str(e))
    return None, None

def format_authors(authors):
    if len(authors) == 1:
        return authors[0]
    elif len(authors) == 2:
        return f"{authors[0]} and {authors[1]}"
    else:
        return ', '.join(authors[:-1]) + f", and {authors[-1]}"

def tei_to_marc(tei_xml, marc_path, pdf_path, llm_metadata):
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    root = etree.fromstring(tei_xml.encode('utf-8'))
    record = Record(force_utf8=True)
    record.leader = "     nam a22     uu 4500"
    record.add_field(Field(tag='001', data=root.findtext('.//tei:idno[@type="MD5"]', default='000000000', namespaces=ns)))
    record.add_field(Field(tag='003', data='HybridExtractor'))
    record.add_field(Field(tag='005', data=datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S.0')))
    pub_year = llm_metadata.get("publication year", "2024")
    lang_sample = llm_metadata.get("title", "")
    try:
        lang_code = detect(lang_sample)
    except:
        lang_code = 'en'
    record.add_field(Field(tag='008', data=generate_008_field(pub_year, lang_code)))
    doi = root.find('./tei:teiHeader//tei:idno[@type="DOI"]', namespaces=ns)
    if doi is not None:
        record.add_field(Field(tag='024', indicators=['7', '#'], subfields=[Subfield('a', doi.text.strip()), Subfield('2', 'doi')]))
    isbn = root.find('.//tei:idno[@type="ISBN"]', ns)
    if isbn is not None:
        record.add_field(Field(tag='020', indicators=['#', '#'], subfields=[Subfield('a', isbn.text.strip())]))
    language_names = {'en': 'English', 'ar': 'Arabic', 'fr': 'French'}
    lang_name = language_names.get(lang_code, lang_code.capitalize())
    record.add_field(Field(tag='040', indicators=['#', '#'], subfields=[Subfield('a', 'HybridExtractor'), Subfield('b', lang_code), Subfield('e', 'rda'), Subfield('c', 'HybridExtractor')]))
    record.add_field(Field(tag='041', indicators=['0', '#'], subfields=[Subfield('a', lang_code)]))
    record.add_field(Field(tag='546', indicators=['#', '#'], subfields=[Subfield('a', f'Text in {lang_name}.')]))
    title = llm_metadata.get("title", "[Title not available]")
    main_title, subtitle = (title.split(':', 1) + [''])[:2]
    authors = llm_metadata.get("authors", [])
    if isinstance(authors, str):
        authors = [authors]
    record.add_field(Field(tag='245', indicators=['1', '0'], subfields=[Subfield('a', main_title.strip() + ' :'), Subfield('b', subtitle.strip()), Subfield('c', format_authors(authors) + '.')]))
    for i, author in enumerate(authors):
        tag = '100' if i == 0 else '700'
        record.add_field(Field(tag=tag, indicators=['1', '#'], subfields=[Subfield('a', author), Subfield('e', 'author.')]))
    abstract_text = ""

    # Try to extract <abstract> from TEI
    abstract_div = root.find('.//tei:abstract/tei:div', ns)
    if abstract_div is not None:
        paragraphs = abstract_div.findall('.//tei:p', ns)
        abstract_parts = [p.text.strip() for p in paragraphs if p.text and p.text.strip()]
        abstract_text = '\n\n'.join(abstract_parts)

    # If no abstract or it's empty, fallback to intro-like section
    if not abstract_text:
        print("[ℹ️] No abstract found. Looking for introduction-like sections...")
        intro_keywords = ['intro', 'background', 'overview', 'context']
        for div in root.findall('.//tei:text//tei:div', ns):
            head = div.find('tei:head', ns)
            if head is not None and head.text:
                head_text = head.text.strip().lower()
                if any(keyword in head_text for keyword in intro_keywords):
                    paragraphs = div.findall('.//tei:p', ns)
                    intro_parts = [p.text.strip() for p in paragraphs if p.text and p.text.strip()]
                    abstract_text = '\n\n'.join(intro_parts)
                    if abstract_text:
                        print(f"[✅] Using section '{head_text}' as abstract fallback.")
                        break

    # Add 520 if we found something
    if abstract_text:
        record.add_field(Field(tag='520', indicators=['#', '#'], subfields=[Subfield('a', abstract_text)]))
    else:
        print("[⚠️] No abstract or introduction-like section found. Field 520 will be skipped.")


    publisher = '[Unknown Publisher]'
    publisher, pub_year = extract_publisher_and_pubyear(root, ns)
    if ((not publisher or publisher == "[Unknown Publisher]") or (not pub_year or pub_year == "NONE")) and doi is not None:
        doi_node = root.find('.//tei:idno[@type="DOI"]', ns)
        if doi_node is not None and doi_node.text:
            cr_publisher, cr_year = fetch_metadata_from_crossref(doi_node.text.strip())
            if cr_publisher and (not publisher or publisher == "[Unknown Publisher]"):
                publisher = cr_publisher
                print("[⚠️] Publisher updated from Crossref.")
            if cr_year and (not pub_year or pub_year == "NONE"):
                pub_year = cr_year
                print("[⚠️] Publication year updated from Crossref.")
    record.add_field(Field(tag='264', indicators=['#', '1'], subfields=[Subfield('a', '[Place of publication not identified]'), Subfield('b', publisher), Subfield('c', pub_year)]))
    record.add_field(Field(tag='300', indicators=['#', '#'], subfields=[Subfield('a', extract_physical_description(pdf_path))]))
    record.add_field(Field(tag='336', indicators=['#', '#'], subfields=[Subfield('a', 'text'), Subfield('b', 'txt'), Subfield('2', 'rdacontent')]))
    record.add_field(Field(tag='337', indicators=['#', '#'], subfields=[Subfield('a', 'computer'), Subfield('b', 'c'), Subfield('2', 'rdamedia')]))
    record.add_field(Field(tag='338', indicators=['#', '#'], subfields=[Subfield('a', 'online resource'), Subfield('b', 'cr'), Subfield('2', 'rdacarrier')]))
    monogr_title = root.findtext('.//tei:monogr/tei:title[@level="j"]', namespaces=ns)
    volume = root.findtext('.//tei:monogr/tei:biblScope[@unit="volume"]', namespaces=ns)
    issue = root.findtext('.//tei:monogr/tei:biblScope[@unit="issue"]', namespaces=ns)
    if monogr_title:
        subfields = [Subfield('t', monogr_title)]
        if volume: subfields.append(Subfield('g', f"Vol. {volume}"))
        if issue: subfields.append(Subfield('g', f"No. {issue}"))
        record.add_field(Field(tag='773', indicators=['0', ' '], subfields=subfields))
    if doi is not None:
        record.add_field(Field(tag='856', indicators=['4', '0'], subfields=[Subfield('u', f"https://doi.org/{doi.text.strip()}"), Subfield('y', 'Full text via DOI')]))
    record.fields = sorted(record.fields, key=lambda f: int(f.tag) if f.tag.isdigit() else float('inf'))
    base_path = marc_path.replace('.mrc', '')
    marc_path = base_path + '.mrc'
    with open(marc_path, 'wb') as f:
        f.write(record.as_marc())
    txt_path = base_path + '.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(str(record))
    marcxml_path = base_path + '.xml'
    with open(marcxml_path, 'wb') as f:
        writer = XMLWriter(f)
        writer.write(record)
        writer.close()
    print(f"\n[💾] Binary MARC: {marc_path}")
    print(f"[💾] Text MARC:   {txt_path}")
    print(f"[💾] MARC XML:    {marcxml_path}")
    return record, marc_path, txt_path, marcxml_path

# --- Q&A Bot Helper Functions (New) ---
def parse_record_for_rag(record, file_path):
    """Extracts key information from a MARC record for RAG."""
    title = record.title or "[No Title]"
    authors = [f.value() for f in record.get_fields('100', '700')]
    author_str = ", ".join(authors) if authors else "[No Author]"
    abstract = record['520'].value() if record['520'] else ""
    control_number = record['001'].value() if record['001'] else "[No ID]"
    
    # Document text for embedding
    document_text = f"Title: {title}\nAuthors: {author_str}\nAbstract: {abstract}"
    
    # Metadata for display and referencing the source file
    metadata = {
        'title': title,
        'authors': authors if authors else ["N/A"],
        'control_number': control_number,
        'file_path': file_path.replace(XML_DB_DIR, output_dir).replace('.xml', '.txt')
    }
    return document_text, metadata

@st.cache_resource
def get_vector_database():
    """Loads or creates the FAISS vector database from XML files."""
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOCUMENTS_PATH) and os.path.exists(METADATA_PATH):
        print("[+] Loading existing vector database from disk...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(DOCUMENTS_PATH, 'rb') as f:
            documents = pickle.load(f)
        with open(METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
        return index, documents, metadata

    print("[*] No vector database found. Creating a new one...")
    xml_files = [f for f in os.listdir(XML_DB_DIR) if f.endswith('.xml')]
    if not xml_files:
        st.warning("No XML records found in the database to build a Q&A index.")
        embedding_dim = EMBEDDING_MODEL.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(embedding_dim)
        return index, [], []

    documents = []
    metadata = []
    for xml_file in xml_files:
        xml_path = os.path.join(XML_DB_DIR, xml_file)
        try:
            records = parse_xml_to_array(xml_path)
            if records:
                doc_text, meta = parse_record_for_rag(records[0], xml_path)
                documents.append(doc_text)
                metadata.append(meta)
        except Exception as e:
            print(f"[!] Error parsing {xml_file}: {e}")

    if not documents:
         st.error("Could not extract any documents from the XML files.")
         embedding_dim = EMBEDDING_MODEL.get_sentence_embedding_dimension()
         index = faiss.IndexFlatL2(embedding_dim)
         return index, [], []

    print(f"[*] Encoding {len(documents)} documents for the vector database...")
    embeddings = EMBEDDING_MODEL.encode(documents, convert_to_tensor=False)
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings).astype('float32'))

    print(f"[+] Saving new vector database with {index.ntotal} vectors...")
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(DOCUMENTS_PATH, 'wb') as f:
        pickle.dump(documents, f)
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)

    return index, documents, metadata


# ─────────────────────────────────────────────
#           STREAMLIT UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="Marc System", layout="centered")
st.title("📚 Marc ")

# --- Updated Tabs ---
tab_titles = [
    "📋 إدخال MARC يدويًا", 
    "📄 تحويل PDF إلى MARC", 
    "🖼️ استخراج بيانات من marc file ", 
    "🔍 السجلات المحفوظة", 
    "🤖 مساعد الأسئلة والأجوبة"
]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_titles)

# ─────────────────────────────────────────────
# 📋 Tab 1: Manual MARC Entry
with tab1:
    st.subheader("✍️ إدخال البيانات يدويًا (كتب أو رسائل جامعية)")

    # Language selection for 008 field
    lang_options = {
        "العربية": "ara",
        "الإنجليزية": "eng",
        "الفرنسية": "fre",
        "الإسبانية": "spa",
        "الألمانية": "ger"
    }
    selected_lang = st.selectbox("008 - لغة المادة", list(lang_options.keys()), index=0)
    lang_code = lang_options[selected_lang]

    st.markdown("**📌 نوع المصدر**")
    source_type = st.radio("اختر نوع المصدر", ["📚 كتاب", "🎓 رسالة جامعية"])

    # الحقول العامة
    col1, col2 = st.columns(2)
    with col1:
        control_001 = st.text_input("001 - رقم التسجيلة أو الضبط (يُولد تلقائيًا إذا تُرك فارغًا)")
        control_003 = st.text_input("003 - معرف النظام (اسم البرنامج مثل MARC-AI)", "MARC-AI")
    with col2:
        pub_year = st.text_input("008 - سنة النشر", "2024")
        form_of_item = st.selectbox("008 - شكل المادة", ["طباعة", "إلكتروني", "ميكروفيلم"], index=0)

    if source_type == "📚 كتاب":
        field_020 = st.text_input("020 - الترقيم الدولي الموحد للكتاب (ISBN)")
        field_250 = st.text_input("250 - بيانات الطبعة")
    else:
        field_502 = st.text_area("502 - بيانات الرسالة العلمية")
        field_502_uni = st.text_input("502$b - الجامعة المانحة")
        field_502_year = st.text_input("502$c - سنة المنح")

    st.markdown("### 🧾 الوصف الببليوغرافي")
    field_040a = st.text_input("040$a - جهة الفهرسة", "المكتبة الوطنية")
    field_100 = st.text_input("100 - اسم المؤلف (المدخل الرئيسي)", placeholder="الاسم الأخير، الاسم الأول")
    
    col245a, col245b = st.columns([3, 1])
    with col245a:
        field_245a = st.text_input("245$a - العنوان الرئيسي", placeholder="العنوان الرئيسي للمصدر")
    with col245b:
        field_245b = st.text_input("245$b - العنوان الفرعي", placeholder="عنوان فرعي إن وجد")
    
    field_245c = st.text_input("245$c - بيان المسؤولية", placeholder="المؤلف أو المحرر")
    
    col264a, col264b, col264c = st.columns(3)
    with col264a:
        field_264a = st.text_input("264$a - مكان النشر", placeholder="المدينة، البلد")
    with col264b:
        field_264b = st.text_input("264$b - الناشر / المؤسسة", placeholder="اسم الناشر")
    with col264c:
        field_264c = st.text_input("264$c - سنة النشر / الانتاج", placeholder="سنة النشر")
    
    col300a, col300b = st.columns(2)
    with col300a:
        field_300a = st.text_input("300$a - عدد الصفحات/الأجزاء", placeholder="مثال: 320 صفحة")
    with col300b:
        field_300b = st.text_input("300$b - الرسوم/الملاحق", placeholder="مثال: رسوم إيضاحية")

    st.markdown("### 📦 المحتوى الإضافي")
    field_504 = st.text_area("504 - تبصرة ببليوجرافية", placeholder="المراجع الببليوجرافية")
    field_520 = st.text_area("520 - الملخص / المستخلص", placeholder="ملخص محتوى المصدر")
    
    if source_type == "🎓 رسالة جامعية":
        field_546 = st.text_input("546 - تبصرة اللغة", placeholder="مثال: النص بالعربية والإنجليزية")

    st.markdown("### 🏷 الفهرسة الموضوعية")
    subjects = st.text_area("650 - رؤوس الموضوعات (افصل بينها بفاصلة)", placeholder="اقتصاد, تعليم, تكنولوجيا")
    
    st.markdown("### 👥 الأسماء الإضافية")
    additional_700 = st.text_area("700 - أسماء أشخاص إضافيين (افصل بينها بفاصلة)", placeholder="مشرف, محرر, مشارك")
    
    st.markdown("### ➕ حقول MARC مخصصة (اختياري)")
    with st.expander("إضافة حقول غير موجودة في النموذج أعلاه"):
        # Initialize session state
        if "custom_fields" not in st.session_state:
            st.session_state.custom_fields = []
        
        field_type = st.radio("نوع الحقل", ["حقل تحكم (001-009)", "حقل بيانات (010-999)"], key="field_type")
        
        tag = st.text_input("وسم الحقل (ثلاثة أرقام)", placeholder="245", key="custom_tag")
        if tag and (len(tag) != 3 or not tag.isdigit()):
            st.error("يجب أن يتكون وسم الحقل من 3 أرقام")
        
        # Control field (001-009)
        if field_type == "حقل تحكم (001-009)":
            data = st.text_area("بيانات الحقل", placeholder="أدخل البيانات هنا")
        
        # Data field (010-999)
        else:
            col1, col2 = st.columns(2)
            with col1:
                ind1 = st.text_input("المؤشر الأول", max_chars=1, value=" ", placeholder="0-9 أو #")
            with col2:
                ind2 = st.text_input("المؤشر الثاني", max_chars=1, value=" ", placeholder="0-9 أو #")
            
            st.markdown("**الحقول الفرعية**")
            subfields = []
            num_subfields = st.number_input("عدد الحقول الفرعية", 1, 10, 1)
            
            for i in range(num_subfields):
                cols = st.columns([1, 5])
                with cols[0]:
                    code = st.text_input(f"رمز الحقل الفرعي {i+1}", max_chars=1, placeholder="a-z أو 0-9")
                    if code and (len(code) != 1 or not code.isalnum()):
                        st.error("يجب أن يكون رمز الحقل الفرعي حرفًا أو رقمًا واحدًا")
                with cols[1]:
                    value = st.text_input(f"قيمة الحقل الفرعي {i+1}", placeholder="النص")
                subfields.append((code, value))
        
        if st.button("➕ أضف هذا الحقل", key="add_custom_field"):
            if not tag or len(tag) != 3 or not tag.isdigit():
                st.error("وسم الحقل غير صالح")
            else:
                new_field = {
                    "tag": tag,
                    "type": "control" if field_type == "حقل تحكم (001-009)" else "data"
                }
                
                if new_field["type"] == "control":
                    if not data.strip():
                        st.error("بيانات الحقل مطلوبة")
                    else:
                        new_field["data"] = data
                        st.session_state.custom_fields.append(new_field)
                        st.success("تمت إضافة الحقل!")
                else:
                    # Validate subfields
                    valid_subfields = True
                    for code, value in subfields:
                        if not code or not value or len(code) != 1 or not code.isalnum():
                            st.error(f"رمز أو قيمة الحقل الفرعي غير صالحة: ${code} {value}")
                            valid_subfields = False
                            break
                    
                    if valid_subfields:
                        new_field["ind1"] = ind1 if ind1.strip() else " "
                        new_field["ind2"] = ind2 if ind2.strip() else " "
                        new_field["subfields"] = [(c, v) for c, v in subfields if c and v]
                        st.session_state.custom_fields.append(new_field)
                        st.success("تمت إضافة الحقل!")
        
        st.markdown("**الحقول المضافة:**")
        for i, field in enumerate(st.session_state.custom_fields):
            st.write(f"{field['tag']}: ", end="")
            if field["type"] == "control":
                st.write(field["data"])
            else:
                subfields_str = " ".join([f"${c} {v}" for c, v in field["subfields"]])
                st.write(f"{field['ind1']}{field['ind2']} {subfields_str}")
            
            if st.button(f"حذف {i+1}", key=f"del_{i}"):
                del st.session_state.custom_fields[i]
                st.rerun()

    # زر إنشاء السجل
    if st.button("💾 حفظ وتوليد الملف"):
        # التحقق من الحقول الإلزامية
        required_fields = {
            "100": field_100,
            "245$a": field_245a,
            "264$b": field_264b,
            "264$c": field_264c
        }
        
        missing_fields = [name for name, value in required_fields.items() if not value.strip()]
        
        if missing_fields:
            st.error(f"❌ الحقول التالية مطلوبة: {', '.join(missing_fields)}")
            st.stop()
        
        record = Record()
        record.leader = '00000nam a2200000 u 4500'

        # توليد التاريخ والوقت
        current_time_005 = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S.0')
        
        # بناء حقل 008 ديناميكيًا
        form_code = {
            "طباعة": "s",
            "إلكتروني": "s",
            "ميكروفيلم": "m"
        }.get(form_of_item, "s")
        
        fixed_field_008 = (
            datetime.datetime.today().strftime("%y%m%d") + 
            f"{form_code}{pub_year}####xx###########{lang_code}#d"
        )

        # توليد 001 تلقائي لو فاضي
        control_001 = control_001.strip() or generate_control_number()

        # الحقول الأساسية
        record.add_field(Field(tag='001', data=control_001))
        record.add_field(Field(tag='003', data=control_003))
        record.add_field(Field(tag='005', data=current_time_005))
        record.add_field(Field(tag='008', data=fixed_field_008))

        if source_type == "📚 كتاب" and field_020:
            record.add_field(Field(tag='020', indicators=['#', '#'], subfields=sf('a', field_020)))

        record.add_field(Field(tag='040', indicators=['#', '#'], subfields=sf('a', field_040a)))
        record.add_field(Field(tag='100', indicators=['1', '#'], subfields=sf('a', field_100)))
        
        # حقل العنوان 245
        subfields_245 = sf('a', field_245a)
        if field_245b:
            subfields_245.extend(sf('b', field_245b))
        if field_245c:
            subfields_245.extend(sf('c', field_245c))
        record.add_field(Field(tag='245', indicators=['1', '0'], subfields=subfields_245))
        
        if source_type == "📚 كتاب" and field_250:
            record.add_field(Field(tag='250', indicators=['#', '#'], subfields=sf('a', field_250)))
        
        # حقل النشر 264
        subfields_264 = []
        if field_264a:
            subfields_264.extend(sf('a', field_264a))
        subfields_264.extend(sf('b', field_264b, 'c', field_264c))
        record.add_field(Field(tag='264', indicators=['#', '1'], subfields=subfields_264))
        
        # حقل الوصف المادي 300
        subfields_300 = sf('a', field_300a)
        if field_300b:
            subfields_300.extend(sf('b', field_300b))
        record.add_field(Field(tag='300', indicators=['#', '#'], subfields=subfields_300))
        
        record.add_field(Field(tag='336', indicators=['#', '#'], subfields=sf('a', 'text', 'b', 'txt', '2', 'rdacontent')))
        record.add_field(Field(tag='337', indicators=['#', '#'], subfields=sf('a', 'unmediated', 'b', 'n', '2', 'rdamedia')))
        record.add_field(Field(tag='338', indicators=['#', '#'], subfields=sf('a', 'volume', 'b', 'nc', '2', 'rdacarrier')))

        if field_504:
            record.add_field(Field(tag='504', indicators=['#', '#'], subfields=sf('a', field_504)))
        if field_520:
            record.add_field(Field(tag='520', indicators=['#', '#'], subfields=sf('a', field_520)))
        
        if source_type == "🎓 رسالة جامعية":
            if field_502:
                subfields_502 = sf('a', field_502)
                if field_502_uni:
                    subfields_502.extend(sf('b', field_502_uni))
                if field_502_year:
                    subfields_502.extend(sf('c', field_502_year))
                record.add_field(Field(tag='502', indicators=['#', '#'], subfields=subfields_502))
            if field_546:
                record.add_field(Field(tag='546', indicators=['#', '#'], subfields=sf('a', field_546)))
        
        # رؤوس الموضوعات (متعددة)
        if subjects:
            for subject in [s.strip() for s in subjects.split(',') if s.strip()]:
                record.add_field(Field(tag='650', indicators=['#', '0'], subfields=sf('a', subject)))
        
        # الأسماء الإضافية (متعددة)
        if additional_700:
            for person in [p.strip() for p in additional_700.split(',') if p.strip()]:
                record.add_field(Field(tag='700', indicators=['1', '#'], subfields=sf('a', person)))

        # Add custom fields to record
        for field in st.session_state.custom_fields:
            if field["type"] == "control":
                record.add_field(Field(tag=field["tag"], data=field["data"]))
            else:
                subfield_objs = []
                for code, value in field["subfields"]:
                    subfield_objs.append(Subfield(code=code, value=value))
                
                record.add_field(Field(
                    tag=field["tag"],
                    indicators=[field["ind1"], field["ind2"]],
                    subfields=subfield_objs
                ))
        
        # SORT FIELDS IN PROPER MARC ORDER
        def get_field_order(field):
            """Get sorting key for fields (control fields first, then by numeric tag)"""
            if field.tag in ['001', '003', '005', '008']:
                return (0, int(field.tag))
            return (1, int(field.tag))
        
        record.fields = sorted(record.fields, key=get_field_order)

        # Generate unique filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{field_245a}"
        
        # Define file paths
        marc_path = os.path.join(output_dir, f"{base_filename}.mrc")
        txt_path = os.path.join(output_dir, f"{base_filename}.txt")
        xml_path = os.path.join(output_dir, f"{base_filename}.xml")
        xml_db_path = os.path.join(XML_DB_DIR, f"{base_filename}.xml")

        # Save files to output directory
        try:
            # Save MARC binary
            with open(marc_path, 'wb') as f:
                f.write(record.as_marc())
            
            # Save MARC text
            with open(txt_path, 'w', encoding='utf-8') as f:
                marc_text = record.as_marc21()
                if isinstance(marc_text, bytes):
                    marc_text = marc_text.decode('utf-8')
                f.write(marc_text)
        
            # Save MARC XML
            with open(xml_path, 'wb') as f:
                writer = XMLWriter(f)
                writer.write(record)
                writer.close()
            
            # Copy XML to database directory
            shutil.copy2(xml_path, xml_db_path)
            # Clear vector database cache to include new record
            if os.path.exists(FAISS_INDEX_PATH): os.remove(FAISS_INDEX_PATH)
            if os.path.exists(DOCUMENTS_PATH): os.remove(DOCUMENTS_PATH)
            if os.path.exists(METADATA_PATH): os.remove(METADATA_PATH)
            st.cache_resource.clear()

            st.success(f"✅ تم إنشاء السجل بنجاح وحفظ الملفات في مجلد output!")
            st.success(f"✅ تم حفظ نسخة XML في قاعدة البيانات: {xml_db_path}")

            # عرض التسجيلة كاملة
            st.subheader("معاينة التسجيلة")
            st.code("\n".join(str(field) for field in record))

            st.session_state.generated_files = {
            "mrc": (marc_path, f"{base_filename}.mrc", "application/marc"),
            "txt": (txt_path, f"{base_filename}.txt", "text/plain"), 
            "xml": (xml_path, f"{base_filename}.xml", "text/xml")
            }

            # Clear custom fields after successful creation
            st.session_state.custom_fields = []

        except Exception as e:
            st.error(f"❌ حدث خطأ أثناء حفظ الملفات: {str(e)}")


    if "generated_files" in st.session_state:
        st.subheader("تحميل الملفات")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.download_button(
                "📥 تحميل MARC (.mrc)",
                data=open(st.session_state.generated_files["mrc"][0], "rb").read(),
                file_name=st.session_state.generated_files["mrc"][1],
                mime=st.session_state.generated_files["mrc"][2],
                key="marc_download"
            ):
                st.toast("تم بدء تحميل ملف MARC")
        
        with col2:
            if st.download_button(
                "📥 تحميل نصي (.txt)",
                data=open(st.session_state.generated_files["txt"][0], "r", encoding="utf-8").read(),
                file_name=st.session_state.generated_files["txt"][1],
                mime=st.session_state.generated_files["txt"][2],
                key="txt_download"
            ):
                st.toast("تم بدء تحميل الملف النصي")
        
        with col3:
            if st.download_button(
                "📥 تحميل XML (.xml)",
                data=open(st.session_state.generated_files["xml"][0], "r", encoding="utf-8").read(),
                file_name=st.session_state.generated_files["xml"][1],
                mime=st.session_state.generated_files["xml"][2],
                key="xml_download"
            ):
                st.toast("تم بدء تحميل ملف XML")

# Cached file readers for Tab4
@lru_cache(maxsize=32)
def read_text_file_cached(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@lru_cache(maxsize=32)
def read_marc_file_cached(file_path):
    try:
        with open(file_path, 'rb') as f:
            reader = MARCReader(f)
            return next(reader)
    except Exception as e:
        return f"Error reading MARC file: {str(e)}"

# ─────────────────────────────────────────────
# 📄 Tab 2: PDF to MARC
with tab2:
    st.subheader("📄 PDF to Marc File")
    st.write("Upload a PDF to generate MARC records")
    api_key = st.secrets["groq_api_key"]
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    current_file_hash = None
    if uploaded_file:
        current_file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
    should_process = (uploaded_file and api_key and (current_file_hash != st.session_state.get('processed_file_hash') or st.session_state.get('processed_file_hash') is None))
    if should_process:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        base_filename = Path(uploaded_file.name).stem
        temp_pdf = os.path.join(temp_dir, f"{base_filename}.pdf")
        marc_bin_path = os.path.join(output_dir, f"{base_filename}.mrc")
        marc_txt_path = os.path.join(output_dir, f"{base_filename}.txt")
        marc_xml_path = os.path.join(output_dir, f"{base_filename}.xml")
        xml_db_path = os.path.join(XML_DB_DIR, f"{base_filename}.xml")
        with st.spinner("Processing PDF..."):
            try:
                with open(temp_pdf, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.info("Sending to GROBID...")
                # Create a reduced PDF with only the first 30 pages
                short_pdf_path = os.path.join(temp_dir, f"{base_filename}_first30.pdf")
                doc = fitz.open(temp_pdf)
                short_doc = fitz.open()  # empty PDF
                for i in range(min(40, len(doc))):
                    short_doc.insert_pdf(doc, from_page=i, to_page=i)
                short_doc.save(short_pdf_path)
                short_doc.close()
                doc.close()

                # Send the short version to GROBID
                tei = send_pdf_to_grobid_header(short_pdf_path)

                st.info("Extracting text snippet...")
                text = extract_text_from_pdf(temp_pdf)
                st.info("Extracting metadata with AI...")
                prompt = f"""Extract the following metadata from this academic text and return ONLY valid JSON (no explanation):\n- title\n- authors (full names)\n\nText:\n{text}"""
                llm_response = ask_groq_model(prompt, api_key)
                llm_metadata = extract_json_block(llm_response)
                st.info("Generating MARC records...")
                record, marc_bin_path, marc_txt_path, marc_xml_path = tei_to_marc(tei, marc_bin_path, temp_pdf, llm_metadata)
                if os.path.exists(marc_xml_path):
                    shutil.copy2(marc_xml_path, xml_db_path)
                    # Clear vector database cache to include new record
                    if os.path.exists(FAISS_INDEX_PATH): os.remove(FAISS_INDEX_PATH)
                    if os.path.exists(DOCUMENTS_PATH): os.remove(DOCUMENTS_PATH)
                    if os.path.exists(METADATA_PATH): os.remove(METADATA_PATH)
                    st.cache_resource.clear()

                st.session_state.current_record = record
                st.session_state.llm_metadata = llm_metadata
                st.session_state.marc_bin_path = marc_bin_path
                st.session_state.marc_txt_path = marc_txt_path
                st.session_state.marc_xml_path = marc_xml_path
                st.session_state.processed_file_hash = current_file_hash
                st.session_state.base_filename = base_filename
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
            finally:
                if os.path.exists(temp_pdf):
                    try:
                        os.remove(temp_pdf)
                    except:
                        pass
    if st.session_state.get('current_record'):
        st.success(f"MARC records generated successfully! Saved in {output_dir}/")
        st.subheader("Extracted Metadata")
        st.json(st.session_state.llm_metadata)
        st.subheader("Text MARC Preview")
        try:
            with open(st.session_state.marc_txt_path, "r", encoding="utf-8") as f:
                st.code(f.read())
        except:
            st.code(str(st.session_state.current_record))
        st.subheader("Download MARC Records")
        col1, col2, col3 = st.columns(3)
        with col1:
            with open(st.session_state.marc_bin_path, "rb") as f:
                st.download_button("Download MARC (.mrc)", f, file_name=f"{st.session_state.base_filename}.mrc")
        with col2:
            with open(st.session_state.marc_txt_path, "r", encoding="utf-8") as f:
                st.download_button("Download Text MARC (.txt)", f, file_name=f"{st.session_state.base_filename}.txt")
        with col3:
            with open(st.session_state.marc_xml_path, "r", encoding="utf-8") as f:
                st.download_button("Download MARC XML (.xml)", f, file_name=f"{st.session_state.base_filename}.xml")

# ─────────────────────────────────────────────
# 🖼️ Tab 3: Read MARC File
with tab3:
    st.subheader("📂 قراءة MARC موجود (.mrc)")
    uploaded_marc = st.file_uploader("⬆️ اختر ملف MARC (.mrc)", type=["mrc"])
    if uploaded_marc is not None:
        try:
            st.success("✅ تم رفع الملف بنجاح. جاري المعالجة...")
            with io.BytesIO(uploaded_marc.read()) as buffer:
                reader = MARCReader(buffer, to_unicode=True, force_utf8=True)
                for i, record in enumerate(reader):
                    with st.expander(f"📄 السجل رقم {i+1}"):
                        st.code(str(record), language="text")    
                        # Save MARC record to XML_DB_DIR for Q&A Bot
                        xml_filename = Path(uploaded_marc.name).stem + ".xml"
                        xml_path = os.path.join(XML_DB_DIR, xml_filename)
                        with open(xml_path, "wb") as f:
                            writer = XMLWriter(f)
                            writer.write(record)
                            writer.close()

                        # Also save MARC and TXT to output folder so Tab 4 can display it
                        base_filename = Path(uploaded_marc.name).stem
                        marc_bin_path = os.path.join(output_dir, f"{base_filename}.mrc")
                        marc_txt_path = os.path.join(output_dir, f"{base_filename}.txt")
                        with open(marc_bin_path, 'wb') as f:
                            f.write(record.as_marc())
                        with open(marc_txt_path, 'w', encoding='utf-8') as f:
                            marc_text = record.as_marc21()
                            f.write(marc_text.decode('utf-8') if isinstance(marc_text, bytes) else marc_text)

                        # Clear vector database cache to include new record
                        if os.path.exists(FAISS_INDEX_PATH): os.remove(FAISS_INDEX_PATH)
                        if os.path.exists(DOCUMENTS_PATH): os.remove(DOCUMENTS_PATH)
                        if os.path.exists(METADATA_PATH): os.remove(METADATA_PATH)
                        st.cache_resource.clear()
                        st.success(f"Record saved to database for Q&A and added to library: {base_filename}")


        except Exception as e:
            st.error("❌ حدث خطأ أثناء قراءة الملف:")
            st.exception(e)

# ─────────────────────────────────────────────
# 🔍 Tab 4: Saved Records
with tab4:
    st.subheader("📚 Library Records")
    try:
        all_files = os.listdir(output_dir)
        txt_files = [f for f in all_files if f.endswith('.txt')]
        record_pairs = []
        for txt_file in txt_files:
            base_name = os.path.splitext(txt_file)[0]
            mrc_file = f"{base_name}.mrc"
            if mrc_file in all_files:
                if not any(r[2] == base_name for r in record_pairs):
                    record_pairs.append((txt_file, mrc_file, base_name))
        if not record_pairs:
            st.info("No complete record pairs found in the output folder.")
        else:
            selected_pair = st.selectbox("Select a record to view:", record_pairs, format_func=lambda x: x[0].replace('.txt', ''), index=0)
            if selected_pair:
                txt_file, mrc_file, base_name = selected_pair
                txt_path = os.path.join(output_dir, txt_file)
                mrc_path = os.path.join(output_dir, mrc_file)
                xml_path = os.path.join(output_dir, f"{base_name}.xml")
                xml_db_path = os.path.join(XML_DB_DIR, f"{base_name}.xml")
                file_stats = os.stat(txt_path)
                st.caption(f"Last modified: {datetime.datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
                st.caption(f"File size: {file_stats.st_size/1024:.2f} KB")
                
                try:
                    record = read_marc_file_cached(mrc_path)
                    if isinstance(record, str):
                        st.error(record)
                    else:
                        st.code("\n".join(str(field) for field in record), language='text')
                        st.subheader("Formatted MARC")
                        for field in record:
                            if field.is_control_field():
                                st.text(f"{field.tag}: {field.data}")
                            else:
                                subfields = " ".join(f"${sf.code} {sf.value}" for sf in field.subfields)
                                st.text(f"{field.tag} {field.indicator1}{field.indicator2}: {subfields}")
                except Exception as e:
                    st.error(f"Error reading MARC file: {str(e)}")
                st.subheader("Download")
                with open(mrc_path, 'rb') as f:
                    st.download_button("Download MARC Record", f, file_name=mrc_file, mime="application/marc")
                st.markdown("---")
                with st.form(key=f"delete_form_{base_name}"):
                    if st.form_submit_button("🗑️ Delete This Record"):
                        try:
                            deleted_files = []
                            for ext in ['.txt', '.mrc', '.xml']:
                                file_path = os.path.join(output_dir, f"{base_name}{ext}")
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                                    deleted_files.append(file_path)
                            if os.path.exists(xml_db_path):
                                os.remove(xml_db_path)
                                deleted_files.append(xml_db_path)
                            if deleted_files:
                                st.success(f"Successfully deleted {len(deleted_files)} files for {base_name}")
                                # Clear caches related to the vector DB so it rebuilds
                                if os.path.exists(FAISS_INDEX_PATH): os.remove(FAISS_INDEX_PATH)
                                if os.path.exists(DOCUMENTS_PATH): os.remove(DOCUMENTS_PATH)
                                if os.path.exists(METADATA_PATH): os.remove(METADATA_PATH)
                                st.rerun()
                            else:
                                st.warning("No files were found to delete")
                        except Exception as e:
                            st.error(f"Error deleting files: {str(e)}")
                            st.info("Deletion status:")
                            for path in [txt_path, mrc_path, xml_path, xml_db_path]:
                                exists = os.path.exists(path)
                                st.write(f"{'❌' if exists else '✅'} {path}")
    except Exception as e:
        st.error(f"Error accessing records: {str(e)}")


# ─────────────────────────────────────────────
# 🤖 Tab 5: Q&A Bot (New)
# ─────────────────────────────────────────────
with tab5:
    st.subheader("🤖 Library Q&A Assistant")
    st.write("Ask natural language questions about books in our library. The database will be built on the first query.")
    
    # Groq API Key
    api_key = st.secrets["groq_api_key"]
    
    user_question = st.text_area("Ask a question:", 
                                 placeholder="Who is the author of 'Deep Learning for Computer Vision'?",
                                 height=100)
    
    if st.button("Ask Question") and api_key and user_question:
        with st.spinner("Searching library and generating answer..."):
            try:
                # Optional manual rebuild
                if st.button("Rebuild Database"):
                    if os.path.exists(FAISS_INDEX_PATH): os.remove(FAISS_INDEX_PATH)
                    if os.path.exists(DOCUMENTS_PATH): os.remove(DOCUMENTS_PATH)
                    if os.path.exists(METADATA_PATH): os.remove(METADATA_PATH)
                    st.cache_resource.clear()
                    st.success("Database cache cleared. It will rebuild on next query.")

                # Get vector database
                index, documents, metadata = get_vector_database()

                if index.ntotal == 0:
                    st.warning("The library database is empty. Please add records first using the other tabs.")
                    st.stop()
                
                # Embed user question
                query_embedding = EMBEDDING_MODEL.encode([user_question])
                
                # Find relevant documents (k=3)
                D, I = index.search(np.array(query_embedding).astype('float32'), 3)
                
                # Prepare context and metadata for the sources found
                relevant_indices = [i for i in I[0] if i < len(documents)] # Ensure indices are valid
                context = "\n\n".join([documents[i] for i in relevant_indices])
                source_metadata = [metadata[i] for i in relevant_indices]
                
                if not context:
                     st.warning("Could not find any relevant documents for your query.")
                     st.stop()
                
                # Prepare prompt for the LLM
                prompt = f"""
                You are a librarian assistant answering questions about books in our library and know that you database contains marc files records.
                Answer ONLY based on the following context extracted from library records.
                If you don't know the answer from the context, say "I couldn't find that information in our library."
                Be concise and accurate.
                
                Context:
                {context}
                
                Question: {user_question}
                Answer:
                """
                
                # Get answer from Groq
                answer = ask_groq_model(prompt, api_key)
                
                # Display results
                st.subheader("Answer:")
                st.write(answer)
                
                # Show sources used
                st.subheader("Sources used:")
                for i, meta in enumerate(source_metadata):
                    with st.expander(f"Source {i+1}: {meta['title']}"):
                        st.write(f"**Authors:** {', '.join(meta['authors'])}")
                        st.write(f"**Control Number:** {meta['control_number']}")
                        
                        # Show preview of the source text file
                        if os.path.exists(meta['file_path']):
                            with open(meta['file_path'], 'r', encoding='utf-8') as f:
                                preview = f.read(5000)
                                st.caption(f"Content preview: {preview}...")
                        else:
                            st.caption(f"Preview file not found at: {meta['file_path']}")
                
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

