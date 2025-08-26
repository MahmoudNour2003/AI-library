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
    """
    Extracts ALL fields from a MARC record for RAG embeddings.
    This ensures Q&A bot can answer about pages, publisher, etc.
    """
    field_texts = []

    for field in record:
        if field.is_control_field():
            field_texts.append(f"{field.tag}: {field.data}")
        else:
            subfields = " ".join([f"${sf.code} {sf.value}" for sf in field.subfields])
            field_texts.append(f"{field.tag} {field.indicator1}{field.indicator2}: {subfields}")

    # Combine into a single text block
    document_text = "\n".join(field_texts)

    # Metadata for display in sources list
    title = record.title or "[No Title]"
    authors = [f.value() for f in record.get_fields('100', '700')]
    control_number = record['001'].value() if record['001'] else "[No ID]"

    metadata = {
        "title": title,
        "authors": authors if authors else ["N/A"],
        "control_number": control_number,
        "file_path": file_path.replace(XML_DB_DIR, output_dir).replace('.xml', '.txt')
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

def extract_text_from_image(image_path, api_key):
    """
    Extracts text from an image using OCR.Space API.
    """
    url = "https://api.ocr.space/parse/image"
    with open(image_path, 'rb') as f:
        r = requests.post(
            url,
            files={"file": f},
            data={"apikey": api_key, "language": "eng"},
            timeout=60
        )
    result = r.json()
    if result.get("IsErroredOnProcessing"):
        raise Exception(result.get("ErrorMessage", "OCR failed"))
    return result["ParsedResults"][0]["ParsedText"]
def llm_metadata_to_marc(llm_metadata, output_base_path):
    """
    Builds a MARC record directly from LLM-extracted metadata (JSON).
    Does NOT require TEI parsing.
    """
    record = Record(force_utf8=True)
    record.leader = "     nam a22     uu 4500"

    # Control fields
    record.add_field(Field(tag='001', data=generate_control_number()))
    record.add_field(Field(tag='003', data='LLMExtractor'))
    record.add_field(Field(tag='005', data=datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S.0')))

    # Publication year
    pub_year = llm_metadata.get("publication year", "2024")

    # Language
    lang_code = llm_metadata.get("language", "en")
    lang_code = lang_code[:3].lower().ljust(3, '#')
    record.add_field(Field(tag='008', data=generate_008_field(pub_year, lang_code)))
    record.add_field(Field(tag='041', indicators=['0', '#'], subfields=[Subfield('a', lang_code)]))

    # Title and subtitle
    title = llm_metadata.get("title", "[Title not available]")
    subtitle = llm_metadata.get("subtitle", "")
    main_title, subtitle_split = (title.split(':', 1) + [''])[:2]
    subtitle = subtitle or subtitle_split

    # Authors
    authors = llm_metadata.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]

    for i, author in enumerate(authors):
        tag = '100' if i == 0 else '700'
        record.add_field(Field(tag=tag, indicators=['1', '#'], subfields=[
            Subfield('a', author),
            Subfield('e', 'author.')
        ]))

    record.add_field(Field(tag='245', indicators=['1', '0'], subfields=[
        Subfield('a', main_title.strip() + (' :' if subtitle else '')),
        Subfield('b', subtitle.strip()),
        Subfield('c', format_authors(authors) + '.')
    ]))

    # Authors fields
    for i, author in enumerate(authors):
        tag = '100' if i == 0 else '700'
        record.add_field(Field(tag=tag, indicators=['1', '#'], subfields=[
            Subfield('a', author),
            Subfield('e', 'author.')
        ]))

    # Abstract
    abstract = llm_metadata.get("abstract", "")
    if abstract:
        record.add_field(Field(tag='520', indicators=['#', '#'], subfields=[Subfield('a', abstract)]))

    #keywords
    keywords = llm_metadata.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    for keyword in keywords:
        if keyword.strip():
            record.add_field(Field(tag='650', indicators=['#', '0'], subfields=[Subfield('a', keyword.strip())]))
    # --- Subjects ---
    subjects = llm_metadata.get("subjects") or []
    if isinstance(subjects, str):
        subjects = [subjects]

    for subject in subjects:
        if subject.strip():
            record.add_field(Field(
                tag='650',
                indicators=['#', '0'],
                subfields=[Subfield('a', subject.strip())]
            ))

    # Publisher & Year
    publisher = llm_metadata.get("publisher", "[Unknown Publisher]")
    record.add_field(Field(tag='264', indicators=['#', '1'], subfields=[
        Subfield('a', '[Place of publication not identified]'),
        Subfield('b', publisher),
        Subfield('c', pub_year)
    ]))

    # Identifiers
    doi = llm_metadata.get("doi")
    if doi:
        record.add_field(Field(tag='024', indicators=['7', '#'], subfields=[
            Subfield('a', doi),
            Subfield('2', 'doi')
        ]))
    isbn = llm_metadata.get("isbn")
    if isbn:
        record.add_field(Field(tag='020', indicators=['#', '#'], subfields=[Subfield('a', isbn)]))

    # Journal info
    journal_title = llm_metadata.get("journal title")
    volume = llm_metadata.get("volume")
    issue = llm_metadata.get("issue")
    pages = llm_metadata.get("pages")

    if journal_title:
        subfields = [Subfield('t', journal_title)]
        if volume: subfields.append(Subfield('g', f"Vol. {volume}"))
        if issue: subfields.append(Subfield('g', f"No. {issue}"))
        if pages: subfields.append(Subfield('g', f"pp. {pages}"))
        record.add_field(Field(tag='773', indicators=['0', ' '], subfields=subfields))
    record.fields = sorted(record.fields, key=lambda f: int(f.tag) if f.tag.isdigit() else float('inf'))
    # Save to files
    marc_path = output_base_path + '.mrc'
    txt_path = output_base_path + '.txt'
    marcxml_path = output_base_path + '.xml'

    with open(marc_path, 'wb') as f:
        f.write(record.as_marc())
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(str(record))
    with open(marcxml_path, 'wb') as f:
        writer = XMLWriter(f)
        writer.write(record)
        writer.close()

    return record, marc_path, txt_path, marcxml_path

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
    st.subheader("✍ إدخال بيانات MARC يدويًا - حقول التحكم")
    
    # Initialize session state for control fields
    if "control_fields" not in st.session_state:
        st.session_state.control_fields = {
            "000": "00000nam a2200000 u 4500",
            "001": "",
            "003": "MARC-AI",
            "005": datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S.0'),
            "008": {
                "entry_date": datetime.datetime.today().strftime("%y%m%d"),
                "pub_status": "s",
                "date1": "2024",
                "date2": "####",
                "place": "xx#",
                "illustrations": "####",
                "target_audience": "#",
                "form_of_item": "s",
                "nature_of_contents": "####",
                "govt_pub": "#",
                "conference": "0",
                "festschrift": "0",
                "index": "0",
                "literary_form": "0",
                "biography": "#",
                "language": "ara",
                "modified_record": "#",
                "cataloging_source": "d"
            }
        }

    
    # حقل 000 - الليدير
    st.session_state.control_fields["000"] = st.text_input(
        "000 - الليدير (Leader)",
        value=st.session_state.control_fields["000"],
        help="24 حرفًا تمثل وصفًا هيكليًا للتسجيلة"
    )
    
    # حقل 001 - رقم التحكم
    st.session_state.control_fields["001"] = st.text_input(
        "001 - رقم التحكم (Control Number)",
        value=st.session_state.control_fields["001"] or generate_control_number(),
        help="معرف فريد للتسجيلة في النظام"
    )
    
    # حقل 003 - معرّف النظام
    st.session_state.control_fields["003"] = st.text_input(
        "003 - معرّف النظام (System Identifier)",
        value=st.session_state.control_fields["003"],
        help="رمز أو اسم النظام الذي أنشأ رقم التحكم"
    )
    
    # حقل 005 - تاريخ ووقت آخر تعديل
    st.session_state.control_fields["005"] = st.text_input(
        "005 - تاريخ ووقت آخر تعديل (Last Modification)",
        value=st.session_state.control_fields["005"],
        help="صيغة YYYYMMDDHHMMSS.0"
    )
    

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.session_state.control_fields["008"]["entry_date"] = st.text_input(
            "008/00-05 - تاريخ الإدخال",
            value=st.session_state.control_fields["008"]["entry_date"],
            max_chars=6,
            help="YYMMDD"
        )
        
        st.session_state.control_fields["008"]["pub_status"] = st.selectbox(
            "008/06 - حالة النشر",
            options=["s", "c", "n", "d", "e", "f", "g", "k", "m", "p", "q", "r", "t", "u"],
            index=0,
            help="s: تاريخ واحد, c: تواريخ متعددة, n: تاريخ غير معروف"
        )
        
        st.session_state.control_fields["008"]["date1"] = st.text_input(
            "008/07-10 - التاريخ 1",
            value=st.session_state.control_fields["008"]["date1"],
            max_chars=4,
            help="سنة النشر الأولى"
        )
        
        st.session_state.control_fields["008"]["date2"] = st.text_input(
            "008/11-14 - التاريخ 2",
            value=st.session_state.control_fields["008"]["date2"],
            max_chars=4,
            help="سنة النشر الأخيرة أو ####"
        )
        
        st.session_state.control_fields["008"]["place"] = st.text_input(
            "008/15-17 - مكان النشر",
            value=st.session_state.control_fields["008"]["place"],
            max_chars=3,
            help="رمز مكان النشر (3 أحرف)"
        )
    
    with col2:
        st.session_state.control_fields["008"]["illustrations"] = st.text_input(
            "008/18-21 - الرسوم الإيضاحية",
            value=st.session_state.control_fields["008"]["illustrations"],
            max_chars=4,
            help="أكواد الرسوم (a, b, c, d, e, f, g, h, j, k)"
        )
        
        st.session_state.control_fields["008"]["target_audience"] = st.selectbox(
            "008/22 - الجمهور المستهدف",
            options=[" ", "a", "b", "c", "d", "e", "f", "g", "j", "k"],
            index=0,
            help="#: غير محدد, a: قبل المدرسة, b: ابتدائي, إلخ"
        )
        
        st.session_state.control_fields["008"]["form_of_item"] = st.selectbox(
            "008/23 - شكل المادة",
            options=[" ", "a", "b", "c", "d", "f", "o", "q", "r", "s"],
            index=9,  # s is the 10th item (0-indexed)
            help="#: غير محدد, s: إلكتروني, إلخ"
        )
        
        st.session_state.control_fields["008"]["nature_of_contents"] = st.text_input(
            "008/24-27 - طبيعة المحتوى",
            value=st.session_state.control_fields["008"]["nature_of_contents"],
            max_chars=4,
            help="أكواد المحتوى (a, b, c, d, e, f, m)"
        )
    
    with col3:
        st.session_state.control_fields["008"]["govt_pub"] = st.selectbox(
            "008/28 - النشر الحكومي",
            options=[" ", "a", "c", "f", "i", "l", "m", "o", "s", "u", "z"],
            index=0,
            help="#: غير حكومي, a: اتحادي/وطني, إلخ"
        )
        
        st.session_state.control_fields["008"]["conference"] = st.selectbox(
            "008/29 - مؤتمر",
            options=["0", "1"],
            index=0,
            help="0: ليس منشور مؤتمر, 1: منشور مؤتمر"
        )
        
        st.session_state.control_fields["008"]["festschrift"] = st.selectbox(
            "008/30 - إهداء",
            options=["0", "1"],
            index=0,
            help="0: ليس إهداء, 1: إهداء"
        )
        
        st.session_state.control_fields["008"]["index"] = st.selectbox(
            "008/31 - فهارس",
            options=["0", "1"],
            index=0,
            help="0: بدون فهارس, 1: به فهارس"
        )
        
        st.session_state.control_fields["008"]["literary_form"] = st.selectbox(
            "008/32 - شكل أدبي",
            options=["0", "1", "c", "d", "e", "f", "h", "i", "j", "m", "p", "s", "u"],
            index=0,
            help="0: غير خيالي, 1: خيالي, إلخ"
        )
        
        st.session_state.control_fields["008"]["biography"] = st.selectbox(
            "008/33 - سيرة ذاتية",
            options=[" ", "a", "b", "c", "d"],
            index=0,
            help="#: لا تحتوي, a: سيرة ذاتية, إلخ"
        )
        
        st.session_state.control_fields["008"]["language"] = st.selectbox(
            "008/35-37 - اللغة",
            options=["ara", "eng", "fre", "spa", "ger"],
            index=0,
            help="رمز اللغة المكون من 3 أحرف"
        )
        
        st.session_state.control_fields["008"]["modified_record"] = st.selectbox(
            "008/38 - تسجيلة معدلة",
            options=[" ", "d", "o", "s"],
            index=0,
            help="#: غير معدلة, d: محذوف, إلخ"
        )
        
        st.session_state.control_fields["008"]["cataloging_source"] = st.selectbox(
            "008/39 - مصدر الفهرسة",
            options=[" ", "a", "c", "d", "u"],
            index=4,  # d is the 5th item (0-indexed)
            help="d: مكتبة أخرى, u: غير معروف"
        )
    

    eight_field = (
        f"{st.session_state.control_fields['008']['entry_date']}"
        f"{st.session_state.control_fields['008']['pub_status']}"
        f"{st.session_state.control_fields['008']['date1']}"
        f"{st.session_state.control_fields['008']['date2']}"
        f"{st.session_state.control_fields['008']['place']}"
        f"{st.session_state.control_fields['008']['illustrations']}"
        f"{st.session_state.control_fields['008']['target_audience']}"
        f"{st.session_state.control_fields['008']['form_of_item']}"
        f"{st.session_state.control_fields['008']['nature_of_contents']}"
        f"{st.session_state.control_fields['008']['govt_pub']}"
        f"{st.session_state.control_fields['008']['conference']}"
        f"{st.session_state.control_fields['008']['festschrift']}"
        f"{st.session_state.control_fields['008']['index']}"
        f"{st.session_state.control_fields['008']['literary_form']}"
        f"{st.session_state.control_fields['008']['biography']}"
        f"#{st.session_state.control_fields['008']['language']}#"  # Position 34 is blank
        f"{st.session_state.control_fields['008']['modified_record']}"
        f"{st.session_state.control_fields['008']['cataloging_source']}"
    )
    
    st.text_input("008 - معاينة الحقل", value=eight_field, disabled=True)
    
    # Initialize session state for MARC data fields
    if "marc_fields" not in st.session_state:
        st.session_state.marc_fields = []


    
    # Field tag input
    tag = st.text_input("وسم الحقل (ثلاثة أرقام)", placeholder="245", key="custom_tag")
    if tag and (len(tag) != 3 or not tag.isdigit()):
        st.error("يجب أن يتكون وسم الحقل من 3 أرقام")
    
    # Data field (010-999)
    col1, col2 = st.columns(2)
    with col1:
        ind1 = st.text_input("المؤشر الأول", max_chars=1, value=" ", placeholder="0-9 أو #")
    with col2:
        ind2 = st.text_input("المؤشر الثاني", max_chars=1, value=" ", placeholder="0-9 أو #")
    
    st.markdown("*الحقول الفرعية*")
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
            # Validate subfields
            valid_subfields = True
            for code, value in subfields:
                if not code or not value or len(code) != 1 or not code.isalnum():
                    st.error(f"رمز أو قيمة الحقل الفرعي غير صالحة: ${code} {value}")
                    valid_subfields = False
                    break
            
            if valid_subfields:
                new_field = {
                    "tag": tag,
                    "ind1": ind1 if ind1.strip() else " ",
                    "ind2": ind2 if ind2.strip() else " ",
                    "subfields": [(c, v) for c, v in subfields if c and v]
                }
                st.session_state.marc_fields.append(new_field)
                st.success("تمت إضافة الحقل!")
    
    st.markdown("*الحقول المضافة:*")
    for i, field in enumerate(st.session_state.marc_fields):
        col1, col2 = st.columns([4, 1])
        with col1:
            subfields_str = " ".join([f"${c} {v}" for c, v in field["subfields"]])
            st.write(f"{field['tag']}: {field['ind1']}{field['ind2']} {subfields_str}")
        
        with col2:
            # Use a unique key for each delete button
            if st.button("🗑", key=f"del_{i}_{field['tag']}"):
                # Remove the field from the list
                st.session_state.marc_fields.pop(i)
                st.rerun()

    # زر إنشاء السجل
    if st.button("💾 حفظ وتوليد الملف"):
        # التحقق من الحقول الإلزامية
        if not st.session_state.control_fields["001"].strip():
            st.session_state.control_fields["001"] = generate_control_number()
        
        record = Record()
        
        # تعيين حقول التحكم
        record.leader = st.session_state.control_fields["000"]
        record.add_field(Field(tag='001', data=st.session_state.control_fields["001"]))
        record.add_field(Field(tag='003', data=st.session_state.control_fields["003"]))
        record.add_field(Field(tag='005', data=st.session_state.control_fields["005"]))
        
        # بناء حقل 008 من المكونات
        eight_field_data = (
            f"{st.session_state.control_fields['008']['entry_date']}"
            f"{st.session_state.control_fields['008']['pub_status']}"
            f"{st.session_state.control_fields['008']['date1']}"
            f"{st.session_state.control_fields['008']['date2']}"
            f"{st.session_state.control_fields['008']['place']}"
            f"{st.session_state.control_fields['008']['illustrations']}"
            f"{st.session_state.control_fields['008']['target_audience']}"
            f"{st.session_state.control_fields['008']['form_of_item']}"
            f"{st.session_state.control_fields['008']['nature_of_contents']}"
            f"{st.session_state.control_fields['008']['govt_pub']}"
            f"{st.session_state.control_fields['008']['conference']}"
            f"{st.session_state.control_fields['008']['festschrift']}"
            f"{st.session_state.control_fields['008']['index']}"
            f"{st.session_state.control_fields['008']['literary_form']}"
            f"{st.session_state.control_fields['008']['biography']}"
            f"#{st.session_state.control_fields['008']['language']}#"  # Position 34 is blank
            f"{st.session_state.control_fields['008']['modified_record']}"
            f"{st.session_state.control_fields['008']['cataloging_source']}"
        )
        
        record.add_field(Field(tag='008', data=eight_field_data))

        # Add all user-defined data fields to record
        for field in st.session_state.marc_fields:
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
        
        # Try to get title for filename
        title = "marc_record"
        for field in record.fields:
            if field.tag == '245':
                for subfield in field.subfields:
                    if subfield.code == 'a':
                        title = subfield.value[:50]  # Limit length
                        break
                break
        
        # Clean title for filename
        import re
        title = re.sub(r'[^\w\s-]', '', title).strip()
        title = re.sub(r'[-\s]+', '_', title)
        
        # Define file paths
        marc_path = os.path.join(output_dir, f"{title}_{timestamp}.mrc")
        txt_path = os.path.join(output_dir, f"{title}_{timestamp}.txt")
        xml_path = os.path.join(output_dir, f"{title}_{timestamp}.xml")
        xml_db_path = os.path.join(XML_DB_DIR, f"{title}_{timestamp}.xml")

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
            "mrc": (marc_path, f"{title}_{timestamp}.mrc", "application/marc"),
            "txt": (txt_path, f"{title}_{timestamp}.txt", "text/plain"), 
            "xml": (xml_path, f"{title}_{timestamp}.xml", "text/xml")
            }

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
# 📄 Tab 2: PDF / Image to MARC
with tab2:
    st.subheader("📄 PDF / Image to MARC File")
    st.write("Upload a PDF (via GROBID) or an Image (via OCR.Space)")

    api_key = st.secrets["groq_api_key"]
    ocr_api_key = st.secrets.get("ocr_space_api_key", "helloworld")

    option = st.radio("Select input type:", ["PDF", "Image"])

    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=["pdf"] if option == "PDF" else ["png", "jpg", "jpeg"]
    )
    current_file_hash = None
    if uploaded_file:
        current_file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()

    should_process = (
        uploaded_file and api_key and
        (current_file_hash != st.session_state.get('processed_file_hash') or 
         st.session_state.get('processed_file_hash') is None)
    )

    if should_process:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        base_filename = Path(uploaded_file.name).stem

        marc_bin_path = os.path.join(output_dir, f"{base_filename}.mrc")
        marc_txt_path = os.path.join(output_dir, f"{base_filename}.txt")
        marc_xml_path = os.path.join(output_dir, f"{base_filename}.xml")
        xml_db_path = os.path.join(XML_DB_DIR, f"{base_filename}.xml")

        if option == "PDF":
            temp_pdf = os.path.join(temp_dir, f"{base_filename}.pdf")
            with st.spinner("Processing PDF..."):
                try:
                    with open(temp_pdf, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.info("Sending to GROBID...")
                    # Limit pages
                    short_pdf_path = os.path.join(temp_dir, f"{base_filename}_first40.pdf")
                    doc = fitz.open(temp_pdf)
                    short_doc = fitz.open()
                    for i in range(min(40, len(doc))):
                        short_doc.insert_pdf(doc, from_page=i, to_page=i)
                    short_doc.save(short_pdf_path)
                    short_doc.close()
                    doc.close()

                    tei = send_pdf_to_grobid_header(short_pdf_path)

                    st.info("Extracting text snippet...")
                    text = extract_text_from_pdf(temp_pdf)
                    st.info("Extracting metadata with AI...")
                    prompt = f"""Extract the following metadata from this academic text and return ONLY valid JSON:
                    - title
                    - authors (full names)

                    Text:
                    {text}
                    """
                    llm_response = ask_groq_model(prompt, api_key)
                    llm_metadata = extract_json_block(llm_response)
                    st.info("Generating MARC records...")
                    record, marc_bin_path, marc_txt_path, marc_xml_path = tei_to_marc(tei, marc_bin_path, temp_pdf, llm_metadata)
                    if os.path.exists(marc_xml_path):
                        shutil.copy2(marc_xml_path, xml_db_path)
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

        elif option == "Image":
            temp_image = os.path.join(temp_dir, f"{base_filename}.png")
            with st.spinner("Processing Image..."):
                try:
                    with open(temp_image, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.info("Extracting text via OCR.Space API...")
                    text = extract_text_from_image(temp_image, ocr_api_key)
                    st.info("Extracting metadata with AI...")
                    prompt = f"""
    Extract the following metadata from this academic text and return ONLY valid JSON, no explanation:
    - title (string)
    - subtitle (string, optional)
    - authors (list of full names)
    - abstract (string, optional)
    - subjects (list of strings, optional)  # MARC 650 field
    - keywords (list of strings, optional)  # Alternative name for subjects
    - publication year (string, YYYY format)
    - publisher (string)
    - doi (string, optional)
    - isbn (string, optional)
    - language (string, ISO 639-1 or 639-3 code)
    - journal title (string, optional)
    - volume (string, optional)
    - issue (string, optional)
    - pages (string, optional)
    - additional notes (string, optional)

    Text:
    {text}
"""


                    llm_response = ask_groq_model(prompt, api_key)
                    llm_metadata = extract_json_block(llm_response)
                    # Minimal TEI for compatibility
                    record, marc_bin_path, marc_txt_path, marc_xml_path = llm_metadata_to_marc(llm_metadata, os.path.join(output_dir, base_filename))

                    if os.path.exists(marc_xml_path):
                        shutil.copy2(marc_xml_path, xml_db_path)
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
                    st.error(f"Error processing image: {str(e)}")
                finally:
                    if os.path.exists(temp_image):
                        try:
                            os.remove(temp_image)
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

                            # Clear session state to avoid FileNotFoundError
                            if "generated_files" in st.session_state:
                                st.session_state.pop("generated_files")

                            if deleted_files:
                                st.success(f"Successfully deleted {len(deleted_files)} files for {base_name}")

                                # Clear session state to avoid referencing deleted files
                                for key in [
                                    "marc_bin_path", "marc_txt_path", "marc_xml_path",
                                    "current_record", "llm_metadata", "base_filename"
                                ]:
                                    if key in st.session_state:
                                        st.session_state.pop(key)

                                # Clear caches so vector DB rebuilds
                                if os.path.exists(FAISS_INDEX_PATH): os.remove(FAISS_INDEX_PATH)
                                if os.path.exists(DOCUMENTS_PATH): os.remove(DOCUMENTS_PATH)
                                if os.path.exists(METADATA_PATH): os.remove(METADATA_PATH)

                                st.rerun()

                            else:
                                st.warning("No files were found to delete")
                        except Exception as e:
                            st.error(f"Error deleting files: {str(e)}")
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

