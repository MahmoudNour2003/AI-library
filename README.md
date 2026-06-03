# 📚 AI-Library: Intelligent MARC Record Management System

AI-Library is a modern, web-based platform built with **Streamlit** that simplifies the cataloging of bibliographic metadata. By combining classical MARC (Machine-Readable Cataloging) tools with AI capabilities (LLMs, OCR, and RAG), it allows librarians, researchers, and developers to easily create, parse, convert, query, and store library records.

---

## ✨ Key Features

### 1. 📋 Manual MARC Entry (`إدخل MARC يدويًا`)
* **Structured Input**: Interactive form fields to specify the Leader (Field `000`), Control Number (`001`), System Identifier (`003`), Last Modification date (`005`), and the complex configuration of Field `008` (containing entry date, publication status, year, place, language, etc.).
* **Custom Tags & Subfields**: Dynamic addition of data fields (tags `010` to `999`), indicators, and subfields (`$a`, `$b`, etc.).
* **Visual Verification**: Live preview of the MARC record string structure during editing.

### 2. 📄 AI-Powered PDF to MARC (`تحويل PDF إلى MARC`)
* **Structure Parsing (GROBID)**: Automatically parses PDF headers, title zones, and citation metrics using the GROBID API.
* **Semantic Analysis (Groq + Llama 3)**: Extracts clean metadata (Title, Authors, Abstract) from PDF text snippets using `llama-3.1-8b-instant` through Groq.
* **Automated Generation**: Maps extracted info into fields like `245` (Title), `100/700` (Authors), `264` (Publisher & date), and `520` (Abstract Summary) to export a valid MARC format.

### 3. 🖼️ AI-Powered Image to MARC (`استخراج بيانات من marc file`)
* **Arabic & English OCR (OCR.Space)**: Recognizes text from scanned book covers, title pages, or catalog cards in Arabic (`ara`) or English (`eng`).
* **Intelligent Entity Extraction**: Leverages Groq to extract structured metadata (titles, subtitles, authors, publishers, publication year, ISBN/DOI, language, journal volume/issue/pages).
* **Direct MARC Conversion**: Automatically instantiates binary, text, and XML MARC records directly from the structured JSON metadata.

### 4. 📂 Read Existing MARC Files
* Upload binary `.mrc` records to display them in a human-readable format.
* Automatically registers uploaded records into the active database to include them in search queries and semantic Q&A.

### 5. 🔍 Saved Records Library
* **Records Viewer**: Browse and inspect saved files in the `output/` directory.
* **Format Exports**: Download generated records in three distinct formats:
  * **Binary MARC** (`.mrc`)
  * **Text Representation** (`.txt`)
  * **MARCXML** (`.xml`)
* **Management**: Safely delete files and trigger automatic updates of indices.

### 6. 🤖 Intelligent Q&A Assistant (`مساعد الأسئلة والأجوبة`)
* **Hybrid Search (FAISS + Embeddings)**: Indexes all saved XML database records dynamically using PyTorch and `SentenceTransformers` (`all-MiniLM-L6-v2`).
* **Retrieval-Augmented Generation (RAG)**: Processes natural language questions (e.g. *"Who is the author of X?"*, *"Which books are written in Arabic?"*), retrieves relevant context, and uses the Groq LLM to respond.
* **Source Attribution**: Highlights the specific MARC record files and metadata fields used to answer the query.

---

## 🛠️ Technology Stack

* **Frontend & Interface**: [Streamlit](https://streamlit.io/)
* **Cataloging Engine**: [pymarc](https://github.com/edsu/pymarc) (MARC21 reader/writer/XML writer)
* **Metadata Extraction**: [Groq Cloud API](https://groq.com/) (`llama-3.1-8b-instant`)
* **OCR System**: [OCR.Space API](https://ocr.space/)
* **Bibliographic Parsing**: [GROBID API](https://github.com/kermitt2/grobid)
* **Vector Search Engine**: [FAISS (CPU-based)](https://github.com/facebookresearch/faiss)
* **Embeddings Model**: `sentence-transformers/all-MiniLM-L6-v2` (from HuggingFace)
* **PDF Utility**: [PyMuPDF (fitz)](https://github.com/pymupdf/PyMuPDF)
* **Language Detection**: `langdetect`

---

## 🚀 Installation & Local Setup

### Prerequisites
Make sure you have Python 3.10+ installed on your system.

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MahmoudNour2003/AI-library.git
   cd AI-library
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Secrets**:
   Create a Streamlit secrets file at `.streamlit/secrets.toml` in your project directory:
   ```toml
   groq_api_key = "YOUR_GROQ_API_KEY_HERE"
   ocr_space_api_key = "YOUR_OCR_SPACE_API_KEY_HERE" # Optional (defaults to "helloworld" if empty)
   ```
   * *Note: You can get your Groq API key from [Groq Console](https://console.groq.com/) and OCR.Space API key from [OCR.Space](https://ocr.space/ocrapi).*

4. **Run the Application**:
   ```bash
   streamlit run app5.py
   ```
   The application will automatically open in your web browser at `http://localhost:8501`.

---

## 🐳 Dev Container Configuration

This project is pre-configured with a Development Container specification (`.devcontainer/devcontainer.json`). If you are using **VS Code** (with the *Dev Containers* extension) or **GitHub Codespaces**, the environment setup will be completely automated:
* Installs all APT system-level dependencies.
* Sets up a Python 3.11 container.
* Automatically installs Python libraries from `requirements.txt`.
* Forwards the app port (`8501`) and initiates the Streamlit server.

---

## 📂 Project Directory Structure

```text
├── .devcontainer/
│   └── devcontainer.json   # Development container configuration
├── output/                 # Contains generated .mrc, .txt, and .xml files
├── xml_database/           # Cached XML records used for vector search indexing
├── app5.py                 # Core application code (Streamlit interface & helper scripts)
├── requirements.txt        # Python dependencies
├── runtime.txt             # Deployment runtime python environment specification
├── faiss_index.bin         # Generated FAISS vector database (auto-built)
├── documents.pkl           # Raw document text strings cache (auto-built)
└── metadata.pkl            # Extracted metadata cache (auto-built)
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information (if applicable).
