# app.py - Enhanced Financial Compliance Checker
import os
import re
import time
import pdfplumber
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import faiss
import uuid
import tempfile
import mimetypes
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.retrievers import BM25Retriever
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
load_dotenv()

app = FastAPI(
    title="Financial Compliance Checker API",
    description="Automated compliance verification for financial statements with enhanced accuracy",
    version="1.2.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in environment")
client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1") 

# Global storage for processing jobs
processing_jobs = {}
completed_reports = {}

# Constants for accuracy improvements
SECTION_CATEGORIES = {
    "valuation": ["accounting_policies", "notes"],
    "notes": ["notes"],
    "audit": ["audit_report"],
    "presentation": ["balance_sheet", "income_statement"],
    "risk": ["risk_management"]
}

class ProcessingStatus(BaseModel):
    job_id: str
    status: str  # "processing", "completed", "error"
    progress: float
    message: str
    total_requirements: Optional[int] = None
    processed_requirements: Optional[int] = None
    found_count: Optional[int] = None

class ComplianceResult(BaseModel):
    id: str
    requirement: str
    category: str
    is_main: bool
    compliance_status: str
    page: Optional[int]
    context: Optional[str] = None
    main_requirement: Optional[str] = None

@dataclass
class ProcessingStats:
    start_time: float
    pdf_pages: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    requirements_processed: int = 0
    
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

class CrossEncoderWrapper:
    """Wrapper for CrossEncoder functionality using transformers directly"""
    def __init__(self, model_name="cross-encoder/ms-marco-electra-base", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Reranker model ready (on {self.device})")
    
    def predict(self, pairs, batch_size=16):
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)
                outputs = self.model(**inputs)
                batch_scores = outputs.logits.squeeze().cpu().numpy().tolist()
                if not isinstance(batch_scores, list):
                    batch_scores = [batch_scores]
                scores.extend(batch_scores)
        return scores

class AdvancedFinancialRAG:
    def __init__(self, use_gpu: bool = True):
        self.stats = ProcessingStats(start_time=time.time())
        self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
        print(f"🚀 Initializing AdvancedFinancialRAG on {self.device}")
        
        # Enhanced embedding model
        print("🔄 Loading BAAI/bge-large-en-v1.5 embedding model...")
        self.embedder = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",  # Improved semantic similarity
            model_kwargs={"device": self.device},
            encode_kwargs={"batch_size": 16, "normalize_embeddings": True}
        )
        print("✅ Embedding model ready")
        
        # Enhanced reranker
        print("🔄 Loading cross-encoder reranker...")
        self.reranker = CrossEncoderWrapper("cross-encoder/ms-marco-electra-base")
        
        # Initialize vector stores
        self.vector_store = None
        self.bm25_retriever = None

    def build_optimized_vector_store(self, documents: List[Document]) -> FAISS:
        print("🔍 Building optimized vector store with hybrid retrieval...")
        start_time = time.time()
        
        # FAISS vector store with improved embeddings
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedder
        )
        
        # BM25 keyword-based retriever
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 3
        
        print(f"✅ Vector store built with {len(documents)} documents in {time.time() - start_time:.2f}s")
        return self.vector_store

def load_and_prepare_checklist(filepath: str, sheet_name: str = "0") -> List[Dict]:
    # Convert to int if it's a digit, otherwise keep as string
    sheet = int(sheet_name) if sheet_name.isdigit() else sheet_name
    try:
        # Get all sheet names
        excel_file = pd.ExcelFile(filepath)
        print(f"📋 Available sheets: {excel_file.sheet_names}")
        # Read the selected sheet
        df = pd.read_excel(filepath, sheet_name=sheet, header=None).fillna("")
        print(f"✅ Loaded sheet: {sheet if isinstance(sheet, int) else sheet_name}")
    except Exception as e:
        print(f"⚠️ Error loading sheet '{sheet_name}': {str(e)}")
        # Try to load the first sheet as fallback
        print("🔄 Trying to load first sheet as fallback...")
        df = pd.read_excel(filepath, sheet_name=0, header=None).fillna("")

    if df.empty:
        print("⚠️ Selected sheet is empty, loading first sheet instead")
        df = pd.read_excel(filepath, sheet_name=0, header=None).fillna("")

    # Handle multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.map(lambda x: '_'.join([str(i) for i in x]))
    
    # Ensure we have at least 3 columns
    if len(df.columns) < 3:
        df = pd.concat([df, pd.DataFrame(columns=range(3 - len(df.columns)))], axis=1)
    
    df.columns = ['A', 'B', 'C'] + list(df.columns[3:])
    structured = []
    current_category = current_main = None
    main_id_counter = 0
    
    print("📋 Processing checklist...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row['A']).strip()
        if not text or text.upper() in ['PRESENTATIE', 'JAARVERSLAGGEVING WAARDERINGSGRONDSLAGEN', 'N/A']:
            continue
            
        if len(text.split()) <= 3 and (text.isupper() or text.startswith('**')):
            current_category = re.sub(r'[\*\:]+', '', text).strip()
            continue
            
        if text.endswith(('.', ':')):
            current_main = {
                'id': f"main_{main_id_counter}",
                'main_condition': text,
                'category': current_category,
                'sub_conditions': [],
                'row_index': idx
            }
            structured.append(current_main)
            main_id_counter += 1
            continue
            
        if any(text.startswith(c) for c in ['-', '•', 'a.', 'b.', 'c.', 'd.', 'e.', 'f.']) and current_main:
            cleaned_text = re.sub(r'^[-•→a-f]\s*\.?\s*', '', text).strip()
            current_main['sub_conditions'].append({
                'id': f"{current_main['id']}_sub",
                'text': cleaned_text,
                'main_id': current_main['id'],
                'main_condition': current_main['main_condition'],
                'category': current_category,
                'row_index': idx
            })
    
    requirements = []
    for item in structured:
        requirements.append({
            'id': item['id'],
            'text': item['main_condition'],
            'category': item['category'],
            'is_main': True,
            'row_index': item['row_index']
        })
        for sub in item['sub_conditions']:
            requirements.append({
                'id': sub['id'],
                'text': sub['text'],
                'main_id': sub['main_id'],
                'main_condition': sub['main_condition'],
                'category': sub['category'],
                'is_main': False,
                'row_index': sub['row_index']
            })
    
    print(f"✅ Processed {len(requirements)} requirements")
    return requirements

def process_financial_pdf(pdf_path: str) -> List[Document]:
    print(f"📄 Processing PDF: {pdf_path}")
    start_time = time.time()
    
    section_patterns = {
        'notes': r'toelichting|notes|verklaring',
        'accounting_policies': r'waardering|valuation|grondslagen',
        'balance_sheet': r'balans|balance',
        'income_statement': r'resultaten|overzicht|baten|lasten',
        'audit_report': r'controleverklaring|audit',
        'risk_management': r'risico|risicobeheer'
    }
    
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for i, page in enumerate(tqdm(pdf.pages, desc="Extracting pages", total=total_pages)):
            text = page.extract_text() or ""
            if not text.strip():
                continue
                
            section = "other"
            for sec_name, pattern in section_patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    section = sec_name
                    break
            
            # Enhanced chunking strategy with better overlap
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,  # Increased chunk size for better context
                chunk_overlap=500,  # Increased overlap
                separators=["\n\n", "\n", "(?<=\. )", "; ", ", ", " "]
            )
            
            chunks = splitter.split_text(text)
            for chunk in chunks:
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        'page': i + 1,
                        'section': section,
                        'source': f"Page {i+1}"
                    }
                ))
    
    print(f"✅ Created {len(documents)} chunks in {time.time() - start_time:.2f}s")
    return documents

def generate_queries(requirement: str, client: OpenAI) -> List[str]:
    """Generate multiple search queries using LLM paraphrasing"""
    prompt = f"Generate 5 diverse search queries (in Dutch) to find evidence for this requirement:\n{requirement}"
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        ).choices[0].message.content
        
        queries = [q.strip() for q in response.split("\n") if q.strip()]
        print(f"🔄 Generated {len(queries)} queries for requirement: {requirement[:50]}...")
        return queries
    except Exception as e:
        print(f"⚠️ Fallback to simple queries: {str(e)}")
        return [
            requirement,
            requirement.lower(),
            requirement.replace("waardering", "valuation"),
            f"Controle of: {requirement}"
        ]

def retrieve_evidence(
    requirement: str, 
    vector_store: FAISS, 
    bm25_retriever: BM25Retriever,
    reranker,
    req_category: str,
    top_k=3
) -> List[Tuple[Document, float]]:
    """Retrieve evidence using hybrid search with section filtering"""
    docs = []
    scores = []
    
    # Section filtering based on category
    category_sections = SECTION_CATEGORIES.get(req_category.lower(), [])
    print(f"🔍 Searching sections {category_sections} for category {req_category}")
    
    # Generate diverse queries
    queries = generate_queries(requirement, client)
    
    # Semantic search with FAISS
    for q in queries:
        try:
            results = vector_store.similarity_search_with_score(q, k=5)
            for doc, score in results:
                # Apply section filtering
                if not category_sections or doc.metadata["section"] in category_sections:
                    docs.append(doc)
                    scores.append(score)
        except Exception as e:
            print(f"⚠️ Retrieval error for query '{q}': {str(e)}")
            continue
    
    # Keyword search with BM25
    try:
        bm25_results = bm25_retriever.invoke(requirement)
        for doc in bm25_results:
            if doc not in docs:  # Avoid duplicates
                docs.append(doc)
    except Exception as e:
        print(f"⚠️ BM25 retrieval error: {str(e)}")
    
    if not docs:
        return []
    
    # Reranking with cross-encoder
    pairs = [(requirement, doc.page_content) for doc in docs]
    try:
        rerank_scores = reranker.predict(pairs, batch_size=16)
    except Exception as e:
        print(f"⚠️ Reranking error: {str(e)}")
        rerank_scores = [0] * len(docs)
    
    # Combine documents with both scores
    scored_docs = list(zip(docs, rerank_scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for doc, score in scored_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            deduped.append((doc, score))
    
    return deduped[:top_k]

def make_context(evidence: List[Tuple[Document, float]], max_words: int = 1600) -> str:
    context = ""
    word_count = 0
    for doc, score in evidence[:3]:
        chunk = f"[Page {doc.metadata['page']} | Section: {doc.metadata['section']} | Score: {score:.2f}]\n{doc.page_content.strip()}\n\n"
        wc = len(chunk.split())
        if word_count + wc > max_words:
            break
        context += chunk
        word_count += wc
    return context.strip()

def check_compliance(requirement: str, evidence: List[Tuple[Document, float]]) -> Dict:
    """Check compliance with enhanced prompt engineering"""
    if not evidence:
        return {"compliance": "Not found", "page": None, "context": None}
    
    context = make_context(evidence)
    
    # Enhanced prompt with few-shot examples
    prompt = f"""### Role ###
You are an expert Dutch financial auditor. Your task is to determine compliance with specific requirements using evidence from the provided context.

### Examples ###
Requirement: "Waardeverminderingen moeten worden toegepast."
Context: "[Page 45 | Section: accounting_policies] De onderneming past waardeverminderingen toe volgens IFRS 9."
Response: Compliance: Found\nPage: 45

Requirement: "Inventaris moet worden geëvalueerd."
Context: "[Page 30 | Section: balance_sheet] Inventaris is geëvalueerd tegen lager van kost of netto verkoopwaarde."
Response: Compliance: Found\nPage: 30

### Requirement ###
{requirement}

### Context Excerpt ###
{context}

### Instructions ###
1. Analyze compliance (fully/partially).
2. Return the SINGLE most relevant page number.
3. If not found, respond "Not found".
4. Prioritize evidence from accounting policies and notes sections.
5. If partial evidence exists, default to "Found".

### Response Format ###
Compliance: [Found/Not found]
Page: [number/null]"""

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,  # Reduced temperature for more deterministic output
            max_tokens=128,
            top_p=0.95
        ).choices[0].message.content
        
        # Parse response
        compliance_match = re.search(r'Compliance:\s*(Found|Not found|Uncertain)', response, re.IGNORECASE)
        page_match = re.search(r'Page:\s*(\d+)', response)
        
        compliance = compliance_match.group(1).capitalize() if compliance_match else "Uncertain"
        page = int(page_match.group(1)) if page_match else None
        
        return {"compliance": compliance, "page": page, "context": context}
    
    except Exception as e:
        print(f"⚠️ API Error: {str(e)}")
        return {"compliance": "Error", "page": None, "context": None}

def process_compliance_job(job_id: str, checklist_path: str, pdf_path: str, sheet_name: str = "0"):
    """Background task to process compliance checking"""
    try:
        processing_jobs[job_id] = ProcessingStatus(
            job_id=job_id,
            status="processing",
            progress=0.0,
            message="Loading checklist and PDF..."
        )
        
        # Load checklist with selected sheet
        checklist = load_and_prepare_checklist(checklist_path, sheet_name)
        processing_jobs[job_id].total_requirements = len(checklist)
        processing_jobs[job_id].message = "Processing PDF..."
        processing_jobs[job_id].progress = 10.0
        
        # Process PDF
        documents = process_financial_pdf(pdf_path)
        processing_jobs[job_id].message = "Building vector store..."
        processing_jobs[job_id].progress = 30.0
        
        # Build RAG system
        rag = AdvancedFinancialRAG()
        rag.build_optimized_vector_store(documents)
        
        processing_jobs[job_id].message = "Checking compliance requirements..."
        processing_jobs[job_id].progress = 50.0
        
        results = []
        found_count = 0
        
        for i, req in enumerate(checklist):
            query = req['text'] if req['is_main'] else f"{req['main_condition']}: {req['text']}"
            evidence = retrieve_evidence(
                query, 
                rag.vector_store, 
                rag.bm25_retriever,
                rag.reranker,
                req.get('category', 'other')
            )
            
            compliance = check_compliance(query, evidence)
            
            result = {
                "id": req['id'],
                "requirement": req['text'],
                "category": req['category'],
                "is_main": req['is_main'],
                "compliance_status": compliance["compliance"],
                "page": compliance["page"],
                "context": compliance["context"]
            }
            
            if not req['is_main']:
                result["main_requirement"] = req['main_condition']
                
            if compliance["compliance"] == "Found":
                found_count += 1
                
            results.append(result)
            
            # Update progress
            progress = 50.0 + (i + 1) / len(checklist) * 45.0
            processing_jobs[job_id].progress = progress
            processing_jobs[job_id].processed_requirements = i + 1
            processing_jobs[job_id].found_count = found_count
        
        # Save results
        df = pd.DataFrame(results)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"reports/compliance_report_{job_id}_{timestamp}.xlsx"
        os.makedirs("reports", exist_ok=True)
        df.to_excel(output_file, index=False)
        
        # Mark as completed
        processing_jobs[job_id].status = "completed"
        processing_jobs[job_id].progress = 100.0
        processing_jobs[job_id].message = "Processing completed!"
        
        completed_reports[job_id] = {
            "results": results,
            "report_file": output_file,
            "found_count": found_count,
            "total_count": len(results)
        }
        
    except Exception as e:
        processing_jobs[job_id].status = "error"
        processing_jobs[job_id].message = f"Error: {str(e)}"
        print(f"Error in processing job {job_id}: {str(e)}")

@app.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    checklist: UploadFile = File(...),
    pdf: UploadFile = File(...),
    sheet_name: str = Form("0")  # Default to first sheet
):
    """Upload files and start processing with sheet selection"""
    job_id = str(uuid.uuid4())
    
    # Validate file types
    if checklist.content_type not in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        raise HTTPException(status_code=400, detail="Checklist must be an Excel file")
        
    if pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Financial statement must be a PDF file")
    
    # Save uploaded files
    os.makedirs("temp", exist_ok=True)
    checklist_path = f"temp/checklist_{job_id}.xlsx"
    pdf_path = f"temp/pdf_{job_id}.pdf"
    
    with open(checklist_path, "wb") as f:
        f.write(await checklist.read())
        
    with open(pdf_path, "wb") as f:
        f.write(await pdf.read())
    
    # Start background processing with sheet name
    background_tasks.add_task(process_compliance_job, job_id, checklist_path, pdf_path, sheet_name)
    return JSONResponse(content={"job_id": job_id, "message": "Processing started"})

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get processing status"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return processing_jobs[job_id]

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Get processing results"""
    if job_id not in completed_reports:
        raise HTTPException(status_code=404, detail="Results not ready or job not found")
    return completed_reports[job_id]

@app.get("/download/{job_id}")
async def download_report(job_id: str):
    """Download Excel report"""
    if job_id not in completed_reports:
        raise HTTPException(status_code=404, detail="Report not ready or job not found")
    report_file = completed_reports[job_id]["report_file"]
    
    # Stream the file for better performance with large files
    def iterfile():
        with open(report_file, mode="rb") as file_like:
            yield from file_like
            
    return StreamingResponse(
        iterfile(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f"attachment; filename=compliance_report_{job_id}.xlsx",
            "Content-Length": str(os.path.getsize(report_file))
        }
    )

@app.get("/sheets/{job_id}")
async def get_excel_sheets(job_id: str):
    """Get available sheets from the uploaded Excel file"""
    checklist_path = f"temp/checklist_{job_id}.xlsx"
    if not os.path.exists(checklist_path):
        raise HTTPException(status_code=404, detail="Checklist file not found")
        
    try:
        excel_file = pd.ExcelFile(checklist_path)
        sheets = [
            {"name": name, "index": i} 
            for i, name in enumerate(excel_file.sheet_names)
        ]
        return {"sheets": sheets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading Excel file: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.2.0", "timestamp": time.time()}

@app.on_event("startup")
async def startup_event():
    """Create necessary directories on startup"""
    os.makedirs("temp", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    print("🚀 Server started. Directories created.")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up temporary files on shutdown"""
    print("🧹 Cleaning up temporary files...")
    for file in os.listdir("temp"):
        os.remove(os.path.join("temp", file))
    print("✅ Temporary files cleaned")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)