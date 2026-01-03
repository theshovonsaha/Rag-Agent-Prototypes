#!/bin/bash

# PDF RAG Agent - Automated Setup Script
# This script sets up a complete RAG system with Streamlit, ChromaDB, Neo4j, and Groq

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}PDF RAG Agent - Automated Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 9 ] && [ "$PYTHON_MINOR" -le 11 ]; then
        PYTHON_CMD="python3"
        echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
    else
        echo -e "${RED}✗ Python 3.9-3.11 required. Found: $PYTHON_VERSION${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Python 3 not found. Install Python 3.9-3.11${NC}"
    exit 1
fi

# Check Docker
echo -e "${YELLOW}Checking Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not found. Install Docker first${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker found${NC}"

# Create project directory
PROJECT_DIR="pdf-rag-agent"
echo -e "${YELLOW}Creating project directory: $PROJECT_DIR${NC}"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# Generate requirements.txt
echo -e "${YELLOW}Creating requirements.txt...${NC}"
cat > requirements.txt << 'EOF'
streamlit==1.39.0
PyPDF2==3.0.1
chromadb==1.4.0
neo4j==5.26.0
groq==0.11.0
python-dotenv==1.0.1
sentence-transformers==3.3.1
EOF
echo -e "${GREEN}✓ requirements.txt created${NC}"

# Generate utils.py
echo -e "${YELLOW}Creating utils.py...${NC}"
cat > utils.py << 'EOF'
import PyPDF2
from typing import List

def extract_text_from_pdf(pdf_file) -> str:
    """Extract all text from PDF."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    
    chunk_size: Characters per chunk
    overlap: Characters shared between chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Don't split mid-word
        if end < len(text) and not text[end].isspace():
            last_space = chunk.rfind(' ')
            if last_space > 0:
                end = start + last_space
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks
EOF
echo -e "${GREEN}✓ utils.py created${NC}"

# Generate rag_system.py
echo -e "${YELLOW}Creating rag_system.py...${NC}"
cat > rag_system.py << 'EOF'
import os
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

class PDFRAGSystem:
    """RAG system combining vector and graph search."""
    
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="pdf_documents")
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.groq_client = Groq(api_key=groq_api_key)
        
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        if not all([neo4j_uri, neo4j_user, neo4j_password]):
            raise ValueError("Neo4j credentials missing in environment variables")
        
        try:
            self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            self.neo4j_driver.verify_connectivity()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j: {e}")
        
        self._setup_neo4j_schema()
    
    def _setup_neo4j_schema(self):
        with self.neo4j_driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
            session.run("CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.document_name)")
    
    def add_document(self, text: str, chunks: List[str], doc_name: str):
        embeddings = self.embedder.encode(chunks).tolist()
        ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"document": doc_name, "chunk_id": i} for i in range(len(chunks))]
        
        self.collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        
        with self.neo4j_driver.session() as session:
            for i, chunk in enumerate(chunks):
                chunk_id = ids[i]
                session.run("""
                    MERGE (c:Chunk {id: $chunk_id})
                    SET c.text = $text, c.document_name = $doc_name, c.chunk_index = $chunk_index
                """, chunk_id=chunk_id, text=chunk, doc_name=doc_name, chunk_index=i)
                
                if i < len(chunks) - 1:
                    next_id = ids[i + 1]
                    session.run("""
                        MATCH (c1:Chunk {id: $chunk_id})
                        MATCH (c2:Chunk {id: $next_id})
                        MERGE (c1)-[:NEXT]->(c2)
                    """, chunk_id=chunk_id, next_id=next_id)
    
    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict]:
        query_embedding = self.embedder.encode([query])[0].tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        
        context_chunks = []
        for i in range(len(results['ids'][0])):
            chunk_id = results['ids'][0][i]
            chunk_text = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            
            with self.neo4j_driver.session() as session:
                neighbors = session.run("""
                    MATCH (c:Chunk {id: $chunk_id})-[:NEXT*0..1]-(neighbor:Chunk)
                    RETURN neighbor.text as text
                    ORDER BY neighbor.chunk_index
                """, chunk_id=chunk_id).data()
                
                extended_text = chunk_text
                if neighbors:
                    neighbor_texts = [n['text'] for n in neighbors if n['text'] != chunk_text]
                    if neighbor_texts:
                        extended_text += "\n\n" + "\n".join(neighbor_texts[:2])
            
            context_chunks.append({'text': extended_text, 'document': metadata['document'], 'chunk_id': metadata['chunk_id']})
        
        return context_chunks
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        context_text = "\n\n---\n\n".join([f"[Source: {c['document']}, Chunk {c['chunk_id']}]\n{c['text']}" for c in context_chunks])
        
        prompt = f"""You are a helpful assistant answering questions from documents.

Context:
{context_text}

Question: {query}

Instructions:
- Answer using only the context above
- If the answer isn't in the context, say "I don't have that information"
- Cite sources by document name and chunk number
- Be concise

Answer:"""
        
        response = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def query(self, question: str) -> Dict:
        context_chunks = self.retrieve_context(question, top_k=3)
        if not context_chunks:
            return {'answer': "No relevant information found.", 'sources': []}
        answer = self.generate_answer(question, context_chunks)
        return {'answer': answer, 'sources': context_chunks}
    
    def close(self):
        self.neo4j_driver.close()
EOF
echo -e "${GREEN}✓ rag_system.py created${NC}"

# Generate app.py
echo -e "${YELLOW}Creating app.py...${NC}"
cat > app.py << 'EOF'
import streamlit as st
from rag_system import PDFRAGSystem
from utils import extract_text_from_pdf, chunk_text

st.set_page_config(page_title="PDF RAG Agent", page_icon="📚", layout="wide")
st.title("📚 PDF RAG Agent")
st.markdown("Upload PDFs. Ask questions. Get cited answers.")

if 'rag_system' not in st.session_state:
    try:
        st.session_state.rag_system = PDFRAGSystem()
        st.session_state.documents = []
    except (ValueError, ConnectionError) as e:
        st.error(f"❌ Setup Error: {e}")
        st.info("Check your .env file and ensure Neo4j is running")
        st.stop()

with st.sidebar:
    st.header("Upload Documents")
    uploaded_file = st.file_uploader("Choose a PDF", type=['pdf'])
    
    if uploaded_file and uploaded_file.name not in st.session_state.documents:
        with st.spinner('Processing PDF...'):
            text = extract_text_from_pdf(uploaded_file)
            chunks = chunk_text(text, chunk_size=500, overlap=50)
            st.session_state.rag_system.add_document(text=text, chunks=chunks, doc_name=uploaded_file.name)
            st.session_state.documents.append(uploaded_file.name)
            st.success(f"✓ Added {uploaded_file.name} ({len(chunks)} chunks)")
    
    if st.session_state.documents:
        st.subheader("Loaded Documents")
        for doc in st.session_state.documents:
            st.text(f"📄 {doc}")

st.header("Ask Questions")
question = st.text_input("Your question:", placeholder="What does this document say about...?")

if st.button("Get Answer") and question:
    with st.spinner('Thinking...'):
        result = st.session_state.rag_system.query(question)
        st.subheader("Answer")
        st.write(result['answer'])
        
        if result['sources']:
            st.subheader("Sources")
            for i, source in enumerate(result['sources'], 1):
                with st.expander(f"Source {i}: {source['document']} (Chunk {source['chunk_id']})"):
                    preview = source['text'][:500]
                    if len(source['text']) > 500:
                        preview += "..."
                    st.text(preview)

st.markdown("---")
st.markdown("Built with Streamlit • ChromaDB • Neo4j • Groq")
EOF
echo -e "${GREEN}✓ app.py created${NC}"

# Create .gitignore
echo -e "${YELLOW}Creating .gitignore...${NC}"
cat > .gitignore << 'EOF'
venv/
.env
__pycache__/
*.pyc
chroma_db/
.DS_Store
*.pdf
EOF
echo -e "${GREEN}✓ .gitignore created${NC}"

# Create .env.example
echo -e "${YELLOW}Creating .env.example...${NC}"
cat > .env.example << 'EOF'
GROQ_API_KEY=your_groq_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=rag_password_2026
EOF
echo -e "${GREEN}✓ .env.example created${NC}"

# Get Groq API key from user
echo
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Configuration${NC}"
echo -e "${YELLOW}========================================${NC}"
echo
echo -e "${YELLOW}Enter your Groq API key${NC}"
echo -e "${YELLOW}(Get one free at: https://console.groq.com)${NC}"
read -p "Groq API Key: " GROQ_KEY

# Create .env file
echo -e "${YELLOW}Creating .env file...${NC}"
cat > .env << EOF
GROQ_API_KEY=$GROQ_KEY
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=rag_password_2026
EOF
echo -e "${GREEN}✓ .env file created${NC}"

# Start Neo4j in Docker
echo -e "${YELLOW}Starting Neo4j in Docker...${NC}"
docker run -d \
    --name pdf-rag-neo4j \
    -p 7474:7474 \
    -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/rag_password_2026 \
    neo4j:latest

echo -e "${YELLOW}Waiting for Neo4j to start (15 seconds)...${NC}"
sleep 15
echo -e "${GREEN}✓ Neo4j started${NC}"

# Create virtual environment
echo -e "${YELLOW}Creating Python virtual environment...${NC}"
$PYTHON_CMD -m venv venv
echo -e "${GREEN}✓ Virtual environment created${NC}"

# Activate and install dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
echo -e "${YELLOW}(This may take 2-3 minutes)${NC}"
source venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create run script
echo -e "${YELLOW}Creating run.sh script...${NC}"
cat > run.sh << 'RUNEOF'
#!/bin/bash
echo "Starting PDF RAG Agent..."

# Check if Neo4j is running
if ! docker ps | grep -q pdf-rag-neo4j; then
    echo "Starting Neo4j..."
    docker start pdf-rag-neo4j
    sleep 10
fi

# Activate venv and run
source venv/bin/activate
streamlit run app.py
RUNEOF
chmod +x run.sh
echo -e "${GREEN}✓ run.sh created${NC}"

# Create shutdown script
echo -e "${YELLOW}Creating shutdown.sh script...${NC}"
cat > shutdown.sh << 'SHUTEOF'
#!/bin/bash
echo "Shutting down PDF RAG Agent..."

# Stop Streamlit (find and kill process)
pkill -f "streamlit run app.py"

# Stop Neo4j
docker stop pdf-rag-neo4j

echo "✓ Shutdown complete"
SHUTEOF
chmod +x shutdown.sh
echo -e "${GREEN}✓ shutdown.sh created${NC}"

# Create test script
echo -e "${YELLOW}Creating test.sh script...${NC}"
cat > test.sh << 'TESTEOF'
#!/bin/bash
echo "Testing PDF RAG Agent setup..."

# Test Python
if ! python3 --version > /dev/null 2>&1; then
    echo "✗ Python not found"
    exit 1
fi
echo "✓ Python found"

# Test virtual environment
if [ ! -d "venv" ]; then
    echo "✗ Virtual environment not found"
    exit 1
fi
echo "✓ Virtual environment exists"

# Test .env file
if [ ! -f ".env" ]; then
    echo "✗ .env file not found"
    exit 1
fi
echo "✓ .env file exists"

# Test Neo4j
if ! docker ps | grep -q pdf-rag-neo4j; then
    echo "✗ Neo4j not running"
    exit 1
fi
echo "✓ Neo4j is running"

# Test Python imports
source venv/bin/activate
python3 -c "import streamlit, chromadb, neo4j, groq" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ All Python packages installed"
else
    echo "✗ Missing Python packages"
    exit 1
fi

echo
echo "✓ All tests passed! Run './run.sh' to start the app"
TESTEOF
chmod +x test.sh
echo -e "${GREEN}✓ test.sh created${NC}"

# Final success message
echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete! 🎉${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Start the app:     ${GREEN}./run.sh${NC}"
echo -e "  2. Open browser:      ${GREEN}http://localhost:8501${NC}"
echo -e "  3. Upload a PDF and ask questions!"
echo
echo -e "${YELLOW}Other commands:${NC}"
echo -e "  • Stop everything:    ${GREEN}./shutdown.sh${NC}"
echo -e "  • Test setup:         ${GREEN}./test.sh${NC}"
echo -e "  • View Neo4j:         ${GREEN}http://localhost:7474${NC}"
echo
echo -e "${YELLOW}Project location:${NC}"
echo -e "  $(pwd)"
echo
