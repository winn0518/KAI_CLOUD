from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os
from werkzeug.utils import secure_filename
import tempfile
import shutil

app = Flask(__name__)
CORS(app)
app.secret_key = 'kai-cloud-secret-key-2024'  # Required for sessions

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'doc'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables
vector_store = None
qa_chain = None
current_docs = []

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_vector_store():
    """Clear existing vector store"""
    global vector_store, qa_chain, current_docs
    vector_store = None
    qa_chain = None
    current_docs = []
    
    # Clean up Chroma database
    if os.path.exists("./chroma_db"):
        try:
            shutil.rmtree("./chroma_db")
            print("Cleared Chroma database")
        except Exception as e:
            print(f"Error clearing Chroma DB: {e}")

def load_document(file_path, file_extension):
    """Load a single document based on file type"""
    try:
        print(f"Loading {file_extension} file: {file_path}")
        
        if file_extension == 'pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == 'txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_extension in ['docx', 'doc']:
            loader = Docx2txtLoader(file_path)
        else:
            return []
        
        documents = loader.load()
        print(f"Loaded {len(documents)} pages/sections")
        return documents
    except Exception as e:
        print(f"Error loading document: {str(e)}")
        return []

def create_vector_store(documents):
    """Create vector store from documents"""
    global vector_store
    
    try:
        # Create better text chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} text chunks")
        
        if len(chunks) == 0:
            print("Warning: No text chunks created from document")
            return False
        
        # Initialize embeddings with a good model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create persistent vector store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db",
            collection_name="document_qa"
        )
        
        # Persist the database
        vector_store.persist()
        print("Vector store created and persisted successfully")
        return True
        
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        return False

def initialize_qa_chain():
    """Initialize the QA chain with LLM"""
    global vector_store, qa_chain
    
    try:
        # Get HuggingFace API token from environment
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            print("Warning: HUGGINGFACEHUB_API_TOKEN not set")
            # For testing, you can set it here temporarily (remove in production)
            hf_token = "your_token_here"
        
        # Initialize LLM - using a free model that works well
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-large",  # Good free model for QA
            huggingfacehub_api_token=hf_token,
            model_kwargs={
                "temperature": 0.1,
                "max_length": 500,
                "top_p": 0.95
            }
        )
        
        # Create retriever with better parameters
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 3,  # Number of documents to retrieve
                "fetch_k": 10  # Number of documents to initially fetch
            }
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # Simple chain type
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "verbose": True
            }
        )
        
        print("QA chain initialized successfully")
        return True
        
    except Exception as e:
        print(f"Error initializing QA chain: {str(e)}")
        return False

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'KAI Cloud Document Q&A',
        'vector_store_ready': vector_store is not None
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    global current_docs
    
    if 'file' not in request.files:
        return jsonify({
            'success': False, 
            'message': 'No file provided'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'success': False, 
            'message': 'No file selected'
        }), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'success': False, 
            'message': 'File type not allowed. Please upload PDF, TXT, DOCX, or DOC files.'
        }), 400
    
    # Clear previous documents
    clear_vector_store()
    
    # Save uploaded file temporarily
    filename = secure_filename(file.filename)
    file_extension = filename.rsplit('.', 1)[1].lower()
    temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        # Save file
        file.save(temp_filepath)
        print(f"File saved to: {temp_filepath}")
        
        # Load document
        documents = load_document(temp_filepath, file_extension)
        
        if not documents:
            return jsonify({
                'success': False, 
                'message': 'Could not extract text from the document'
            }), 400
        
        # Create vector store
        if not create_vector_store(documents):
            return jsonify({
                'success': False, 
                'message': 'Failed to process document for Q&A'
            }), 500
        
        # Initialize QA chain
        if not initialize_qa_chain():
            return jsonify({
                'success': False, 
                'message': 'Failed to initialize Q&A system'
            }), 500
        
        current_docs = documents
        
        return jsonify({
            'success': True,
            'message': f'Document uploaded successfully! Processed {len(documents)} pages.',
            'filename': filename,
            'pages': len(documents)
        })
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({
            'success': False, 
            'message': f'Error processing file: {str(e)}'
        }), 500
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
                print(f"Cleaned up temp file: {temp_filepath}")
            except:
                pass

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle Q&A requests"""
    global qa_chain
    
    if not qa_chain:
        return jsonify({
            'success': False, 
            'message': 'Please upload a document first'
        }), 400
    
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({
            'success': False, 
            'message': 'No question provided'
        }), 400
    
    question = data['question'].strip()
    if not question:
        return jsonify({
            'success': False, 
            'message': 'Question cannot be empty'
        }), 400
    
    print(f"Processing question: {question}")
    
    try:
        # Get answer from QA chain
        result = qa_chain({"query": question})
        
        answer = result.get('result', 'No answer found.')
        source_docs = result.get('source_documents', [])
        
        # Format response
        response_data = {
            'success': True,
            'answer': answer,
            'question': question
        }
        
        # Add source information if available
        if source_docs:
            sources = []
            for i, doc in enumerate(source_docs[:3]):  # Limit to top 3
                source_info = {
                    'content': doc.page_content[:300] + ('...' if len(doc.page_content) > 300 else ''),
                    'page': doc.metadata.get('page', 'N/A'),
                    'source': doc.metadata.get('source', 'Document')
                }
                sources.append(source_info)
            
            response_data['sources'] = sources
            response_data['source_count'] = len(sources)
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error answering question: {str(e)}")
        return jsonify({
            'success': False, 
            'message': f'Error processing question: {str(e)}'
        }), 500

@app.route('/clear', methods=['POST'])
def clear_documents():
    """Clear current documents and reset"""
    try:
        clear_vector_store()
        return jsonify({
            'success': True,
            'message': 'Documents cleared successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error clearing documents: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs("./chroma_db", exist_ok=True)
    
    # Print startup info
    print("=" * 50)
    print("KAI Cloud Document Q&A System")
    print("=" * 50)
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Chroma DB path: ./chroma_db")
    print("Server starting on http://localhost:5000")
    print("=" * 50)
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )