from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import google.auth.transport.requests
import google.oauth2.credentials
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import requests
import os
import json
import google.generativeai as genai
import chromadb
from chromadb.config import Settings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import uuid

from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Simple Embedding Function (NO HUGGING FACE!)
class SimpleEmbeddingFunction:
    """A simple embedding function that doesn't require Hugging Face"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=384)  # 384 dimensions like MiniLM
        self.is_fitted = False
        
    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        if not self.is_fitted:
            # Fit on some sample text first
            sample_texts = [
                "machine learning", "artificial intelligence", "natural language processing",
                "computer science", "data analysis", "deep learning", "neural networks"
            ]
            self.vectorizer.fit(sample_texts)
            self.is_fitted = True
            
        # Convert to dense arrays and ensure correct shape
        embeddings = self.vectorizer.transform(texts).toarray()
        
        # Normalize to unit length (important for similarity search)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings = embeddings / norms
        
        return embeddings.tolist()

# Use our simple embedding function
embedding_function = SimpleEmbeddingFunction()

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=app.config['VECTOR_DB_PATH'])


# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    google_id = db.Column(db.String(100), unique=True)
    email = db.Column(db.String(100), unique=True)
    name = db.Column(db.String(100))
    documents = db.relationship('UserDocument', backref='user', lazy=True)

class UserDocument(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    document_id = db.Column(db.String(200))
    title = db.Column(db.String(200))
    content = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Google OAuth Setup
def get_google_flow():
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": app.config['GOOGLE_CLIENT_ID'],
                "client_secret": app.config['GOOGLE_CLIENT_SECRET'],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                # Static redirect URIs that must match Google Console
                "redirect_uris": [ "http://127.0.0.1:5000/callback"]
            }
        },
        scopes=[
            "openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
            "https://www.googleapis.com/auth/documents.readonly",
            "https://www.googleapis.com/auth/drive.readonly"
        ]
    )
    flow.redirect_uri = url_for('google_callback', _external=True)
    return flow

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('documents'))
    return render_template('index.html')

@app.route('/login')
def login():
    flow = get_google_flow()
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
    )
    session['state'] = state
    return redirect(authorization_url)


@app.route('/callback')
def google_callback():
    flow = get_google_flow()
    # Add redirect_uri to fetch_token as well
    flow.fetch_token(
        authorization_response=request.url
    )
    
    credentials = flow.credentials
    id_info = id_token.verify_oauth2_token(
        credentials.id_token, google.auth.transport.requests.Request(), 
        app.config['GOOGLE_CLIENT_ID']
    )
    
    # Find or create user
    user = User.query.filter_by(google_id=id_info['sub']).first()
    if not user:
        user = User(
            google_id=id_info['sub'],
            email=id_info['email'],
            name=id_info['name']
        )
        db.session.add(user)
        db.session.commit()
    
    login_user(user)
    return redirect(url_for('documents'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/debug')
def debug_info():
    return {
        'current_redirect_uri': url_for('google_callback', _external=True),
        'app_url': request.url_root
    }


@app.route('/documents')
@login_required
def documents():
    # Get Google Docs service
    flow = get_google_flow()
    credentials = flow.credentials
    
    drive_service = build('drive', 'v3', credentials=credentials)
    
    # List Google Docs
    results = drive_service.files().list(
        q="mimeType='application/vnd.google-apps.document'",
        pageSize=100,
        fields="files(id, name)"
    ).execute()
    
    docs = results.get('files', [])
    
    # Get user's selected documents
    user_docs = UserDocument.query.filter_by(user_id=current_user.id).all()
    selected_doc_ids = [doc.document_id for doc in user_docs]
    
    return render_template('documents.html', documents=docs, selected_doc_ids=selected_doc_ids)

@app.route('/add_document', methods=['POST'])
@login_required
def add_document():
    document_id = request.form.get('document_id')
    
    # Check if document already exists
    existing_doc = UserDocument.query.filter_by(
        user_id=current_user.id, document_id=document_id
    ).first()
    
    if not existing_doc:
        # Get document content
        flow = get_google_flow()
        credentials = flow.credentials
        
        docs_service = build('docs', 'v1', credentials=credentials)
        drive_service = build('drive', 'v3', credentials=credentials)
        
        # Get document title
        file_info = drive_service.files().get(fileId=document_id).execute()
        title = file_info.get('name', 'Untitled Document')
        
        # Get document content
        doc = docs_service.documents().get(documentId=document_id).execute()
        content = extract_text_from_doc(doc)
        
        # Save to database
        user_doc = UserDocument(
            user_id=current_user.id,
            document_id=document_id,
            title=title,
            content=content
        )
        db.session.add(user_doc)
        db.session.commit()
        
        # Add to vector store
        add_to_vector_store(user_doc, content)
    
    return redirect(url_for('documents'))

@app.route('/remove_document', methods=['POST'])
@login_required
def remove_document():
    document_id = request.form.get('document_id')
    
    # Remove from database
    user_doc = UserDocument.query.filter_by(
        user_id=current_user.id, document_id=document_id
    ).first()
    
    if user_doc:
        db.session.delete(user_doc)
        db.session.commit()
        
        # Remove from vector store
        remove_from_vector_store(user_doc)
    
    return redirect(url_for('documents'))

@app.route('/chat')
@login_required
def chat():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    user_message = request.json.get('message')
    
    # Retrieve relevant documents
    relevant_docs = retrieve_relevant_documents(user_message, current_user.id)
    
    # Generate response
    response = generate_response(user_message, relevant_docs)
    
    return jsonify({'response': response})

# Helper Functions
def extract_text_from_doc(doc):
    """Extract text from Google Doc structure"""
    content = []
    
    if 'body' in doc and 'content' in doc['body']:
        for element in doc['body']['content']:
            if 'paragraph' in element:
                for para_element in element['paragraph']['elements']:
                    if 'textRun' in para_element and 'content' in para_element['textRun']:
                        content.append(para_element['textRun']['content'])
    
    return '\n'.join(content)

def add_to_vector_store(user_doc, content):
    """Add document content to vector store"""
    collection_name = f"user_{user_doc.user_id}"
    
    try:
        collection = chroma_client.get_collection(collection_name)
    except:
        collection = chroma_client.create_collection(name=collection_name)
    
    # Split content into chunks
    chunks = split_text_into_chunks(content)
    
    # Generate embeddings using our simple function
    embeddings = embedding_function(chunks)
    
    # Add chunks to vector store
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{
            'document_id': user_doc.document_id,
            'title': user_doc.title,
            'chunk_index': i
        } for i in range(len(chunks))],
        ids=[f"{user_doc.document_id}_{i}" for i in range(len(chunks))]
    )

def remove_from_vector_store(user_doc):
    """Remove document from vector store"""
    collection_name = f"user_{user_doc.user_id}"
    
    try:
        collection = chroma_client.get_collection(collection_name)
        # Delete all chunks for this document
        collection.delete(where={'document_id': user_doc.document_id})
    except:
        pass

def split_text_into_chunks(text, chunk_size=500):
    """Split text into chunks of specified size"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

def retrieve_relevant_documents(query, user_id, top_k=3):
    """Retrieve relevant documents from vector store"""
    collection_name = f"user_{user_id}"
    
    try:
        collection = chroma_client.get_collection(collection_name)
        
        # Generate query embedding using our simple function
        query_embedding = embedding_function([query])
        
        # Query vector store
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        return results['documents'][0] if results['documents'] else []
    except:
        return []

def generate_response(query, relevant_docs):
    """Generate response using Google Gemini"""
    genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
    
    # Build context from relevant documents
    context = "\n\n".join(relevant_docs) if relevant_docs else ""
    
    if context:
        prompt = f"""Based on the following documents, answer the user's question. 
        If the information is not in the documents, say so but still provide a helpful response.

        Documents:
        {context}

        Question: {query}

        Answer:"""
    else:
        prompt = f"""Answer the user's question using your general knowledge.

        Question: {query}

        Answer:"""
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        answer = response.text
        
        if not relevant_docs and "not in the documents" not in answer.lower():
            answer = "I couldn't find this information in your documents, but based on general knowledge: " + answer
        
        return answer
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Initialize database
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
