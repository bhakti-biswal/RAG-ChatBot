import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY')
    SQLALCHEMY_DATABASE_URI = 'sqlite:///rag_chatbot.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Google OAuth
    GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
    GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET')
    GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"
    
    # Gemini API Key
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    
    # Vector database path
    VECTOR_DB_PATH = 'vector_store'
