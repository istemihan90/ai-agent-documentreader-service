import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone 
import google.generativeai as genai
import base64
import httpx 
from flask_cors import CORS 
from google.cloud import storage # New: For GCS thumbnail retrieval
import io # New: For handling image bytes

load_dotenv()

app = Flask(__name__)
CORS(app) 

# Retrieve environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID") # New: For GCS client
GCP_PROCESSED_BUCKET_NAME = os.getenv("GCP_PROCESSED_BUCKET_NAME") # New: For GCS client

# Initialize OpenAI and Pinecone clients
openai_client = OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client(verify=False))
pinecone_client = Pinecone(api_key=PINECONE_API_KEY, verify_ssl=False)
pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME) 

# Initialize Google Cloud Storage client (AI Agent also needs to access GCS for thumbnails)
storage_client = storage.Client(project=GCP_PROJECT_ID)


# Configure Gemini API
if GOOGLE_GEMINI_API_KEY:
    genai.configure(api_key=GOOGLE_GEMINI_API_KEY)
else:
    print("GOOGLE_GEMINI_API_KEY ortam değişkeni bulunamadı. Gemini API işlevselliği kısıtlı olabilir.")

# Helper function to get embeddings (from OpenAI, for text queries)
def get_embedding(text):
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Helper function to download thumbnail from GCS and convert to base64
def get_thumbnail_from_gcs(file_name: str):
    """
    Belirtilen dosya adının küçük resmini GCS'den indirir ve base64 string olarak döndürür.
    """
    thumbnail_file_name = f"{os.path.splitext(file_name)[0]}_thumbnail.jpeg"
    try:
        bucket = storage_client.bucket(GCP_PROCESSED_BUCKET_NAME)
        blob = bucket.blob(thumbnail_file_name)
        if blob.exists():
            thumbnail_bytes = blob.download_as_bytes()
            return base64.b64encode(thumbnail_bytes).decode('utf-8')
        else:
            print(f"Uyarı: {thumbnail_file_name} GCS'de bulunamadı.")
            return None
    except Exception as e:
        print(f"GCS'den küçük resim indirme hatası: {e}")
        return None

# Helper function to perform RAG (Retrieval-Augmented Generation)
def retrieve_and_generate(query_text, image_data=None, document_name=None):
    """
    Verilen metin ve isteğe bağlı görsel/doküman sorgusuna göre Pinecone'dan ilgili belge parçalarını çeker
    ve LLM kullanarak bir yanıt üretir.
    """
    retrieved_chunks = []
    if query_text:
        query_embedding = get_embedding(query_text)
        query_results = pinecone_index.query(
            vector=query_embedding,
            top_k=5, 
            include_metadata=True
        )
        for match in query_results.matches:
            if 'chunk_text' in match.metadata:
                retrieved_chunks.append(match.metadata['chunk_text'])
    
    context = "\n".join(retrieved_chunks)
    
    # Check if we need to retrieve a thumbnail from GCS based on document_name
    thumbnail_base64 = None
    if document_name and not image_data: # Only fetch thumbnail if no direct image data provided
        thumbnail_base64 = get_thumbnail_from_gcs(document_name)
        if thumbnail_base64:
            print(f"'{document_name}' için GCS'den küçük resim başarıyla çekildi.")

    messages_parts = []
    if query_text:
        messages_parts.append(query_text)

    # Prioritize directly uploaded image, then GCS thumbnail
    if image_data:
        image_bytes = base64.b64decode(image_data)
        messages_parts.append({
            "mime_type": "image/jpeg", 
            "data": image_bytes
        })
    elif thumbnail_base64:
        thumbnail_bytes = base64.b64decode(thumbnail_base64)
        messages_parts.append({
            "mime_type": "image/jpeg", 
            "data": thumbnail_bytes
        })

    # Use Gemini for multimodal or if no context from Pinecone, otherwise OpenAI
    if messages_parts and any(isinstance(part, dict) and "mime_type" in part for part in messages_parts):
        # If there's an image (either direct upload or thumbnail), use Gemini Vision Pro
        # Add context to Gemini prompt if available
        if context:
            gemini_prompt = f"Aşağıdaki belgelerden ve görselden faydalanarak soruyu yanıtlayın. Eğer bilgi yoksa, 'Bilgi bulunamadı' diye belirtin.\n\nBelgeler:\n{context}\n\nSoru: {query_text}\nCevap:"
            # Replace query_text in messages_parts with the enriched prompt
            messages_parts[0] = gemini_prompt if query_text else "" # Ensure text part is first
        
        try:
            model = genai.GenerativeModel('gemini-pro-vision') 
            response = model.generate_content(messages_parts)
            return response.text
        except Exception as e:
            print(f"Gemini API veya görsel işleme hatası: {e}")
            return "Üzgünüm, görselle ilgili sorunuza yanıt veremiyorum veya görsel işlenirken bir hata oluştu."

    else:
        # For text-only queries (no images), use OpenAI
        if context:
            prompt = f"Aşağıdaki belgelerden ve bilgilerden faydalanarak soruyu yanıtlayın. Eğer bilgi yoksa, 'Bilgi bulunamadı' diye belirtin.\n\nBelgeler:\n{context}\n\nSoru: {query_text}\nCevap:"
        else:
            prompt = f"Soru: {query_text}\nCevap:"
        
        chat_completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": prompt}]
        )
        return chat_completion.choices[0].message.content

# Health check endpoint for the AI Agent
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "service": "ai-agent-documentreader"}), 200

# Main endpoint for AI Agent queries (other agents will use this)
@app.route("/query", methods=["POST"]) 
def query_document_endpoint():
    data = request.get_json()
    query_text = data.get("queryText")
    image_data = data.get("imageData") # Base64 encoded image string (optional from frontend)
    document_name = data.get("documentName") # New: Optional, to hint at which doc's thumbnail to use

    if not query_text and not image_data and not document_name:
        return jsonify({"error": "Sorgu metni, görsel verisi veya doküman adı gerekli."}), 400

    try:
        answer = retrieve_and_generate(query_text, image_data, document_name)
        return jsonify({"answer": answer}), 200
    except Exception as e:
        print(f"Sorgu işlenirken bir hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Sorgu işlenirken hata oluştu: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

