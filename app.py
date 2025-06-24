import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone # Corrected: Removed 'Index' as it's not a direct import
import google.generativeai as genai
import base64
import httpx # For verify=False with OpenAI and Pinecone clients

load_dotenv()

app = Flask(__name__)

# Retrieve environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

# Initialize OpenAI and Pinecone clients
# verify=False is included for potential SSL certificate issues in some environments.
openai_client = OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client(verify=False))
pinecone_client = Pinecone(api_key=PINECONE_API_KEY, verify_ssl=False)
# Initialize the Pinecone Index instance from the client
pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME) 

# Configure Gemini API
if GOOGLE_GEMINI_API_KEY:
    genai.configure(api_key=GOOGLE_GEMINI_API_KEY)
else:
    print("GOOGLE_GEMINI_API_KEY ortam değişkeni bulunamadı. Gemini API işlevselliği kısıtlı olabilir.")

# Helper function to get embeddings (from OpenAI, for text queries)
def get_embedding(text):
    """
    OpenAI'nin embedding modelini kullanarak verilen metin için embedding oluşturur.
    """
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Helper function to perform RAG (Retrieval-Augmented Generation)
def retrieve_and_generate(query_text, image_data=None):
    """
    Verilen metin ve isteğe bağlı görsel sorgusuna göre Pinecone'dan ilgili belge parçalarını çeker
    ve LLM kullanarak bir yanıt üretir.
    """
    retrieved_chunks = []
    if query_text:
        # Metin sorgusu için embedding alın
        query_embedding = get_embedding(query_text)
        # İlgili belge parçalarını Pinecone'dan sorgulayın
        query_results = pinecone_index.query(
            vector=query_embedding,
            top_k=5, # En alakalı 5 parçayı alın
            include_metadata=True
        )
        for match in query_results.matches:
            if 'chunk_text' in match.metadata:
                retrieved_chunks.append(match.metadata['chunk_text'])
    
    # Pinecone'dan çekilen bağlamı birleştirin
    context = "\n".join(retrieved_chunks)
    
    # Gemini/OpenAI için içeriği hazırlayın
    messages = []
    if image_data:
        # Multimodal (görsel + metin) sorgular için Gemini Vision modelini kullanın
        # Base64 kodlu görsel verisini decode edin
        try:
            image_bytes = base64.b64decode(image_data)
            image_part = {
                "mime_type": "image/jpeg", # Varsayılan olarak JPEG, dinamik olarak ayarlanabilir
                "data": image_bytes
            }
            # Eğer bir metin sorgusu da varsa, onu da ekleyin
            if query_text:
                messages.append({"role": "user", "parts": [query_text, image_part]})
            else:
                messages.append({"role": "user", "parts": [image_part]})
            
            # Gemini Vision Pro modelini kullanın
            model = genai.GenerativeModel('gemini-pro-vision') # Or 'gemini-1.0-pro-vision'
            response = model.generate_content(messages)
            return response.text
        except Exception as e:
            print(f"Gemini API veya görsel işleme hatası: {e}")
            return "Üzgünüm, görselle ilgili sorunuza yanıt veremiyorum veya görsel işlenirken bir hata oluştu."

    else:
        # Sadece metin sorguları için OpenAI veya Gemini Pro (metin-sadece) kullanın
        # Mevcut ise çekilen bağlamı önceliklendirin
        if context:
            prompt = f"Aşağıdaki belgelerden ve bilgilerden faydalanarak soruyu yanıtlayın. Eğer bilgi yoksa, 'Bilgi bulunamadı' diye belirtin.\n\nBelgeler:\n{context}\n\nSoru: {query_text}\nCevap:"
        else:
            prompt = f"Soru: {query_text}\nCevap:"
        
        # Tercihe/maliyete göre OpenAI ve Gemini-Pro arasında seçim yapabilirsiniz
        # Embedding modeliyle tutarlılık için burada OpenAI kullanılıyor
        chat_completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo", # Veya "gpt-4" tercih edilirse
            messages=[{"role": "user", "content": prompt}]
        )
        return chat_completion.choices[0].message.content

# Health check endpoint for the AI Agent
@app.route("/health", methods=["GET"])
def health_check():
    """
    AI Ajanı servisi için basit bir sağlık kontrol endpoint'i.
    """
    return jsonify({"status": "ok", "service": "ai-agent-documentreader"}), 200

# Main endpoint for AI Agent queries (other agents will use this)
@app.route("/query", methods=["POST"])
def query_document_endpoint():
    """
    Diğer AI ajanlarından veya kullanıcılardan gelen sorguları işler.
    Metin ve isteğe bağlı görsel verisi alarak belge tabanlı yanıtlar üretir.
    """
    data = request.get_json()
    query_text = data.get("queryText")
    image_data = data.get("imageData") # Base64 encoded image string (optional)

    if not query_text and not image_data:
        return jsonify({"error": "Sorgu metni veya görsel verisi gerekli."}), 400

    try:
        answer = retrieve_and_generate(query_text, image_data)
        # Diğer ajanların kolayca tüketebilmesi için cevabı döndür
        return jsonify({"answer": answer}), 200
    except Exception as e:
        print(f"Sorgu işlenirken bir hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Sorgu işlenirken hata oluştu: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

