import streamlit as st
import requests
import json
from typing import List, Dict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
import time
import warnings
import urllib3
from PIL import Image
import io

# Suppress SSL warnings (you should use proper SSL verification in production)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'api_url' not in st.session_state:
    st.session_state.api_url = "https://uat-ai.goodyear.com.cn/zhipuapi/blade-demo/chatSession/response/v3/sse"

if 'api_token' not in st.session_state:
    st.session_state.api_token = "bbcf83b55ffb445f9f4bf369ec2451ab"

if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

if 'ssl_verify' not in st.session_state:
    st.session_state.ssl_verify = False

# Goodyear logo URL
GOODYEAR_LOGO_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJQAAACUCAMAAABC4vDmAAAAflBMVEX///8oU5wkUZsAO5IhT5oYS5iSosaImL8AQpUAPpMMRpcVSZgAQJT8/P0AOJH4+fvg5O7Z3up/kLvByd3l6fEAMo/J0OHs7vQvV56otNDU2ec7XqFLaqdrgrQANZDy9PhWcquzvdZhe7CerMt5h7YAJ4sAHIhDZKQAIolkda2KhPLhAAAEbElEQVR4nO2W25KjOAyGbWNzsEMAm3AmHJKZbL//C64k0r21U53Zqp2Z7Rt9FwkGI/2WZBkhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhfsgyhNA0Vd22bTmEvv1qPUAey7QbtZYdIHW051+tCIjPrh9WpXR2jqKr9ctXCxJtHEazz52WegwD5LD4akVCbN2lUdJoZa393bVU/H29xfL5+qvOWVtOflIqvg2/WEOIkyTZPhxvfp59v/w13Nc5GZ7XCeE3HPXX8zwMiZHa1j3dxhof/D3Uxzzv/Q3NLnDh74eF5IPqB5I2Y11msvP4VNg5Y/FWQ8Oms06uzmUeBvlqjYHoGHN6Q/9zGMYkMWZMRJwZYy7wTmvTtegjGLkoii4JGplTGH6jVe9owXX4G70ObpxppW5DM2+HppOWph9Gpbsah6mWtheTllmAPJ6klGk/SCWjG4qstrAk6R7norzAI+NFsbusFbOBkdz6/obhaByOziVctmBJ6rlc8W96qSmFp/vHsLFSqk4UTkkbQypQxaUQWsGkQjQRPD0VBYjSE+S3Nnsow3f0VqA39Vju1kAAzvCa80+b9EhKCv2A+sxNPPAWLfsT6lRJaW8f4wmWaGZRgVUDXXCFoRpFe8G/WsQgWa/5ABrVo6ZWvmd1CLQ6XI7czbkHqygq3d7rI3P6fbylKHcoriize7FfKc7XD8UVJgESRd6nvMR1QcQ2dGhKgUGH4c3JQ1SenLV+hBQX1YBSeIni01u83Ocdy6jWp9vzRXGs0j21zZ9rykewpHQukvl+h6q6obXs8A4BC+7Q6ClgbfuA2WYT3qFkcLgHCIlOI4xBMZIovE3zpbKnDn0kdmzJHsqNcM4j37GkbPkie2jJJSI/aa2jwztUy/I4ktqj8a4qVkUaKRhdvYyaXhLNH/3sZOrDGyaQKueKGVlQg5o8NY7hqobDLgQGywKuruTmhSZRd+i9p/2hupJehhXh9pKqhG5BGssOs7hRVtRDlLjcS4sBsYNL+2TF9C+4Pk2xqRQu9VlSqxl7Sh8U4zOv+INeX0CWukrEmBAo7C2j/OC7uOdbSxoDqHCwf++GIoQ5xr1RKmXiOMwxmQoYRqobETKs66P79hb7laPMtuJO4gaMl3lRUcAMDsY6n585zy10qQbLP71jd/QpqtiNNKA8l5TUalQqneHpdgJH9fRcMQXB0kb0tO/xmKjyzq3QzXfqF3U7KfLjFG3q16HKbHqNMjhQsR2Lakz1+LA2pU4sCn929uJMtkO22u8wy16/pZGkgExam2Y9Gr/Id7RxdJ4U51ns562/UIcc8KGrhjPeh50S4YTXx8yyxQeJp2OgjUdnx+T9BMgHP+pxpi+SMnlODeS6PGnV7OE5sbgBMUUth4PvmHgr4uOEa8lDXdH9RQz0/6Oz7z+yTM40+6/+NvhJ1mxq1vDv8/5PNhtvb78hAT/F0vj467/G/0n/qiEzDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMgfwLiFEhpRVxfrgAAAABJRU5ErkJggg=="

def set_page_config():
    """Set page configuration with Goodyear branding."""
    try:
        st.set_page_config(
            page_title="Sales Chat Assistant",
            page_icon=GOODYEAR_LOGO_URL ,  # Racing flag emoji as fallback
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception as e:
        st.error(f"Error setting page config: {e}")

def apply_goodyear_styling():
    """Apply Goodyear branding colors to the app."""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
        font-family: 'Roboto', sans-serif;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1y4p8pa {
        background: linear-gradient(180deg, #1a365d 0%, #2c5282 100%);
    }
    
    /* Header styling */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1rem 0;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    
    .logo-img {
        max-height: 60px;
        margin-right: 20px;
        filter: brightness(1.2);
    }
    
    /* Main title styling */
    .main-title {
        color: #d4af37;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        margin: 0;
        font-family: 'Roboto', sans-serif;
    }
    
    /* Subtitle styling */
    .subtitle {
        color: #ffffff;
        text-align: center;
        font-size: 1.2rem;
        font-style: italic;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Chat messages styling */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.95);
        border-left: 4px solid #d4af37;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    /* User message styling */
    .stChatMessage[data-testid="chat-message-user"] {
        background: linear-gradient(135deg, #2c5282 0%, #3182ce 100%);
        color: white;
        border-left: 4px solid #d4af37;
    }
    
    /* Assistant message styling */
    .stChatMessage[data-testid="chat-message-assistant"] {
        background: rgba(255, 255, 255, 0.98);
        border-left: 4px solid #1e3a5f;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.95);
        border: 2px solid #d4af37;
        border-radius: 10px;
        color: #1e3a5f;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #d4af37 0%, #b8941f 100%);
        color: #1e3a5f;
        border: none;
        border-radius: 10px;
        font-weight: bold;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #b8941f 0%, #9a7a1a 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.3);
    }
    
    /* Sidebar styling */
    .sidebar-header {
        color: #d4af37;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Error and success messages */
    .stSuccess {
        background: rgba(72, 187, 120, 0.1);
        border-left: 4px solid #48bb78;
        color: #ffffff;
    }
    
    .stError {
        background: rgba(245, 101, 101, 0.1);
        border-left: 4px solid #f56565;
        color: #ffffff;
    }
    
    .stWarning {
        background: rgba(237, 137, 54, 0.1);
        border-left: 4px solid #ed8936;
        color: #ffffff;
    }
    
    /* Footer styling */
    .footer {
        color: rgba(255, 255, 255, 0.7);
        text-align: center;
        font-size: 0.9rem;
        margin-top: 2rem;
        padding: 1rem;
        background: rgba(0, 0, 0, 0.1);
        border-radius: 10px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

def display_header():
    """Display the application header with Goodyear branding."""
    try:
        # Create header with logo and title
        st.markdown(f"""
        <div class="header-container">
            <img src="{GOODYEAR_LOGO_URL}" alt="Goodyear Logo">
            <div>
                <h1 class="main-title">Sales Chat Assistant</h1>
                <p class="subtitle">Your Expert Tire Knowledge Assistant</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        # Fallback header without logo
        st.markdown("""
        <div class="header-container">
            <div>
                <h1 class="main-title"> Sales Chat Assistant</h1>
                <p class="subtitle">Your Expert Tire Knowledge Assistant</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def test_api_connection():
    """Test API connectivity and return status."""
    try:
        response = requests.get(
            st.session_state.api_url.replace('/v3/sse', '/health') if '/v3/sse' in st.session_state.api_url else st.session_state.api_url,
            timeout=10,
            verify=st.session_state.ssl_verify,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        return True, f"Connection successful (Status: {response.status_code})"
    except requests.exceptions.SSLError as e:
        return False, f"SSL Error: {str(e)[:100]}... Try disabling SSL verification."
    except requests.exceptions.ConnectionError:
        return False, "Connection failed: Unable to reach the server. Check URL and network."
    except requests.exceptions.Timeout:
        return False, "Connection timeout: Server took too long to respond."
    except Exception as e:
        return False, f"Connection error: {str(e)[:100]}"

def create_session_with_retries():
    """Create a session with retry strategy for robust API calls."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
def create_enhanced_prompt(message: str, chat_history: List[Dict]) -> str:
    """Create an enhanced prompt that ensures comprehensive tire knowledge retrieval"""
    
    # Enhanced context with explicit instructions for knowledge retrieval
    context = """You are an expert tire consultant with comprehensive knowledge about all aspects of tires. 
You have access to detailed technical information about:
- Tire construction and materials (rubber compounds, steel belts, fabric layers, sidewall construction)
- Manufacturing processes and components
- Load indices and speed ratings (complete tables and specifications)
- Tire nomenclature and sizing systems
- Performance characteristics and applications
- Safety standards and regulations
- Maintenance and care instructions

IMPORTANT: Always provide detailed, specific answers using your complete knowledge base. 
For questions about tire construction or materials, explain the components in detail.
For technical specifications, provide exact numbers and explanations.
Answer naturally and conversationally while being comprehensive and accurate."""

    # Add question-specific guidance
    message_lower = message.lower()
    
    if any(word in message_lower for word in ['made of', 'materials', 'construction', 'components', 'rubber', 'steel']):
        context += """\n\nFor this tire construction/materials question, provide detailed information about:
- Rubber compounds and their properties
- Steel belt construction and purpose  
- Fabric layers (polyester, nylon, etc.)
- Sidewall materials and construction
- Tread compound specifications
- Manufacturing processes involved"""
    
    elif any(word in message_lower for word in ['load index', 'load rating', 'weight capacity']):
        context += "\n\nFor load index questions, provide specific weight capacities and explain the complete load index system."
    
    elif any(word in message_lower for word in ['speed rating', 'speed symbol', 'maximum speed']):
        context += "\n\nFor speed rating questions, provide specific speed limits and explain the complete speed rating system."
    
    # Build conversation context
    recent_context = ""
    if chat_history:
        recent_messages = chat_history[-2:] if len(chat_history) > 2 else chat_history
        recent_context = f"\n\nRecent conversation context: {json.dumps(recent_messages, ensure_ascii=False)}"

def identify_question_type(question: str) -> str:
    """Identify the type of tire question being asked."""
    if not question:
        return 'standard'
    
    question_lower = question.lower()
    
    # Check for reverse lookup patterns first
    if ('load index' in question_lower and 
        any(phrase in question_lower for phrase in ['holds', 'up to', 'weight']) and
        any(phrase in question_lower for phrase in ['what is', 'which', 'what load'])):
        return 'reverse_load_lookup'
    
    if ('speed' in question_lower and 
        any(phrase in question_lower for phrase in ['symbol', 'rating']) and
        any(phrase in question_lower for phrase in ['up to', 'maximum']) and
        any(phrase in question_lower for phrase in ['what is', 'which'])):
        return 'reverse_speed_lookup'
    
    return 'standard'

def clean_and_normalize_text(text: str) -> str:
    """Enhanced text cleaning to handle encoding issues and artifacts."""
    if not text:
        return ""
    
    # Handle various encoding issues
    text = text.replace("Â", "").replace("â", "").replace("ââ", "")
    text = text.replace("\\n", "\n").replace("\\t", " ").replace("\\r", "")
    
    # Fix common encoding artifacts
    encoding_fixes = {
        "â€™": "'", "â€œ": '"', "â€": '"', "â€¢": "•",
        "â€": "-", "â€": "–", "Ã¡": "á", "Ã©": "é",
        "Ã­": "í", "Ã³": "ó", "Ãº": "ú", "Ã±": "ñ"
    }
    
    for old, new in encoding_fixes.items():
        text = text.replace(old, new)
    
    # Remove multiple consecutive special characters
    text = re.sub(r'[â]{2,}', '', text)
    text = re.sub(r'[Â]{2,}', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text.strip()

def extract_precise_answer(full_text: str, question: str) -> str:
    """Extract precise, direct answer to the specific question asked."""
    if not full_text or not question:
        return full_text or "No response received."
    
    # Clean the text first
    clean_text = clean_and_normalize_text(full_text)
    question_lower = question.lower()
    
    # Split into sentences for analysis
    sentences = re.split(r'[.!?]+', clean_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if not sentences:
        return clean_text
    
    # Extract specific numbers/symbols from the question
    question_numbers = re.findall(r'\b\d+\b', question)
    question_symbols = re.findall(r'\b[A-Z]\b', question.upper())
    
    # Find the most relevant sentence
    question_keywords = set(re.findall(r'\b\w{3,}\b', question_lower))
    
    # Add the specific numbers and symbols as high-priority keywords
    for num in question_numbers:
        question_keywords.add(num)
    for sym in question_symbols:
        question_keywords.add(sym.lower())
    
    best_sentence = ""
    best_score = 0
    
    for sentence in sentences[:5]:  # Check first 5 sentences
        sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
        
        # Check for exact matches of numbers and symbols from question
        exact_matches = 0
        for num in question_numbers:
            if num in sentence:
                exact_matches += 3
        for sym in question_symbols:
            if sym in sentence.upper():
                exact_matches += 3
        
        # Regular keyword overlap
        overlap = len(question_keywords.intersection(sentence_words))
        
        # Bonus for having numbers (often important for technical questions)
        number_bonus = len(re.findall(r'\d+', sentence))
        
        total_score = exact_matches + overlap + number_bonus
        
        if total_score > best_score:
            best_score = total_score
            best_sentence = sentence
    
    if best_sentence and best_score > 0:
        return best_sentence.strip() + '.'
    
    # Final fallback
    return sentences[0].strip() + '.' if sentences else clean_text

def parse_sse_stream(response, user_question=None):
    """Parse SSE stream with improved content extraction and error handling."""
    result_data = ''
    all_content_chunks = []
    
    try:
        for line in response.iter_lines(decode_unicode=True):
            if line is None:
                continue
                
            line = line.strip()
            
            if not line or line.startswith(("id:", "event:", "retry:")):
                continue

            # Extract JSON data
            json_str = ""
            if line.startswith("data: "):
                json_str = line[6:].strip()
            elif line.startswith("data:"):
                json_str = line[5:].strip()
            else:
                json_str = line.strip()

            if json_str == "[DONE]":
                break

            # Parse JSON and extract content
            try:
                json_data = json.loads(json_str)
                
                # Extract content using multiple strategies
                content = None
                content_fields = ["resultSetString", "content", "message", "text", "data", "answer", "response", "result"]
                
                for field in content_fields:
                    if field in json_data and json_data[field]:
                        content = str(json_data[field])
                        break
                
                # Check nested structures
                if not content:
                    for key, value in json_data.items():
                        if isinstance(value, dict):
                            for sub_key in content_fields:
                                if sub_key in value and value[sub_key]:
                                    content = str(value[sub_key])
                                    break
                        if content:
                            break

                if content:
                    all_content_chunks.append(content)

            except json.JSONDecodeError:
                # Handle non-JSON responses
                if json_str and len(json_str) > 5:
                    if not any(prefix in json_str.lower() for prefix in ['error', 'event:', 'id:', 'retry:']):
                        all_content_chunks.append(json_str)

        # Combine all content chunks
        result_data = ''.join(all_content_chunks)
        
        # Apply extraction if we have content and a question
        if result_data and user_question:
            result_data = extract_precise_answer(result_data, user_question)
        else:
            result_data = clean_and_normalize_text(result_data)
        
        return result_data if result_data else "⚠️ No relevant content found in the response."

    except Exception as e:
        return f"💥 Error parsing response: {str(e)}"

def create_targeted_prompt(message: str, chat_history: List[Dict]) -> str:
    """Create a highly targeted prompt for precise answers."""
    message_lower = message.lower()
    question_type = identify_question_type(message)
    
    # Extract specific numbers and symbols from the question
    question_numbers = re.findall(r'\b\d+\b', message)
    
    # Handle specific question types with targeted prompts
    if 'load index' in message_lower and question_numbers:
        target_index = question_numbers[0]
        return f"What is the exact weight capacity in kg for load index {target_index}? Answer format: 'Load index {target_index} can hold [WEIGHT]kg per tire.'"
    
    elif 'speed' in message_lower and 'symbol' in message_lower:
        return f"Question: {message}\n\nProvide the specific speed rating and maximum speed in km/h. Be precise with the symbol and speed."
    
    elif any(word in message_lower for word in ['195', '55', 'r16', 'nomenclature', 'means']):
        return f"Question: {message}\n\nExplain what each part of the tire size designation represents. Be specific about what each number means."
    
    elif 'components' in message_lower and 'tire' in message_lower:
        return f"Question: {message}\n\nList the main tire components and what each is made of. Be specific about materials."
    
    else:
        return f"Question: {message}\n\nAnswer with specific facts, numbers, and measurements. Be precise and direct."

def send_to_api(message: str, chat_history: List[Dict]) -> str:
    """Send message to API and get response with improved error handling."""
    try:
        enhanced_message = create_targeted_prompt(message, chat_history)
        
        # Build conversation context
        recent_context = chat_history[-2:] if len(chat_history) > 2 else chat_history
        chat_messages = recent_context + [{"role": "user", "content": enhanced_message}]
        
        payload = {
            "chatList": chat_messages,
            "chatType": "RAG",
            "dynamicPrompt": "You are a tire expert. Provide precise, accurate answers with specific numbers and facts. Focus on the exact question asked.",
            "knowledgeFlag": False,
            "knowledgeMetas": [{"knowledgeId": "8b28ee55aab1494fad250df98271e909", "fileIds": ["5025"]}],
            "questioner": "1914564261695836161",
            "token": st.session_state.api_token,
            "temperature": 0.7,
            "maxTokens": 1024
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json, */*",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
        
        session = create_session_with_retries()
        
        response = session.post(
            st.session_state.api_url,
            json=payload,
            headers=headers,
            timeout=(30, 120),
            verify=st.session_state.ssl_verify,
            stream=True,
            allow_redirects=True
        )

        if response.status_code == 401:
            return "❌ Authentication failed: Invalid API token."
        elif response.status_code == 403:
            return "❌ Access forbidden: Permission denied."
        elif response.status_code == 404:
            return "❌ API endpoint not found: Please verify the API URL."
        elif response.status_code != 200:
            return f"❌ API error {response.status_code}: {response.text[:200]}"

        return parse_sse_stream(response, message)

    except requests.exceptions.Timeout:
        return "⏰ Request timed out. The server is taking too long to respond."
    except requests.exceptions.ConnectionError as e:
        if "Name or service not known" in str(e):
            return "🔌 DNS Error: Cannot resolve server address. Check API URL."
        elif "Connection refused" in str(e):
            return "🔌 Connection refused: Server not accepting connections."
        else:
            return "🔌 Connection error: Unable to reach API server."
    except requests.exceptions.SSLError as e:
        return f"🔒 SSL Error: {str(e)[:200]}. Try disabling SSL verification."
    except Exception as e:
        return f"💥 Unexpected error: {str(e)[:200]}"



def main():
    """Main application function."""
    # Set page configuration
    set_page_config()
    
    # Apply styling
    apply_goodyear_styling()
    
    # Display header
    display_header()
    
    # Display sidebar
    display_sidebar()
    
    # Validate settings
    if not validate_settings():
        st.warning("⚠️ Please configure API settings in the sidebar before chatting.")
        return
    
    # Main chat interface
    st.markdown("### 💬 Chat Interface")
    
    # Display chat messages
    for i, msg in enumerate(st.session_state.messages):
        role_icon = "🧑" if msg["role"] == "user" else "🤖"
        role_name = "You" if msg["role"] == "user" else "Assistant"
        
        with st.chat_message(msg["role"]):
            st.markdown(f"**{role_icon} {role_name}:** {msg['content']}")

    # Chat input
    if prompt := st.chat_input("💬 Ask a specific tire question..."):
        # Validate input
        if not prompt.strip():
            st.warning("⚠️ Please enter a question.")
            return
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(f"**🧑 You:** {prompt}")
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching tire knowledge base..."):
                # Get response from API
                response = send_to_api(prompt, st.session_state.messages[:-1])
                
                # Display response
                if any(error in response for error in ["No response", "API error", "Connection error", "Error:", "timed out", "SSL Error"]):
                    st.error(f"❌ {response}")
                    
                    # Provide fallback suggestions
                    st.markdown("**💡 Suggestions:**")
                    st.markdown("- Try rephrasing your question")
                    st.markdown("- Check your internet connection")  
                    st.markdown("- Try one of the example questions")
                    if "SSL Error" in response:
                        st.markdown("- Try disabling SSL verification in settings")
                    
                else:
                    # Success - display the answer
                    st.markdown(f"**🤖 Assistant:** {response}")
                    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div class='footer'>"
        "🏁 Powered by Goodyear AI Technology | Built with Streamlit"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
