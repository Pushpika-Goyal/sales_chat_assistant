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
icon_image = Image.open(r"C:\Users\ZB06039\OneDrive - Goodyear\Pictures\goodyear.png")

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

# Goodyear Color Scheme
def apply_goodyear_styling():
    """Apply Goodyear branding colors to the app."""
    st.markdown("""
    <style>
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a365d 0%, #2c5282 100%);
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
    
    /* Title styling */
    .main-title {
        color: #d4af37;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        color: #ffffff;
        text-align: center;
        font-size: 1.2rem;
        font-style: italic;
        margin-bottom: 2rem;
        opacity: 0.9;
    }
    
    /* Sidebar header styling */
    .sidebar-header {
        color: #d4af37;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
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
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(212, 175, 55, 0.1);
        border-radius: 5px;
        color: #d4af37;
        font-weight: bold;
    }
    
    /* Metrics styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(212, 175, 55, 0.3);
    }
    
    /* Success message styling */
    .stSuccess {
        background: rgba(72, 187, 120, 0.1);
        border-left: 4px solid #48bb78;
        color: #ffffff;
    }
    
    /* Error message styling */
    .stError {
        background: rgba(245, 101, 101, 0.1);
        border-left: 4px solid #f56565;
        color: #ffffff;
    }
    
    /* Warning message styling */
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
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #d4af37 !important;
    }
    
    /* Sidebar logo container */
    .sidebar-logo {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        margin-bottom: 2rem;
        border: 2px solid rgba(212, 175, 55, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

def test_api_connection():
    """Test API connectivity and return status."""
    try:
        # Simple connectivity test
        test_url = st.session_state.api_url.replace('/v3/sse', '/health') if '/v3/sse' in st.session_state.api_url else st.session_state.api_url
        
        response = requests.get(
            test_url, 
            timeout=10,
            verify=st.session_state.ssl_verify,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
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
        total=6,  # Reduced retries for faster failure detection
        backoff_factor=2.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def identify_question_type(question: str) -> str:
    """Identify the type of tire question being asked."""
    question_lower = question.lower()
    
    # Check for reverse lookup patterns first (more specific)
    if ('load index' in question_lower and 
        any(phrase in question_lower for phrase in ['holds', 'up to', 'weight']) and
        any(phrase in question_lower for phrase in ['what is', 'which', 'what load'])):
        return 'reverse_load_lookup'
    
    if ('speed' in question_lower and 
        any(phrase in question_lower for phrase in ['symbol', 'rating']) and
        any(phrase in question_lower for phrase in ['up to', 'maximum']) and
        any(phrase in question_lower for phrase in ['what is', 'which'])):
        return 'reverse_speed_lookup'

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
    
    # Handle escape sequences safely
    try:
        # Only decode if it looks like it has escape sequences
        if '\\' in text and any(seq in text for seq in ['\\n', '\\t', '\\r']):
            text = text.encode().decode('unicode_escape')
    except:
        pass
    
    return text.strip()

def detect_tricky_patterns(question: str) -> dict:
    """Detect tricky question patterns and extract key information."""
    question_lower = question.lower()
    
    patterns = {
        'reverse_load': {
            'pattern': r'(?:what|which).*?load index.*?(?:holds?|carry|support).*?(\d+)\s*kg',
            'type': 'reverse_load_lookup'
        },
        'reverse_speed': {
            'pattern': r'(?:what|which).*?(?:speed|symbol|rating).*?(?:up to|maximum).*?(\d+)\s*(?:km/h|mph)',
            'type': 'reverse_speed_lookup'
        },
        'comparison': {
            'pattern': r'(?:difference|compare|vs|versus).*?(?:between|and)',
            'type': 'comparison_question'
        },
        'range': {
            'pattern': r'(?:between|from).*?(\d+).*?(?:to|and).*?(\d+)',
            'type': 'range_question'
        }
    }
    
    for pattern_name, pattern_info in patterns.items():
        match = re.search(pattern_info['pattern'], question_lower)
        if match:
            return {
                'type': pattern_info['type'],
                'matches': match.groups(),
                'pattern': pattern_name
            }
    
    return {'type': 'standard', 'matches': [], 'pattern': None}

def extract_nomenclature_answer(full_text: str, question: str) -> str:
    """Extract specific answers for tire nomenclature questions."""
    clean_text = clean_and_normalize_text(full_text)
    question_lower = question.lower()
    
    # Check if asking about specific part of nomenclature
    if 'symbol r' in question_lower or 'r indicates' in question_lower:
        # Look for radial construction explanation
        patterns = [
            r'r.*?(?:indicates|means|represents).*?(radial.*?construction)',
            r'(radial.*?construction).*?r',
            r'r.*?(radial.*?tire)',
            r'symbol r.*?(radial)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, clean_text.lower())
            if match:
                return f"The symbol R indicates {match.group(1)}."
    
    elif any(num in question for num in ['195', '55', '16']) and 'indicates' in question_lower:
        # Extract specific number explanations
        sentences = re.split(r'[.!?]+', clean_text)
        
        if '195' in question:
            for sentence in sentences:
                if '195' in sentence and any(word in sentence.lower() for word in ['width', 'section', 'millimeter', 'mm']):
                    return f"195 indicates the nominal section width of the tire in millimeters."
        
        elif '55' in question:
            for sentence in sentences:
                if '55' in sentence and any(word in sentence.lower() for word in ['aspect', 'ratio', 'height', 'sidewall']):
                    return f"55 indicates the aspect ratio - the sidewall height is 55% of the tire width."
        
        elif '16' in question:
            for sentence in sentences:
                if '16' in sentence and any(word in sentence.lower() for word in ['rim', 'wheel', 'diameter', 'inch']):
                    return f"16 indicates the rim diameter in inches."
    
    # General nomenclature explanation
    elif 'nomenclature' in question_lower or 'what does' in question_lower:
        # Look for complete explanations
        for sentence in re.split(r'[.!?]+', clean_text):
            sentence_clean = sentence.strip()
            if len(sentence_clean) > 50 and any(word in sentence_clean.lower() for word in ['width', 'ratio', 'diameter', 'construction']):
                return sentence_clean + '.'
    
    return clean_text

def extract_precise_answer(full_text: str, question: str) -> str:
    """Extract precise, direct answer to the specific question asked."""
    if not full_text or not question:
        return full_text
    
    # Clean the text first
    clean_text = clean_and_normalize_text(full_text)
    question_lower = question.lower()
    question_type = identify_question_type(question)
    
    # Split into sentences for analysis
    sentences = re.split(r'[.!?]+', clean_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if not sentences:
        return clean_text
    
    # Extract specific numbers/symbols from the question
    question_numbers = re.findall(r'\b\d+\b', question)
    question_symbols = re.findall(r'\b[A-Z]\b', question.upper())
    
    # Handle reverse load index lookup
    if question_type == 'reverse_load_lookup':
        # Extract target weight from question
        weight_matches = re.findall(r'(\d+)\s*kg', question_lower)
        if weight_matches:
            target_weight = weight_matches[0]
            
            # Look for sentences that mention this weight and an index
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if f'{target_weight}kg' in sentence_lower or f'{target_weight} kg' in sentence_lower:
                    # Extract load index from this sentence
                    index_match = re.search(r'(?:load index|index)\s*(\d+)', sentence_lower)
                    if index_match:
                        found_index = index_match.group(1)
                        return f"Load index {found_index} can hold {target_weight}kg per tire."
            
            # Alternative: look for pattern "XX kg corresponds to index YY"
            for sentence in sentences:
                if target_weight in sentence and any(word in sentence.lower() for word in ['index', 'corresponds', 'rated']):
                    index_match = re.search(r'(?:index|rated)\s*(\d+)', sentence.lower())
                    if index_match:
                        return f"Load index {index_match.group(1)} can hold {target_weight}kg per tire."
    
    # Handle reverse speed rating lookup
    elif question_type == 'reverse_speed_lookup':
        # Extract target speed from question
        speed_matches = re.findall(r'(\d+)\s*(?:km/h|mph)', question_lower)
        if speed_matches:
            target_speed = speed_matches[0]
            
            # Look for sentences that mention this speed and a symbol
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if f'{target_speed}km/h' in sentence_lower or f'{target_speed} km/h' in sentence_lower:
                    # Extract speed symbol from this sentence
                    symbol_match = re.search(r'(?:symbol|rating)\s*([a-z])', sentence_lower)
                    if symbol_match:
                        found_symbol = symbol_match.group(1).upper()
                        return f"Speed symbol {found_symbol} indicates maximum speed of {target_speed} km/h."
    
    # Load index specific questions (e.g., "what is load index 87?")
    elif 'load index' in question_lower and question_numbers:
        target_index = question_numbers[0]  # Get the specific index asked about
        
        # Look for sentences that mention the specific load index
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if f'load index {target_index}' in sentence_lower or f'index {target_index}' in sentence_lower:
                # Extract weight information for this specific index
                weight_match = re.search(rf'(?:load index {target_index}|index {target_index}).*?(\d+)\s*kg', sentence_lower)
                if weight_match:
                    return f"Load index {target_index} can hold {weight_match.group(1)}kg per tire."
                else:
                    return sentence.strip() + '.'
        
        # Alternative: look for pattern "index XX holds YY kg"
        for sentence in sentences:
            if target_index in sentence and 'kg' in sentence.lower():
                # Check if this sentence is about the specific index
                index_match = re.search(rf'\b{target_index}\b.*?(\d+)\s*kg', sentence)
                if index_match:
                    return f"Load index {target_index} can hold {index_match.group(1)}kg per tire."
    
    # Speed rating questions (e.g., "what does speed symbol V mean?")
    elif 'speed' in question_lower and 'symbol' in question_lower and question_symbols:
        target_symbol = question_symbols[0]  # Get the specific symbol asked about
        
        # Look for sentences that mention the specific speed symbol
        for sentence in sentences:
            sentence_upper = sentence.upper()
            sentence_lower = sentence.lower()
            
            if f'symbol {target_symbol}' in sentence_upper or f'rating {target_symbol}' in sentence_upper:
                # Extract speed information for this specific symbol
                speed_match = re.search(r'(\d+)\s*(?:km/h|mph)', sentence_lower)
                if speed_match:
                    return f"Speed symbol {target_symbol} indicates maximum speed of {speed_match.group(1)} km/h."
                else:
                    return sentence.strip() + '.'
        
        # Alternative: look for pattern "symbol X indicates YY km/h"
        for sentence in sentences:
            if target_symbol in sentence.upper() and any(unit in sentence.lower() for unit in ['km/h', 'mph']):
                speed_match = re.search(r'(\d+)\s*(?:km/h|mph)', sentence.lower())
                if speed_match:
                    return f"Speed symbol {target_symbol} indicates maximum speed of {speed_match.group(1)} km/h."
    
    # Tire components questions
    elif any(word in question_lower for word in ['components', 'parts', 'made of']) and 'tire' in question_lower:
        component_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(component in sentence_lower for component in ['tread', 'belts', 'sidewall', 'bead']) and \
               any(material in sentence_lower for material in ['rubber', 'steel', 'fabric', 'made of']):
                component_sentences.append(sentence.strip())
                if len(component_sentences) >= 3:
                    break
        
        if component_sentences:
            return '. '.join(component_sentences) + '.'
    
    # Nomenclature questions (tire size meaning)
    elif any(word in question_lower for word in ['195', '55', 'r16', 'nomenclature', 'means', 'represents', 'indicates']):
        return extract_nomenclature_answer(full_text, question)
    
    # Tread depth questions
    elif 'tread depth' in question_lower or ('depth' in question_lower and 'tire' in question_lower):
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if 'mm' in sentence_lower and any(word in sentence_lower for word in ['depth', 'tread', 'change', 'replace']):
                return sentence.strip() + '.'
    
    # If no specific pattern matches, find the most relevant sentence
    question_keywords = set(re.findall(r'\b\w{3,}\b', question_lower))
    
    # Add the specific numbers and symbols as high-priority keywords
    for num in question_numbers:
        question_keywords.add(num)
    for sym in question_symbols:
        question_keywords.add(sym.lower())
    
    best_sentence = ""
    best_score = 0
    
    for sentence in sentences[:8]:  # Check first 8 sentences
        sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
        
        # Check for exact matches of numbers and symbols from question
        exact_matches = 0
        for num in question_numbers:
            if num in sentence:
                exact_matches += 5  # High weight for exact number matches
        for sym in question_symbols:
            if sym in sentence.upper():
                exact_matches += 5  # High weight for exact symbol matches
        
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

def post_process_response(response: str, question: str) -> str:
    """Final cleanup and formatting of responses."""
    if not response:
        return "I couldn't find a specific answer to your question."
    
    # Clean encoding issues
    response = clean_and_normalize_text(response)
    
    # Remove redundant phrases
    redundant_phrases = [
        "Based on the provided context,",
        "According to the information,",
        "The answer is:",
        "In summary,",
        "To answer your question,"
    ]
    
    for phrase in redundant_phrases:
        response = response.replace(phrase, "").strip()
    
    # Ensure proper sentence structure
    if response and not response.endswith('.'):
        response += '.'
    
    # Capitalize first letter
    if response:
        response = response[0].upper() + response[1:] if len(response) > 1 else response.upper()
    
    return response

def parse_sse_stream(response, user_question=None):
    """Parse SSE stream with improved content extraction and error handling."""
    result_data = ''
    debug_info = []
    all_content_chunks = []
    
    try:
        for line in response.iter_lines(decode_unicode=True):
            if line is None:
                continue
                
            line = line.strip()
            
            if not line or line.startswith(("id:", "event:", "retry:")):
                continue

            if st.session_state.debug_mode:
                debug_info.append(f"📥 Processing line: {line[:100]}...")

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
                
                # Primary content fields
                content_fields = ["resultSetString", "content", "message", "text", "data", "answer", "response", "result"]
                
                for field in content_fields:
                    if field in json_data and json_data[field]:
                        content = str(json_data[field])
                        if st.session_state.debug_mode:
                            debug_info.append(f"✅ Found content in field: {field}")
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

                # Handle streaming format (OpenAI-style)
                if not content and 'choices' in json_data:
                    for choice in json_data['choices']:
                        delta = choice.get('delta') or choice.get('message')
                        if delta and 'content' in delta and delta['content']:
                            content = str(delta['content'])
                            break

                if content:
                    all_content_chunks.append(content)

            except json.JSONDecodeError as e:
                if st.session_state.debug_mode:
                    debug_info.append(f"⚠️ JSON decode error: {str(e)}")
                # Handle non-JSON responses
                if json_str and len(json_str) > 5:
                    if not any(prefix in json_str.lower() for prefix in ['error', 'event:', 'id:', 'retry:']):
                        all_content_chunks.append(json_str)

        # Combine all content chunks
        result_data = ''.join(all_content_chunks)
        
        # Apply specialized extraction based on question type
        if result_data and user_question:
            question_lower = user_question.lower()
            
            if any(word in question_lower for word in ['nomenclature', 'indicates', 'symbol r', '195', '55', '16']):
                result_data = extract_nomenclature_answer(result_data, user_question)
            else:
                result_data = extract_precise_answer(result_data, user_question)
            
            # Final post-processing
            result_data = post_process_response(result_data, user_question)
        else:
            result_data = clean_and_normalize_text(result_data)
        
        # Debug information
        if st.session_state.debug_mode:
            with st.expander("🔍 Debug Information"):
                st.write(f"📊 Content chunks found: {len(all_content_chunks)}")
                st.write(f"📝 Total content length: {len(''.join(all_content_chunks))}")
                st.write(f"📝 Final result length: {len(result_data)}")
                st.write(f"🎯 Question type: {identify_question_type(user_question or '')}")
                
                if debug_info:
                    st.write("Debug messages:")
                    for info in debug_info[-5:]:
                        st.write(info)
                
                if all_content_chunks:
                    st.write("Content chunks preview:")
                    for i, chunk in enumerate(all_content_chunks[:3]):
                        st.write(f"Chunk {i+1}: {chunk[:100]}...")

        return result_data if result_data else "⚠️ No relevant content found in the response."

    except requests.exceptions.RequestException as e:
        error_msg = f"🌐 Network error: {str(e)}"
        if st.session_state.debug_mode:
            st.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"💥 Error parsing response: {str(e)}"
        if st.session_state.debug_mode:
            st.error(error_msg)
        return error_msg

def create_targeted_prompt(message: str, chat_history: List[Dict]) -> str:
    """Create a highly targeted prompt for precise answers."""
    
    message_lower = message.lower()
    question_type = identify_question_type(message)
    
    # Extract specific numbers and symbols from the question
    question_numbers = re.findall(r'\b\d+\b', message)
    question_symbols = re.findall(r'\b[A-Z]\b', message.upper())
    
    # Handle reverse lookups with specific prompts
    if question_type == 'reverse_load_lookup':
        weight_matches = re.findall(r'(\d+)\s*kg', message_lower)
        if weight_matches:
            target_weight = weight_matches[0]
            return f"What is the exact load index number for a tire that can hold {target_weight}kg? Provide the specific load index number and confirm it handles exactly {target_weight}kg or the closest capacity. Format: 'Load index [NUMBER] can hold [WEIGHT]kg per tire.'"
    
    elif question_type == 'reverse_speed_lookup':
        speed_matches = re.findall(r'(\d+)\s*(?:km/h|mph)', message_lower)
        if speed_matches:
            target_speed = speed_matches[0]
            return f"What is the speed rating symbol for tires rated up to {target_speed} km/h? Provide the specific symbol and confirm the maximum speed. Format: 'Speed symbol [LETTER] indicates maximum speed of [SPEED] km/h.'"
    
    # Specific handling for nomenclature questions
    elif 'symbol r' in message_lower and 'indicates' in message_lower:
        return f"What does the symbol R mean in tire size designation like 195/55R16? Explain specifically what R indicates about tire construction. Answer format: 'The symbol R indicates [EXPLANATION].'"
    
    elif any(num in message for num in ['195', '55', '16']) and 'indicates' in message_lower:
        if '195' in message:
            return f"In tire size 195/55R16, what does the number 195 specifically represent? Answer format: '195 indicates [SPECIFIC MEANING].'"
        elif '55' in message:
            return f"In tire size 195/55R16, what does the number 55 specifically represent? Answer format: '55 indicates [SPECIFIC MEANING].'"
        elif '16' in message:
            return f"In tire size 195/55R16, what does the number 16 specifically represent? Answer format: '16 indicates [SPECIFIC MEANING].'"
    
    elif 'aspect ratio' in message_lower:
        return f"What is aspect ratio in tire nomenclature? Explain what it means and how it's calculated. Be specific about the relationship between sidewall height and tire width."
    
    elif 'components' in message_lower and 'tire' in message_lower:
        return f"List the main components of a tire and specify what materials each component is made of. Include tread, belts, sidewall, and other major parts."
    
    # Specific prompt templates for different question types
    elif 'load index' in message_lower and question_numbers:
        target_index = question_numbers[0]
        return f"What is the exact weight capacity in kg for load index {target_index}? Answer format: 'Load index {target_index} can hold [WEIGHT]kg per tire.'"
    
    elif 'speed' in message_lower and 'symbol' in message_lower and question_symbols:
        target_symbol = question_symbols[0]
        return f"What is the maximum speed for speed rating symbol {target_symbol}? Answer format: 'Speed symbol {target_symbol} indicates maximum speed of [SPEED] km/h.'"
    
    elif any(word in message_lower for word in ['components', 'parts', 'made of']) and 'tire' in message_lower:
        return f"Question: {message}\n\nList the main tire components and what each is made of. Be specific about materials (rubber, steel, fabric, etc.)."
    
    elif any(word in message_lower for word in ['195', '55', 'r16', 'nomenclature', 'means']):
        return f"Question: {message}\n\nExplain what each part of the tire size designation represents. Be specific about what each number means."
    
    elif 'tread depth' in message_lower:
        return f"Question: {message}\n\nProvide the specific tread depth measurement in mm and when to change tires."
    
    else:
        return f"Question: {message}\n\nAnswer with specific facts, numbers, and measurements. Be precise and direct."

def send_to_api(message: str, chat_history: List[Dict]) -> str:
    """Send message to API and get response with improved error handling."""
    try:
        enhanced_message = create_targeted_prompt(message, chat_history)
        
        # Build conversation context (limit to recent messages to avoid token limits)
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
            "temperature": 0.7,  # Lower temperature for more precise answers
            "maxTokens": 1024    # Reduced to encourage concise answers
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json, */*",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
        
        session = create_session_with_retries()
        
        # Make request with proper SSL handling
        response = session.post(
            st.session_state.api_url,
            json=payload,
            headers=headers,
            timeout=(30,180),  # Reduced timeout for faster failure detection
            verify=st.session_state.ssl_verify,  # Use configurable SSL verification
            stream=True,
            allow_redirects=True  # Allow redirects
        )

        if response.status_code == 401:
            return "❌ Authentication failed: Invalid API token. Please check your token."
        elif response.status_code == 403:
            return "❌ Access forbidden: You don't have permission to access this resource."
        elif response.status_code == 404:
            return "❌ API endpoint not found: Please verify the API URL."
        elif response.status_code != 200:
            return f"❌ API error {response.status_code}: {response.text[:200]}"

        return parse_sse_stream(response, message)

    except requests.exceptions.Timeout:
        return "⏰ Request timed out. The server is taking too long to respond."
    except requests.exceptions.ConnectionError as e:
        if "Name or service not known" in str(e) or "nodename nor servname provided" in str(e):
            return "🔌 DNS Error: Cannot resolve the server address. Check the API URL."
        elif "Connection refused" in str(e):
            return "🔌 Connection refused: The server is not accepting connections."
        else:
            return "🔌 Connection error: Unable to reach the API server. Check your network connection."
    except requests.exceptions.SSLError as e:
        return f"🔒 SSL Error: {str(e)[:200]}. Try disabling SSL verification in settings."
    except Exception as e:
        return f"💥 Unexpected error: {str(e)[:200]}"

def validate_settings():
    """Validate API settings."""
    if not st.session_state.api_url:
        st.error("❌ API URL is required!")
        return False
    if not st.session_state.api_token:
        st.error("❌ API Token is required!")
        return False
    
    # Basic URL validation
    if not st.session_state.api_url.startswith(('http://', 'https://')):
        st.error("❌ API URL must start with http:// or https://")
        return False
    
    return True

def main():
    """Main application function."""
    st.set_page_config(
        page_title="Sales Chat Assistant", 
        page_icon=icon_image, 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.image(icon_image, width=200)
    st.title("Sales Chat Assistant")
    st.markdown("*Ask specific questions about tire...")

 # Chat container
    chat_container = st.container()
    
    with chat_container:
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
                # Show thinking indicator
                thinking_placeholder = st.empty()
                thinking_placeholder.markdown("🤔 *Analyzing your question...*")
                
                # Get response from API
                response = send_to_api(prompt, st.session_state.messages[:-1])
                
                # Clear thinking indicator
                thinking_placeholder.empty()
                
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
                    
                    # Show question type if in debug mode
                    if st.session_state.debug_mode:
                        question_type = identify_question_type(prompt)
                        st.info(f"🏷️ Question type: {question_type}")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "|Powered by RAG Technology"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
