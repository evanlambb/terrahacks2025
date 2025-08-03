from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import whisper
import os
import logging
import json
from datetime import datetime
import time
from dotenv import load_dotenv
# Import the LangGraph agent
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simple_email import send_email
import tempfile
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Whisper model
model = whisper.load_model("base")

# Global conversation history for single user
conversation_history = []

stage = 1


def add_message(role, content):
    """Add message to conversation history"""
    conversation_history.append({
        "role": role, 
        "content": content
    })
    
    # Keep only last 10 messages to prevent memory bloat
    if len(conversation_history) > 10:
        conversation_history.pop(0)  # Remove oldest message


def get_conversation_context():
    """Get conversation context for Gemini prompt"""
    if not conversation_history:
        return ""
    
    context = "Previous conversation:\n"
    for msg in conversation_history[-6:]:  # Last 6 messages for context
        role_name = "User" if msg["role"] == "user" else "Assistant"
        context += f"{role_name}: {msg['content']}\n"
    
    return context + "\n"


def load_system_prompt():
    """Load the system prompt from prompt_maya.txt"""
    try:
        prompt_path = os.path.join(os.path.dirname(__file__), 'prompt_maya.txt')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load system prompt: {e}")
        return "You are a helpful assistant."


def detect_mood_and_generate_response(transcript):
    """Detect mood from transcript and generate appropriate response with conversation context"""
    try:
        # Load system prompt
        system_prompt = load_system_prompt()
        print(f"Using system prompt: {system_prompt}")
        # Get conversation context
        context = get_conversation_context()
        
        # Create the full prompt with system prompt, context, and current message
        full_prompt = f"""{system_prompt}

{context}Current user message: "{transcript}"

Based on your character as Maya and the conversation history (if any), respond naturally as Maya would. Also analyze the intended mood/emotion from the user's message (choose from: happy, sad, angry), kind messages should have a happy mood, while critical or negative messages should have a sad or angry mood.
Include what stage in the conversation you are at (1, 2, 3, or 4). The previous stage was {stage}, if the previous message was suitable for the criteria then move to the next stage and respond according to the next stage. Only move to the next stage if the criteria in the system prompt has been met.
Also include whether the conversations is over or not, ie both parties have said goodbye and stage 4 has been reached.

Please respond in this exact JSON format:
{{
    "mood": "detected_mood",
    "intensity": intensity_score_0_to_100,
    "response": "Maya's natural response as defined in the system prompt",
    "stage": current_stage_number,
    "conversation_over": true or false
}}
"""
        
        logger.info("Calling Gemini for mood detection and response generation...")
        response = gemini_model.generate_content(full_prompt)
        
        # Parse the JSON response
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        result = json.loads(response_text)
        logger.info(f"Gemini response: mood={result.get('mood')}, intensity={result.get('intensity')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Mood detection/response generation failed: {e}")
        # Return fallback response
        return {
            "mood": "neutral",
            "intensity": 50,
            "response": "Hey, I'm having some trouble processing that right now. Mind saying it again?",
            "stage": 1
        }


def transcribe_audio_file(audio_file):
    """Helper function to transcribe an audio file using Whisper"""
    tmp_path = None
    try:
        # Create temporary file in current directory for better Whisper compatibility
        import uuid
        filename = f"temp_audio_{uuid.uuid4().hex[:8]}.wav"
        tmp_path = os.path.join(os.getcwd(), filename)

        # Save the uploaded audio to the temporary file
        audio_file.save(tmp_path)

        # Verify file exists and has content
        if not os.path.exists(tmp_path):
            raise FileNotFoundError(
                f"Temporary file was not created: {tmp_path}")

        file_size = os.path.getsize(tmp_path)
        if file_size == 0:
            raise ValueError("Audio file is empty")

        logger.info(
            f"Processing audio file: {tmp_path} (size: {file_size} bytes)")

        # Use memory-based approach to bypass file system issues
        logger.info("Starting Whisper transcription using memory approach...")
        try:
            import soundfile as sf
            import numpy as np

            # Read audio data directly into memory
            audio_data, sample_rate = sf.read(tmp_path)
            logger.info(
                f"Loaded audio: {len(audio_data)} samples at {sample_rate}Hz")

            # Convert to float32 and ensure it's 1D
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)  # Convert stereo to mono
            audio_data = audio_data.astype(np.float32)

            # Resample to 16kHz if needed (Whisper's expected sample rate)
            if sample_rate != 16000:
                import librosa
                audio_data = librosa.resample(
                    audio_data, orig_sr=sample_rate, target_sr=16000)
                logger.info("Resampled audio to 16kHz")

            # Transcribe the audio data directly
            result = model.transcribe(audio_data, verbose=True)

        except ImportError as import_e:
            logger.error(f"Missing required audio library: {import_e}")
            logger.info("Please install: pip install soundfile librosa")
            raise Exception(
                "Missing audio processing libraries. Run: pip install soundfile librosa")
        except Exception as memory_e:
            logger.error(f"Memory-based transcription failed: {memory_e}")
            logger.info("Falling back to file-based approach...")

            # Last resort: try the original file path approach
            try:
                result = model.transcribe(tmp_path)
            except Exception as final_e:
                logger.error(f"All transcription methods failed: {final_e}")
                raise final_e

        logger.info(f"Transcription successful: {result['text']}")
        return result

    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.info(f"Cleaned up temporary file: {tmp_path}")
            except Exception as cleanup_e:
                logger.warning(
                    f"Failed to clean up temporary file {tmp_path}: {cleanup_e}")


@app.route('/voice-chat', methods=['POST'])
def voice_chat():
    """Main voice chat endpoint - transcribe, detect mood, and respond"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio = request.files['file']
        if audio.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Step 1: Transcribe audio
        logger.info("Transcribing audio...")
        transcription_result = transcribe_audio_file(audio)
        transcript = transcription_result['text'].strip()

        if not transcript:
            return jsonify({'error': 'No speech detected'}), 400

        # Step 2: Add user message to conversation history
        add_message("user", transcript)

        # Step 3: Detect mood and generate response using Gemini with context
        logger.info("Detecting mood and generating response with conversation context...")
        gemini_result = detect_mood_and_generate_response(transcript)

        # Step 4: Add AI response to conversation history
        add_message("assistant", gemini_result['response'])

        return jsonify({
            'transcript': transcript,
            'ai_response': gemini_result['response'],
            'mood': gemini_result['mood'],
            'mood_intensity': gemini_result['intensity']
        })

    except Exception as e:
        logger.error(f"Voice chat error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get-score-and-email', methods=['GET'])
def get_score_and_email():
    global conversation_history
    try:
        send_email("evanlamb848@gmail.com", conversation_history)

        return jsonify({'status': 'success', 
                        'message': 'Email sent successfully',
                        'timestamp': datetime.now().isoformat()}), 200

    except Exception as e:
        logger.error(f"Error during get score and email: {str(e)}")
        return jsonify({'error': f'Get score and email failed: {str(e)}'}), 500
        


@app.route('/voice-chat-stream', methods=['POST'])
def voice_chat_stream():
    """Streaming voice chat endpoint - transcribe, detect mood, and stream response"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio = request.files['file']
        if audio.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Transcribe audio
        logger.info("Starting streaming voice chat - transcribing audio...")
        try:
            transcription_result = transcribe_audio_file(audio)
            transcript = transcription_result['text'].strip()
        except Exception as transcription_error:
            logger.error(f"Transcription failed: {transcription_error}")
            return jsonify({'error': f'Transcription failed: {str(transcription_error)}'}), 500

        if not transcript:
            return jsonify({'error': 'No speech detected'}), 400

        logger.info(f"Transcription successful: {transcript}")

        def generate():
            try:
                global stage
                # Send transcript first
                yield f"data: {json.dumps({'type': 'transcript', 'content': transcript})}\n\n"
                
                # Add user message to conversation history
                add_message("user", transcript)
                
                # Get mood and response from Gemini with conversation context
                logger.info("Getting mood detection and response from Gemini with context...")
                gemini_result = detect_mood_and_generate_response(transcript)
                
                # Add AI response to conversation history
                add_message("assistant", gemini_result['response'])
                
                # Send mood data
                yield f"data: {json.dumps({'type': 'mood', 'content': gemini_result['mood'] + ' ' + str(gemini_result['intensity'])})}\n\n"
                
                # Send the complete response immediately
                logger.info("Sending AI response...")
                yield f"data: {json.dumps({'type': 'text', 'content': gemini_result['response']})}\n\n"
                stage = max(gemini_result['stage'], stage)
                yield f"data: {json.dumps({'type': 'stage', 'content': str(stage)})}\n\n"
                yield f"data: {json.dumps({'type': 'conversation_over', 'content': str(gemini_result['conversation_over'])})}\n\n"
                yield f"data: {json.dumps({'type': 'complete'})}\n\n"
                
            except Exception as stream_error:
                logger.error(f"Stream generation error: {stream_error}")
                yield f"data: {json.dumps({'type': 'error', 'content': str(stream_error)})}\n\n"

        return Response(generate(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        })

    except Exception as e:
        logger.error(f"Streaming endpoint error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


@app.route('/conversation', methods=['GET'])
def get_conversation():
    """Get current conversation history"""
    return jsonify({
        'conversation_history': conversation_history,
        'message_count': len(conversation_history)
    })


@app.route('/conversation', methods=['DELETE'])
def clear_conversation():
    """Clear conversation history"""
    global conversation_history
    conversation_history = []
    logger.info("Conversation history cleared")
    return jsonify({'message': 'Conversation history cleared'})


if __name__ == '__main__':
    print("Starting simplified Whisper + Gemini Flask server...")
    print("Endpoints: /voice-chat (standard), /voice-chat-stream (streaming)")
    print("Flow: Audio → Whisper Transcription → Gemini Mood Detection & Response")
    app.run(host='0.0.0.0', port=5001, debug=True)
