from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import whisper
import tempfile
import os
import logging
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
# Import the LangGraph agent
from agent.chatbot import get_chatbot_response, stream_chatbot_response

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for Unity client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Load Whisper model (this might take a moment on first run)
model = whisper.load_model("base")


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


def stream_ai_response(transcript):
    """Generator function that yields AI response chunks using LangGraph"""
    try:
        # Buffer for building complete sentences
        sentence_buffer = ""

        for chunk in stream_chatbot_response(transcript):
            if chunk:
                sentence_buffer += chunk

                # Check if we have a complete sentence
                if any(punct in chunk for punct in ['.', '!', '?', '\n']):
                    if sentence_buffer.strip():
                        yield f"data: {json.dumps({'type': 'text', 'content': sentence_buffer.strip()})}\n\n"
                        sentence_buffer = ""

        # Send any remaining content
        if sentence_buffer.strip():
            yield f"data: {json.dumps({'type': 'text', 'content': sentence_buffer.strip()})}\n\n"

        # Send completion signal
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

# @app.route('/transcribe', methods=['POST'])
# def transcribe():
#     try:
#         if 'file' not in request.files:
#             return jsonify({'error': 'No audio file provided'}), 400

#         audio = request.files['file']
#         if audio.filename == '':
#             return jsonify({'error': 'No file selected'}), 400

#         # Use the helper function to transcribe
#         result = transcribe_audio_file(audio)

#         return jsonify({
#             'text': result['text'],
#             'language': result.get('language', 'unknown'),
#             'segments': result.get('segments', [])
#         })

#     except Exception as e:
#         logger.error(f"Error during transcription: {str(e)}")
#         return jsonify({'error': f'Transcription failed: {str(e)}'}), 500


@app.route('/voice-chat', methods=['POST'])
def voice_chat():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio = request.files['file']
        if audio.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Step 1: Transcribe the audio using Whisper
        logger.info("Starting voice chat - transcribing audio...")
        transcription_result = transcribe_audio_file(audio)
        transcript = transcription_result['text'].strip()

        if not transcript:
            return jsonify({'error': 'No speech detected in audio'}), 400

        logger.info(f"Transcription completed: {transcript}")

        # Step 2: Use LangGraph agent instead of direct OpenAI
        logger.info("Sending transcript to LangGraph agent...")
        try:
            ai_response = get_chatbot_response(transcript)
            logger.info(f"LangGraph agent response: {ai_response}")

        except Exception as agent_e:
            logger.error(f"LangGraph agent error: {agent_e}")
            return jsonify({'error': f'AI response failed: {str(agent_e)}'}), 500

        # Step 3: Return both transcript and AI response
        return jsonify({
            'transcript': transcript,
            'ai_response': ai_response,
            'language': transcription_result.get('language', 'unknown')
        })

    except Exception as e:
        logger.error(f"Error during voice chat: {str(e)}")
        return jsonify({'error': f'Voice chat failed: {str(e)}'}), 500


@app.route('/voice-chat-stream', methods=['POST'])
def voice_chat_stream():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio = request.files['file']
        if audio.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Step 1: Transcribe the audio
        logger.info("Starting streaming voice chat - transcribing audio...")
        transcription_result = transcribe_audio_file(audio)
        transcript = transcription_result['text'].strip()

        if not transcript:
            return jsonify({'error': 'No speech detected in audio'}), 400

        logger.info(f"Transcription completed: {transcript}")

        # Step 2: Create streaming response using LangGraph
        def generate():
            # First, send the transcript
            yield f"data: {json.dumps({'type': 'transcript', 'content': transcript})}\n\n"

            # Then stream the AI response
            for chunk in stream_ai_response(transcript):
                yield chunk

        return Response(generate(), mimetype='text/plain', headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Type': 'text/event-stream',
            'Access-Control-Allow-Origin': '*'
        })

    except Exception as e:
        logger.error(f"Error during streaming voice chat: {str(e)}")
        return jsonify({'error': f'Streaming voice chat failed: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


if __name__ == '__main__':
    print("Starting Whisper Flask server with ChatGPT integration...")
    print("Make sure to install required packages: pip install flask flask-cors openai-whisper openai python-dotenv soundfile librosa")
    print("Also ensure your OPENAI_API_KEY is set in your .env file")
    app.run(host='0.0.0.0', port=5000, debug=True)
