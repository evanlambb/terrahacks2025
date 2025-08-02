from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import whisper
import tempfile
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

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for Unity client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        # First, detect mood and intensity from the transcript
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from agent.chatbot_tools import get_mood_with_intensity
        
        mood_data = get_mood_with_intensity(transcript)
        logger.info(f"Detected mood: {mood_data['mood']} with intensity: {mood_data['intensity']}")
        
        # Send mood detection result first
        yield f"data: {json.dumps({'type': 'mood', 'content': mood_data['mood'] + ' ' + str(mood_data['intensity'])})}\n\n"

        # Buffer for building complete sentences
        sentence_buffer = ""

        from agent.chatbot import graph, config, HumanMessage
        for event in graph.stream(
            {"messages": [HumanMessage(content=transcript)]}, 
            config=config):
            
            for value in event.values():
                if "messages" in value and value["messages"]:
                    chunk = value["messages"][-1].content
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

        # Step 1.5: Detect mood and intensity
        logger.info("Detecting mood from transcript...")
        try:
            from agent.chatbot_tools import get_mood_with_intensity
            mood_data = get_mood_with_intensity(transcript)
            logger.info(f"Detected mood: {mood_data['mood']} with intensity: {mood_data['intensity']}")
        except Exception as mood_e:
            logger.error(f"Mood detection error: {mood_e}")
            mood_data = {'mood': 'neutral', 'intensity': 50}

        # Step 2: Use LangGraph agent directly
        logger.info("Sending transcript to LangGraph agent...")
        try:
            from agent.chatbot import graph, config, HumanMessage
            result = graph.invoke(
                {"messages": [HumanMessage(content=transcript)]}, 
                config=config
            )
            ai_response = result["messages"][-1].content
            logger.info(f"LangGraph agent response: {ai_response}")

        except Exception as agent_e:
            logger.error(f"LangGraph agent error: {agent_e}")
            return jsonify({'error': f'AI response failed: {str(agent_e)}'}), 500

        # Step 3: Return transcript, AI response, and mood data
        return jsonify({
            'transcript': transcript,
            'ai_response': ai_response,
            'mood': mood_data['mood'],
            'mood_intensity': mood_data['intensity'],
            'language': transcription_result.get('language', 'unknown')
        })

    except Exception as e:
        logger.error(f"Error during voice chat: {str(e)}")
        return jsonify({'error': f'Voice chat failed: {str(e)}'}), 500

@app.route('/get-score-and-email', methods=['POST'])
def get_score_and_email():
    try:
        data = request.get_json()
        email = data.get('email')
        chat = data.get('chat')

        send_email(email, chat)

        return jsonify({'status': 'success', 
                        'message': 'Email sent successfully',
                        'email': email,
                        'timestamp': datetime.now().isoformat()}), 200

    except Exception as e:
        logger.error(f"Error during get score and email: {str(e)}")
        return jsonify({'error': f'Get score and email failed: {str(e)}'}), 500
        


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


@app.route('/session/<session_id>/history', methods=['GET'])
def get_session_history(session_id):
    """Get conversation history for a session"""
    try:
        # This would require implementing a way to retrieve conversation history
        # from the LangGraph memory system
        logger.info(f"Requested history for session: {session_id}")
        return jsonify({
            'session_id': session_id,
            'message': 'History retrieval not yet implemented',
            'note': 'This would require additional implementation to access LangGraph memory'
        })
    except Exception as e:
        logger.error(f"Error getting session history: {str(e)}")
        return jsonify({'error': f'History retrieval failed: {str(e)}'}), 500


@app.route('/test-memory', methods=['POST'])
def test_memory_persistence():
    """Test endpoint to verify memory persistence across multiple requests"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        user_message = data['message']
        test_id = data.get('test_id', 'unknown')
        turn_number = data.get('turn_number', 0)
        
        logger.info(f"=== MEMORY TEST TURN {turn_number} ===")
        logger.info(f"Test ID: {test_id}")
        logger.info(f"User message: {user_message}")
        logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 0.5: Test mood detection
        logger.info("Step 0.5: Testing mood detection...")
        try:
            from agent.chatbot_tools import get_mood_with_intensity
            mood_data = get_mood_with_intensity(user_message)
            logger.info(f"Step 0.5: Detected mood: {mood_data['mood']} with intensity: {mood_data['intensity']}")
        except Exception as mood_e:
            logger.error(f"Step 0.5: Mood detection error: {mood_e}")
            mood_data = {'mood': 'neutral', 'intensity': 50}
        
        # Step 1: Test direct LangGraph response
        logger.info("Step 1: Calling LangGraph agent...")
        start_time = time.time()
        
        try:
            from agent.chatbot import graph, config, HumanMessage
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_message)]}, 
                config=config
            )
            ai_response = result["messages"][-1].content
            response_time = time.time() - start_time
            
            logger.info(f"Step 1: LangGraph response received in {response_time:.2f}s")
            logger.info(f"Step 1: AI Response: {ai_response}")
            
        except Exception as agent_e:
            logger.error(f"Step 1: LangGraph agent error: {agent_e}")
            return jsonify({'error': f'AI response failed: {str(agent_e)}'}), 500
        
        # Step 2: Test streaming response (simulate a few chunks)
        logger.info("Step 2: Testing streaming response...")
        stream_chunks = []
        try:
            from agent.chatbot import graph, config, HumanMessage
            for event in graph.stream(
                {"messages": [HumanMessage(content=user_message)]}, 
                config=config):
                
                for value in event.values():
                    if "messages" in value and value["messages"]:
                        chunk = value["messages"][-1].content
                        if chunk:
                            stream_chunks.append(chunk)
                            logger.info(f"Step 2: Stream chunk {len(stream_chunks)}: {chunk[:100]}...")
                            if len(stream_chunks) >= 3:  # Just test first few chunks
                                break
                        if len(stream_chunks) >= 3:
                            break
                if len(stream_chunks) >= 3:
                    break
        except Exception as stream_e:
            logger.error(f"Step 2: Streaming error: {stream_e}")
        
        # Step 3: Check if state file was created/updated
        logger.info("Step 3: Checking state file...")
        state_file = "server/state.json"
        state_info = "No state file found"
        try:
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                    state_info = f"State file exists with emotion: {state_data.get('current_emotion', 'unknown')}"
                    logger.info(f"Step 3: {state_info}")
            else:
                logger.info("Step 3: No state file found yet")
        except Exception as state_e:
            logger.error(f"Step 3: Error reading state file: {state_e}")
        
        # Step 4: Memory verification summary
        logger.info("Step 4: Memory verification summary...")
        logger.info(f"  - Turn number: {turn_number}")
        logger.info(f"  - Test ID: {test_id}")
        logger.info(f"  - Response time: {response_time:.2f}s")
        logger.info(f"  - Stream chunks received: {len(stream_chunks)}")
        logger.info(f"  - State file status: {state_info}")
        
        # Step 5: Provide context for next turn
        context_hint = ""
        if turn_number == 0:
            context_hint = "First turn - no previous context"
        elif turn_number == 1:
            context_hint = "Second turn - should remember first turn"
        else:
            context_hint = f"Turn {turn_number} - should remember all previous turns"
        
        logger.info(f"Step 5: Context hint: {context_hint}")
        logger.info("=== END MEMORY TEST TURN ===\n")
        
        return jsonify({
            'test_id': test_id,
            'turn_number': turn_number,
            'user_message': user_message,
            'ai_response': ai_response,
            'mood': mood_data['mood'],
            'mood_intensity': mood_data['intensity'],
            'response_time': response_time,
            'stream_chunks_count': len(stream_chunks),
            'state_file_status': state_info,
            'context_hint': context_hint,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"Error during memory test: {str(e)}")
        return jsonify({'error': f'Memory test failed: {str(e)}'}), 500


@app.route('/test-memory-reset', methods=['POST'])
def test_memory_reset():
    """Test endpoint to verify memory can be reset by restarting the server"""
    try:
        logger.info("=== MEMORY RESET TEST ===")
        logger.info("This endpoint simulates what happens when the server restarts")
        logger.info("In a real scenario, you would restart the Flask server")
        logger.info("For testing, we'll just log the current state")
        
        # Check current state
        state_file = "server/state.json"
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state_data = json.load(f)
                logger.info(f"Current state before 'reset': {state_data}")
        else:
            logger.info("No state file found")
        
        logger.info("=== END MEMORY RESET TEST ===\n")
        
        return jsonify({
            'message': 'Memory reset test completed',
            'note': 'To actually reset memory, restart the Flask server',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"Error during memory reset test: {str(e)}")
        return jsonify({'error': f'Memory reset test failed: {str(e)}'}), 500


if __name__ == '__main__':
    print("Starting Whisper Flask server with Gemini integration...")
    print("Make sure to install required packages: pip install flask flask-cors openai-whisper google-generativeai python-dotenv soundfile librosa")
    print("Also ensure your GOOGLE_API_KEY is set in your .env file")
    app.run(host='0.0.0.0', port=5000, debug=True)
