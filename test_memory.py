#!/usr/bin/env python3
"""
Comprehensive test script for the entire Flask application
Tests: Audio transcription, voice chat, memory persistence, and streaming
"""

import requests
import json
import time
import uuid
import os
import tempfile
import wave
import numpy as np

# Flask server URL
BASE_URL = "http://localhost:5000"

def create_test_audio(text, filename="test_audio.wav"):
    """Create a simple test audio file with synthesized speech"""
    try:
        # Try to use gTTS for text-to-speech
        from gtts import gTTS
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filename)
        print(f"Created audio file: {filename}")
        return filename
    except ImportError:
        print("gTTS not available, creating dummy audio file...")
        # Create a dummy WAV file if gTTS is not available
        sample_rate = 16000
        duration = 2.0  # 2 seconds
        samples = int(sample_rate * duration)
        
        # Generate a simple sine wave
        frequency = 440  # A4 note
        t = np.linspace(0, duration, samples, False)
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
        
        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)
        
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        print(f"Created dummy audio file: {filename}")
        return filename

def test_health_endpoint():
    """Test the health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_voice_chat_endpoint(audio_file, test_id, turn_number):
    """Test the voice-chat endpoint with audio file"""
    print(f"\n--- Testing Voice Chat (Turn {turn_number + 1}) ---")
    
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': (audio_file, f, 'audio/wav')}
            
            response = requests.post(f"{BASE_URL}/voice-chat", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Voice chat successful")
                print(f"   Transcript: {data.get('transcript', 'N/A')}")
                print(f"   AI Response: {data.get('ai_response', 'N/A')[:100]}...")
                print(f"   Language: {data.get('language', 'N/A')}")
                return data
            else:
                print(f"❌ Voice chat failed: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        print(f"❌ Voice chat error: {e}")
        return None

def test_voice_chat_stream_endpoint(audio_file, test_id, turn_number):
    """Test the streaming voice-chat endpoint"""
    print(f"\n--- Testing Streaming Voice Chat (Turn {turn_number + 1}) ---")
    
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': (audio_file, f, 'audio/wav')}
            
            response = requests.post(f"{BASE_URL}/voice-chat-stream", files=files, stream=True)
            
            if response.status_code == 200:
                print(f"✅ Streaming voice chat successful")
                transcript = ""
                ai_response = ""
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])  # Remove 'data: ' prefix
                                if data.get('type') == 'transcript':
                                    transcript = data.get('content', '')
                                    print(f"   Transcript: {transcript}")
                                elif data.get('type') == 'text':
                                    ai_response += data.get('content', '')
                                    print(f"   AI: {data.get('content', '')}")
                                elif data.get('type') == 'complete':
                                    print(f"   ✅ Stream completed")
                                    break
                            except json.JSONDecodeError:
                                continue
                
                return {
                    'transcript': transcript,
                    'ai_response': ai_response
                }
            else:
                print(f"❌ Streaming voice chat failed: {response.status_code}")
                return None
                
    except Exception as e:
        print(f"❌ Streaming voice chat error: {e}")
        return None

def test_memory_endpoint(message, test_id, turn_number):
    """Test the memory endpoint with text message"""
    print(f"\n--- Testing Memory Endpoint (Turn {turn_number + 1}) ---")
    
    try:
        payload = {
            'message': message,
            'test_id': test_id,
            'turn_number': turn_number
        }
        
        response = requests.post(
            f"{BASE_URL}/test-memory",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Memory test successful")
            print(f"   AI Response: {data.get('ai_response', 'N/A')[:100]}...")
            print(f"   Context: {data.get('context_hint', 'N/A')}")
            return data
        else:
            print(f"❌ Memory test failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Memory test error: {e}")
        return None

def comprehensive_application_test():
    """Test the entire application flow"""
    print("🚀 COMPREHENSIVE APPLICATION TEST")
    print("=" * 60)
    
    # Generate test ID
    test_id = str(uuid.uuid4())[:8]
    print(f"Test ID: {test_id}")
    
    # Test 1: Health endpoint
    if not test_health_endpoint():
        print("❌ Health check failed, stopping test")
        return
    
    # Test conversation flow
    test_conversations = [
        {
            'text': "Hello! My name is Alice.",
            'description': "Introduction"
        },
        {
            'text': "What's my name?",
            'description': "Memory test 1"
        },
        {
            'text': "How are you feeling today?",
            'description': "Mood detection test"
        },
        {
            'text': "What did I tell you my name was?",
            'description': "Memory test 2"
        },
        {
            'text': "Can you remember our conversation so far?",
            'description': "Full memory test"
        }
    ]
    
    results = []
    
    for i, conv in enumerate(test_conversations):
        print(f"\n{'='*20} TURN {i+1}: {conv['description']} {'='*20}")
        
        # Create audio file for this turn
        audio_file = f"test_audio_{i+1}.wav"
        create_test_audio(conv['text'], audio_file)
        
        # Test 2: Voice chat endpoint
        voice_result = test_voice_chat_endpoint(audio_file, test_id, i)
        
        # Test 3: Streaming voice chat endpoint
        stream_result = test_voice_chat_stream_endpoint(audio_file, test_id, i)
        
        # Test 4: Memory endpoint (direct text)
        memory_result = test_memory_endpoint(conv['text'], test_id, i)
        
        # Store results
        turn_result = {
            'turn': i + 1,
            'description': conv['description'],
            'text': conv['text'],
            'voice_chat': voice_result,
            'streaming': stream_result,
            'memory': memory_result
        }
        results.append(turn_result)
        
        # Clean up audio file
        if os.path.exists(audio_file):
            os.remove(audio_file)
        
        # Small delay between turns
        time.sleep(2)
    
    # Summary
    print(f"\n{'='*60}")
    print("COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        print(f"\nTurn {result['turn']}: {result['description']}")
        print(f"  Text: {result['text']}")
        
        if result['voice_chat']:
            print(f"  ✅ Voice Chat: {result['voice_chat'].get('ai_response', '')[:50]}...")
        else:
            print(f"  ❌ Voice Chat: Failed")
            
        if result['streaming']:
            print(f"  ✅ Streaming: {result['streaming'].get('ai_response', '')[:50]}...")
        else:
            print(f"  ❌ Streaming: Failed")
            
        if result['memory']:
            print(f"  ✅ Memory: {result['memory'].get('context_hint', '')}")
        else:
            print(f"  ❌ Memory: Failed")
    
    # Memory persistence analysis
    print(f"\n{'='*60}")
    print("MEMORY PERSISTENCE ANALYSIS")
    print(f"{'='*60}")
    
    memory_working = True
    for i, result in enumerate(results):
        if i == 0:
            # First turn - should have no previous context
            if result['memory'] and 'First turn' in result['memory'].get('context_hint', ''):
                print(f"✅ Turn {i+1}: Correctly identified as first turn")
            else:
                print(f"❌ Turn {i+1}: Should be identified as first turn")
                memory_working = False
        elif i == 1:
            # Second turn - should remember the name
            ai_response = result['memory'].get('ai_response', '') if result['memory'] else ''
            if 'Alice' in ai_response:
                print(f"✅ Turn {i+1}: Correctly remembered 'Alice'")
            else:
                print(f"❌ Turn {i+1}: Should remember 'Alice'")
                memory_working = False
        elif i == 4:
            # Last turn - should have full conversation context
            ai_response = result['memory'].get('ai_response', '') if result['memory'] else ''
            if 'conversation' in ai_response.lower() or 'Alice' in ai_response:
                print(f"✅ Turn {i+1}: Has conversation context")
            else:
                print(f"❌ Turn {i+1}: Should have conversation context")
                memory_working = False
    
    print(f"\n{'='*60}")
    if memory_working:
        print("🎉 MEMORY PERSISTENCE: WORKING CORRECTLY!")
        print("The agent maintains conversation context across requests.")
    else:
        print("⚠️  MEMORY PERSISTENCE: ISSUES DETECTED!")
        print("The agent may not be maintaining conversation context properly.")
    
    print(f"\nTo test memory reset:")
    print("1. Restart the Flask server")
    print("2. Run this test again")
    print("3. The agent should NOT remember 'Alice' in the second run")

def test_memory_reset():
    """Test memory reset functionality"""
    print(f"\n{'='*60}")
    print("MEMORY RESET TEST")
    print(f"{'='*60}")
    
    try:
        response = requests.post(f"{BASE_URL}/test-memory-reset")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Memory reset test: {data['message']}")
            print(f"   Note: {data['note']}")
        else:
            print(f"❌ Memory reset test failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Memory reset test error: {e}")

if __name__ == "__main__":
    print("🎯 COMPREHENSIVE FLASK APPLICATION TEST")
    print("Testing: Audio transcription, voice chat, streaming, and memory persistence")
    print("Make sure your Flask server is running on http://localhost:5000")
    print()
    
    try:
        # Run comprehensive test
        comprehensive_application_test()
        
        # Test memory reset
        test_memory_reset()
        
        print(f"\n{'='*60}")
        print("TEST COMPLETED!")
        print("Check the Flask server logs for detailed information.")
        print("All endpoints should work correctly for Unity integration.")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        print("Make sure the Flask server is running and accessible") 