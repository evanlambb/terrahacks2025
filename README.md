# TerraHacks 2025 - AI Voice Chat with Memory

A Flask-based voice chat application with AI memory persistence using LangGraph, Whisper, and Google Gemini.

## Features

- 🎤 **Voice Chat**: Real-time audio transcription using OpenAI Whisper
- 🧠 **Memory Persistence**: Conversation context maintained across sessions using LangGraph
- 📡 **Streaming Responses**: Real-time AI response streaming with Google Gemini
- 🎭 **Character AI**: Role-playing as Aaron, a CS student persona
- 🔧 **RESTful API**: Easy integration with Unity or other clients

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Environment Variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

### 3. Start the Server

```bash
python server/whisper_server.py
```

The server will start on `http://localhost:5000`

### 4. Test the Application

#### Quick Memory Test
```bash
python test_memory.py --simple
```

#### Full Comprehensive Test
```bash
python test_memory.py
```

## API Endpoints

### Health Check
- **GET** `/health` - Check server status

### Voice Chat
- **POST** `/voice-chat` - Send audio file, get transcript + AI response
- **POST** `/voice-chat-stream` - Send audio file, get streaming response

### Memory Testing
- **POST** `/test-memory` - Test memory persistence with text input
- **POST** `/test-memory-reset` - Test memory reset functionality

### Session Management
- **GET** `/session/<session_id>/history` - Get conversation history

## Testing Results

✅ **Memory Persistence**: Working perfectly - agent remembers conversation context
✅ **Text-based Chat**: Fully functional with LangGraph integration
✅ **Character AI**: Aaron persona working correctly
⚠️ **Audio Transcription**: Requires real speech (dummy audio not detected)

## Current Status

### Working Features
- ✅ Memory persistence across conversation turns
- ✅ Text-based chat with character AI
- ✅ Streaming responses
- ✅ Health endpoint
- ✅ Comprehensive testing framework

### Known Issues
- ⚠️ Audio transcription fails with dummy/synthetic audio files
- ⚠️ Requires real speech recordings for voice chat testing

### For Real Audio Testing
1. Install gTTS: `pip install gTTS`
2. Or use actual speech recordings
3. Voice chat endpoints will work with real speech

## Project Structure

```
terrahacks2025/
├── agent/
│   ├── chatbot.py          # LangGraph agent implementation
│   └── chatbot_tools.py    # Tools for weather, mood detection, state saving
├── server/
│   └── whisper_server.py   # Flask server with Whisper integration
├── test_memory.py          # Comprehensive testing framework
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Development

### Adding New Features
1. Extend the LangGraph agent in `agent/chatbot.py`
2. Add new tools in `agent/chatbot_tools.py`
3. Create new endpoints in `server/whisper_server.py`
4. Update tests in `test_memory.py`

### Memory System
The application uses LangGraph's checkpoint memory system to maintain conversation context. Each conversation turn is saved and retrieved automatically.

### Character AI
The AI plays the role of Aaron, a 20-year-old CS student with specific personality traits and conversation patterns.

## Troubleshooting

### Common Issues
1. **"No speech detected in audio"** - Use real speech recordings, not dummy audio
2. **Memory not persisting** - Check that the server is running and LangGraph is properly configured
3. **API key errors** - Ensure your `.env` file contains a valid Google API key

### Testing
- Use `python test_memory.py --simple` for quick memory verification
- Use `python test_memory.py` for comprehensive testing
- Check server logs for detailed error information

## License

This project is part of TerraHacks 2025.
