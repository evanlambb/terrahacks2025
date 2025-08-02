# Terrahacks 2025 - LangGraph Chatbot

A conversational AI chatbot built with LangGraph, featuring tool integration for enhanced capabilities.

## Features

- Interactive chatbot with memory
- Weather information tool
- Extensible tool system
- Conversation state management

## Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd terrahacks2025
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Run the chatbot**
   ```bash
   python agent/chatbot.py
   ```

## Usage

- Type your messages and press Enter
- Type `quit`, `exit`, or `q` to exit
- The chatbot can provide weather information for cities

## Project Structure

```
terrahacks2025/
├── agent/
│   ├── chatbot.py          # Main chatbot implementation
│   └── chatbot_tools.py    # Tool definitions
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (create this)
└── README.md             # This file
```

## Adding New Tools

To add new tools, edit `agent/chatbot_tools.py` and add new functions with the `@tool` decorator.
