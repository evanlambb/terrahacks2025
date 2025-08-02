# Migration from ChatGPT to Google Gemini

This project has been migrated from OpenAI's ChatGPT to Google's Gemini API.

## Changes Made

### Dependencies
- Replaced `langchain-openai` with `langchain-google-genai`
- Replaced `openai>=1.0.0` with `google-generativeai>=0.3.0`

### Code Changes
- Updated all imports from `ChatOpenAI` to `ChatGoogleGenerativeAI`
- Changed model from `gpt-4o-mini` to `gemini-1.5-flash`
- Removed unused OpenAI client from whisper server

### Environment Variables
You need to update your `.env` file:

**Before:**
```
OPENAI_API_KEY=your_openai_api_key_here
```

**After:**
```
GOOGLE_API_KEY=your_google_api_key_here
```

## Getting Your Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file as `GOOGLE_API_KEY`

## Installation

Run the following command to install the updated dependencies:

```bash
pip install -r requirements.txt
```

## Notes

- The model `gemini-1.5-flash` is used as a fast and efficient alternative to `gpt-4o-mini`
- All existing functionality should work the same way
- Gemini has different rate limits and pricing compared to OpenAI
