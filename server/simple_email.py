import requests
import os
import json

import google.generativeai as genai
from convert_chats import *


def send_email(to_email, chats):
    """
    Send email using Resend API - simplest possible implementation
    """
    # Replace with your actual API key from Resend dashboard
    API_KEY = os.getenv("RESEND_API_KEY")

    url = "https://api.resend.com/emails"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    llm_response = generate_analysis(chats)

    # Add research section at the bottom
    research_section = """
    
---

## Research References

This analysis is based on the following peer-reviewed research:

1. **Adolescent perspectives on depression as a disease of loneliness: a qualitative study with youth and other stakeholders in urban Nepal**  
   [https://doi.org/10.1186/s13034-022-00481-y](https://doi.org/10.1186/s13034-022-00481-y)

2. **Kids helping kids: The lived experience of adolescents who support friends with mental health needs**  
   [https://doi.org/10.1111/jcap.12299](https://doi.org/10.1111/jcap.12299)
"""

    message = f"{llm_response}{research_section}"
    subject = "Peer Support Analysis Report"

    # Email data
    data = {
        # Free sender address (no setup needed)
        "from": "onboarding@resend.dev",
        "to": [to_email],
        "subject": subject,
        "html": f"<pre style='white-space: pre-wrap; font-family: Arial, sans-serif;'>{message}</pre>"
    }

    # Send the email
    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        print(f"✅ Email sent successfully to {to_email}")
        return True
    else:
        print(f"❌ Failed to send email: {response.text}")
        return False


# Example usage
if __name__ == "__main__":
    # Send email to yourself with sample chat data
    sample_chats = [
        "Hey Aaron, how have you been lately?",
        "I've been okay, just busy with school stuff.",
        "I noticed you seem a bit tired in class. Everything alright?",
        "Yeah, just haven't been sleeping well lately. Been staying up working on projects."
    ]

    send_email(
        # Must be your Resend account email in testing mode
        to_email="evanlamb848@gmail.com",
        chats=sample_chats
    )

# takes a list of strings that are the chat history and returns a string that is the analysis.
# Even indexes are the user's messages and odd indexes are the AI's messages.


def generate_analysis(chats):
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    chats_json = convert_chats_to_json(chats, "chats.json")

    response = model.generate_content(f"""
SYSTEM:
You are “PeerSupportEvaluator,” an expert in assessing peer-to-peer conversations where a friend comforts a depressed university student. Your job is to read a transcript of the User’s supportive responses and produce a clear, human-friendly evaluation.

When given a conversation, follow these steps:

1. Identify each time the User offers comfort or support.

2. Rate the User’s responses in three areas (each from 1 to 5 stars):
   • Empathetic Language (40%):  
     – ★☆☆☆☆ No warmth or reflection  
     – ★★★☆☆ Basic mirroring and validation  
     – ★★★★★ Deep emotional insight, gentle patience  
   • Supportive Phrases (30%):  
     – ★☆☆☆☆ No known supportive phrases  
     – ★★★☆☆ Occasional “I care about you,” “Your feelings matter”  
     – ★★★★★ Natural, heartfelt use of “You matter to me,” “How can I help?”  
   • Avoidance of Harm (30%):  
     – ★☆☆☆☆ Includes invalidating remarks or unsolicited advice  
     – ★★★☆☆ Mostly respectful, with a minor slip  
     – ★★★★★ Always respectful, never judgmental or directive  

3. Calculate an overall star rating out of 5, using the weighted formula:
   > overall = Empathy×0.40 + Supportive×0.30 + Avoidance×0.30  

4. For each category, give 1–2 brief quotes from the transcript and explain why they earned their stars.

5. Offer one or two practical tips for improving in each area.

6. Present your evaluation in plain text like this:

Empathy: ★★★★☆  
• Quote: “It sounds like you’ve been feeling overwhelmed…” — reflects and validates emotion.  
• Tip: Try longer pauses to let your friend collect their thoughts.

Supportive Phrases: ★★★☆☆  
• Quote: “Your feelings are valid.” — good validation, but could add “I’m here for you.”  
• Tip: Use “You matter to me” more often to reinforce worth.

Avoidance of Harm: ★★★★★  
• Quote: “Would you like some space?” — shows respect for autonomy.  

Overall: ★★★★☆  
General feedback: You show strong empathy, just add more strategic encouragement.

Begin your research-anchored evaluation now. Find the chats below.

{chats_json}
    """)

    print(response.text)
    return response.text
