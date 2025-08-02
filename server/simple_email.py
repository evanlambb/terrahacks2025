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
    API_KEY = "re_Xu9Xyoig_5VLKuRJNSh51gZA16dtFKLB2"  # Get this from resend.com
    
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
        "from": "onboarding@resend.dev",  # Free sender address (no setup needed)
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
        to_email="evanlamb848@gmail.com",  # Must be your Resend account email in testing mode
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
   You are an expert peer-support evaluator. Given a conversation transcript between the user and a peer exhibiting early signs of depression, you will:

1. Score the User’s supportive dialogue from 0–100, weighting each research-backed criterion equally. For each, consider the study’s population, methodology, and key findings:

1. Early Sign Recognition (Irritability)

Source: Wahid et al. (2022), Child and Adolescent Psychiatry and Mental Health

Population & Method: Qualitative interviews with 30 Nepali adolescents diagnosed with depression.

Key Finding: Teens more often reported irritability, restlessness, and social withdrawal than overt sadness when depressed. Early peers noticing small mood swings can prompt timely support.

2. Sleep Concern Inquiry

Source: Mental Health First Aid USA (2018) guidelines

Basis: Expert consensus and review of adolescent depression literature.

Key Finding: Sleep disturbances (insomnia or hypersomnia) are core diagnostic criteria in adolescents and strongly correlate with worsening mood and concentration. Asking about sleep shows attunement to clinical red flags.

3. Academic Function Check

Source: Roach et al. (2021), Journal of Child and Adolescent Psychiatric Nursing

Population & Method: Mixed-methods study of 50 teens supporting friends with mental health needs.

Key Finding: Friends commonly observed missed assignments, declining grades, and study avoidance as early depression indicators. Discussing school performance invites reflection on functional impact.

4. Open-Ended, Empathetic Questions

Source: Villines (2023), Medical News Today (summarizing Samaritans Active Listening)

Basis: Guidelines distilled from crisis-support training programs.

Key Finding: Questions that cannot be answered “yes” or “no” (e.g., “What’s been hardest about your sleep lately?”) foster more detailed disclosures, giving the speaker control over depth and pacing.

5. Validation & Reflective Listening

Source: Samaritans Active Listening Guide (2023)

Basis: Decades of crisis-helpline data and qualitative feedback.

Key Finding: Reflecting back feelings (“It sounds like you’re overwhelmed”) and explicit validation (“Your feelings are valid”) reduce shame and isolation, making teens feel truly heard.

6. Avoidance of Minimizing or Fix-It Language

Source: Mental Health First Aid USA (2018)

Basis: Compilation of peer-support missteps observed in adolescent populations.

Key Finding: Comments like “Just cheer up” or “Others have it worse” shut down conversations and reinforce stigma. Teens need compassion, not quick fixes or comparisons.

7. Offer of Presence & Supportive Actions

Source: Mayo Clinic Staff (2023), Depression: Supporting a family member or friend

Basis: Clinical best practices for non-professional support.

Key Finding: Practical help—studying together, bringing meals, or sitting quietly—builds belonging and counters inertia. Even small gestures signal “You’re not alone.”

2. List Strengths

For each positive point, provide the transcript line number and reference the specific study (e.g., “(Validation – Samaritans 2023: reduced isolation)”).

3. List Improvements

For each suggestion, cite the transcript line number and tie it back to the research context (e.g., “(Sleep Inquiry – MHFA USA 2018: core symptom missed)”).

Output Format: 
- Score out of 100
- Analysis of the things that they didd well or poorly

Do not include any other text in your response.



Example Transcript Snippet:

…
User: "Hey Aaron, I’ve noticed you seem more irritable at dinner…"  
System: "Yeah, I’m exhausted and barely sleeping."  
User: "What time have you been going to bed lately, and how restful has it been?"  
…
Begin your research-anchored evaluation now. Find the chats below.

{chats_json}
    """)

    print(response.text)
    return response.text






