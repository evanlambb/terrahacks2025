"""
Quick email test - just run this after setting up your API key
"""
from email_sender import send_to_myself

# Send a quick test email
if __name__ == "__main__":
    print("🚀 Sending test email...")
    
    success = send_to_myself(
        subject="🎉 Python Email Setup Working!",
        message="Congratulations! Your Python email setup is working perfectly. This took less than 5 minutes to set up!"
    )
    
    if success:
        print("📧 Check your inbox!")
    else:
        print("💡 Make sure you've set your RESEND_API_KEY and MY_EMAIL environment variables")