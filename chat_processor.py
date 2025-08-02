import json
from typing import List, Dict

def process_chats_to_json(chats: List[str], output_file: str = "processed_chats.json") -> List[Dict[str, str]]:
    """
    Process a list of chat messages where even indexes are user messages 
    and odd indexes are system messages, then save as JSON.
    
    Args:
        chats: List of chat messages
        output_file: Output JSON filename
        
    Returns:
        List of dictionaries with user/system message pairs
    """
    
    processed_chats = []
    
    # Process messages in pairs
    for i in range(0, len(chats) - 1, 2):
        user_message = chats[i]
        system_message = chats[i + 1] if i + 1 < len(chats) else ""
        
        chat_pair = {
            "user": user_message,
            "system": system_message
        }
        processed_chats.append(chat_pair)
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_chats, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Processed {len(processed_chats)} chat pairs and saved to {output_file}")
    return processed_chats

def load_chats_from_file(input_file: str) -> List[str]:
    """
    Load chats from a text file (one message per line) or JSON file
    
    Args:
        input_file: Path to input file
        
    Returns:
        List of chat messages
    """
    if input_file.endswith('.json'):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                print("❌ JSON file should contain a list of messages")
                return []
    else:
        # Text file - one message per line
        with open(input_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() if line.strip()]

def example_usage():
    """Example of how to use the chat processor"""
    
    # Example chat list (even = user, odd = system)
    example_chats = [
        "Hello, how are you?",                           # User (index 0)
        "I'm doing great! How can I help you today?",    # System (index 1)
        "Can you tell me about the weather?",            # User (index 2) 
        "Sure! The weather today is sunny and warm.",    # System (index 3)
        "That's wonderful, thank you!",                  # User (index 4)
        "You're welcome! Is there anything else I can help with?",  # System (index 5)
        "No, that's all for now."                        # User (index 6)
    ]
    
    # Process and save
    processed = process_chats_to_json(example_chats, "example_chats.json")
    
    # Display the result
    print("\n📋 Processed chats:")
    for i, chat in enumerate(processed):
        print(f"\nChat {i + 1}:")
        print(f"  User: {chat['user']}")
        print(f"  System: {chat['system']}")

if __name__ == "__main__":
    print("🤖 Chat Processor - Convert chat lists to JSON format")
    print("=" * 50)
    
    # Run example
    example_usage()
    
    print("\n" + "=" * 50)
    print("💡 To use with your own data:")
    print("1. Import the functions: from chat_processor import process_chats_to_json")
    print("2. Call: process_chats_to_json(your_chat_list, 'output.json')")
    print("3. Or load from file: load_chats_from_file('input.txt')")