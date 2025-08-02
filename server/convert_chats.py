import json

def convert_chats_to_json(chats, output_filename="chats.json"):
    """
    Convert a list of chats to JSON format where:
    - Even indexes (0, 2, 4...) are user messages  
    - Odd indexes (1, 3, 5...) are system messages
    
    Args:
        chats: List of chat messages
        output_filename: Name of output JSON file
    """
    
    result = []
    
    # Process messages in pairs
    for i in range(0, len(chats), 2):
        user_msg = chats[i] if i < len(chats) else ""
        system_msg = chats[i + 1] if i + 1 < len(chats) else ""
        
        result.append({
            "user": user_msg,
            "system": system_msg
        })
    
    # Save to JSON
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Converted {len(result)} chat pairs to {output_filename}")
    return result

# Example usage - replace this with your actual chat data
if __name__ == "__main__":
    # Your chats list goes here
    my_chats = [
        "Hello there!",                    # User (index 0)
        "Hi! How can I help you?",         # System (index 1)  
        "What's the weather like?",        # User (index 2)
        "It's sunny and 75°F today.",      # System (index 3)
        "Thanks!"                          # User (index 4)
        # System message missing for last user message (will be empty)
    ]
    
    # Convert to JSON
    converted = convert_chats_to_json(my_chats, "my_chats.json")
    
    # Print result
    print("\n📋 Converted chats:")
    for i, chat in enumerate(converted):
        print(f"{i+1}. User: '{chat['user']}'")
        print(f"   System: '{chat['system']}'")
        print()