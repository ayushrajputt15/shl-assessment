import requests
import json
import os

def run_chat():
    url = "http://127.0.0.1:8000/chat"
    history = []
    
    print("=" * 60)
    print("      SHL ASSESSMENT RECOMMENDER - INTERACTIVE CHAT      ")
    print("=" * 60)
    print("Type your message and press Enter. Type 'exit' to quit.")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break
            
        if not user_input:
            continue
            
        if user_input.lower() == 'exit':
            print("Exiting chat.")
            break
            
        history.append({"role": "user", "content": user_input})
        
        # Send request
        try:
            resp = requests.post(url, json={"messages": history}, timeout=30)
            if resp.status_code != 200:
                print(f"\nError ({resp.status_code}): {resp.text}")
                history.pop()  # Remove last message on failure
                continue
                
            data = resp.json()
            reply = data.get("reply", "")
            recommendations = data.get("recommendations", [])
            eoc = data.get("end_of_conversation", False)
            
            print(f"\nAgent: {reply}")
            
            if recommendations:
                print("\nShortlist Recommendations:")
                for idx, item in enumerate(recommendations):
                    print(f"  {idx+1}. [{item['test_type']}] {item['name']}")
                    print(f"     URL: {item['url']}")
            
            # Save assistant reply to history
            history.append({"role": "assistant", "content": reply})
            
            if eoc:
                print("\n" + "=" * 60)
                print("Conversation ended by agent.")
                print("=" * 60)
                break
                
        except Exception as e:
            print(f"\nConnection failed: {e}")
            print("Make sure the FastAPI server is running! (python main.py)")
            history.pop()

if __name__ == '__main__':
    run_chat()
