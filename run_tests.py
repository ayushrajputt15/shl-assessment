import os
import re
import json
import requests
import time
import subprocess
import sys
from dotenv import load_dotenv

# Load env variables for tests
load_dotenv(override=True)

# Setup local uvicorn process for testing
def start_server():
    print("Starting local FastAPI server for test replay...")
    # Open uvicorn.log file to write logs (avoids pipe buffer overflow hang)
    log_file = open("uvicorn.log", "w", encoding="utf-8")
    # Run uvicorn as a subprocess
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=log_file,
        stderr=log_file,
        cwd=r'c:\Users\Ayush\Desktop\SHL'
    )
    # Wait for server to wake up
    time.sleep(2.5)
    return proc, log_file

def stop_server(proc, log_file):
    print("Stopping local FastAPI server...")
    proc.terminate()
    proc.wait()
    log_file.close()

def parse_trace(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Split content into turns
    turns_raw = re.split(r'### Turn \d+', content)
    turns = []
    
    for tr in turns_raw[1:]:
        # Extract User message
        user_match = re.search(r'\*\*User\*\*\s*\n\s*>\s*(.*?)(?=\n\s*\*\*Agent\*\*)', tr, re.DOTALL)
        if not user_match:
            continue
        user_msg = user_match.group(1).strip()
        user_msg = re.sub(r'^>\s*', '', user_msg, flags=re.MULTILINE).strip()
        
        # Extract Agent message
        agent_match = re.search(r'\*\*Agent\*\*\s*\n\s*(.*?)(?=\n\s*_(?:No recommendations|`end_of_conversation`|\|))', tr, re.DOTALL)
        agent_reply = agent_match.group(1).strip() if agent_match else ""
        
        # Extract recommendations from table
        table_pattern = r'\|\s*#\s*\|\s*Name\s*\|\s*Test Type\s*\|\s*Keys\s*\|.*?\n((?:\|.*?\n)+)'
        table_match = re.search(table_pattern, tr)
        recs = []
        if table_match:
            rows = table_match.group(1).strip().split('\n')
            for row in rows:
                if '---' in row:
                    continue
                parts = [p.strip() for p in row.split('|')]
                if len(parts) >= 8:
                    name = parts[2]
                    test_type = parts[3]
                    url = parts[7].strip('<>')
                    recs.append({
                        'name': name,
                        'test_type': test_type,
                        'url': url
                    })
                    
        # Extract end_of_conversation
        eoc_match = re.search(r'_\s*`end_of_conversation`\s*:\s*\*\*(true|false)\*\*', tr)
        eoc = eoc_match.group(1) == 'true' if eoc_match else False
        
        turns.append({
            'user': user_msg,
            'agent': agent_reply,
            'expected_recs': recs,
            'expected_eoc': eoc
        })
        
    return turns

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9]', '', text)
    return text

def get_sequence_key(user_messages: list) -> str:
    normalized_list = [normalize_text(msg) for msg in user_messages]
    return json.dumps(normalized_list)

def run_replay_tests():
    conv_dir = r'c:\Users\Ayush\Desktop\SHL\GenAI_SampleConversations'
    server_proc, log_file = start_server()
    
    mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
    trace_db = {}
    if mock_mode:
        trace_db_path = "trace_database.json"
        if os.path.exists(trace_db_path):
            try:
                with open(trace_db_path, "r", encoding="utf-8") as f:
                    trace_db = json.load(f)
                print(f"Loaded offline trace database with {len(trace_db)} entries.")
            except Exception as e:
                print(f"Error loading offline trace database: {e}")
                
    if not mock_mode:
        print("WARNING: MOCK_MODE=false. Running live LLM tests. Adding a 4.5-second pacing delay between requests to avoid Gemini Free Tier rate limits...")
        
    try:
        # Check /health endpoint
        try:
            r = requests.get("http://127.0.0.1:8000/health", timeout=5)
            print(f"Health check status: {r.status_code}, response: {r.json()}")
        except Exception as e:
            print(f"Server health check failed: {e}")
            return
            
        total_convs = 0
        passed_convs = 0
        
        for filename in sorted(os.listdir(conv_dir), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0):
            if filename.endswith('.md'):
                total_convs += 1
                file_path = os.path.join(conv_dir, filename)
                print(f"\nReplaying {filename}...")
                turns = parse_trace(file_path)
                
                conversation_history = []
                conv_passed = True
                
                for idx, turn in enumerate(turns):
                    print(f"  Turn {idx+1}:")
                    user_query = turn['user']
                    print(f"    User: {user_query}")
                    
                    conversation_history.append({"role": "user", "content": user_query})
                    
                    if mock_mode:
                        user_messages = [m["content"] for m in conversation_history if m["role"] == "user"]
                        seq_key = get_sequence_key(user_messages)
                        
                        if seq_key in trace_db:
                            resp_json = trace_db[seq_key]
                            agent_reply = resp_json.get("reply", "")
                            recommendations = resp_json.get("recommendations", [])
                            end_of_conversation = resp_json.get("end_of_conversation", False)
                            latency = 0.01
                            print(f"    Agent (Mocked): {agent_reply[:100]}... (latency: {latency:.2f}s)")
                        else:
                            print(f"    [FAIL] Turn {idx+1} not found in offline database!")
                            conv_passed = False
                            break
                    else:
                        # Add pacing delay in live mode
                        if idx > 0:
                            time.sleep(4.5)
                            
                        # Call local service
                        start_time = time.time()
                        try:
                            resp = requests.post(
                                "http://127.0.0.1:8000/chat",
                                json={"messages": conversation_history},
                                timeout=35 # Allow a bit extra for live Gemini/Groq calls
                            )
                        except requests.exceptions.Timeout:
                            print(f"    [FAIL] Turn {idx+1} timed out!")
                            conv_passed = False
                            break
                        except Exception as e:
                            print(f"    [FAIL] Turn {idx+1} call failed: {e}")
                            conv_passed = False
                            break
                            
                        latency = time.time() - start_time
                        
                        if resp.status_code != 200:
                            print(f"    [FAIL] Server returned status code {resp.status_code}: {resp.text}")
                            conv_passed = False
                            break
                            
                        resp_json = resp.json()
                        agent_reply = resp_json.get("reply", "")
                        recommendations = resp_json.get("recommendations", [])
                        end_of_conversation = resp_json.get("end_of_conversation", False)
                        
                        print(f"    Agent: {agent_reply[:100]}... (latency: {latency:.2f}s)")
                    
                    # Log history for next turn
                    conversation_history.append({"role": "assistant", "content": agent_reply})
                    
                    # Verify recommendations
                    expected_recs = turn['expected_recs']
                    
                    actual_urls = {r['url'].lower().strip() for r in recommendations}
                    expected_urls = {r['url'].lower().strip() for r in expected_recs}
                    
                    if actual_urls != expected_urls:
                        print(f"    [FAIL] Recommendations mismatch!")
                        print(f"      Expected URLs: {sorted(list(expected_urls))}")
                        print(f"      Actual URLs:   {sorted(list(actual_urls))}")
                        conv_passed = False
                    else:
                        print(f"    [PASS] Recommendations match expected list (Size: {len(recommendations)})")
                        
                if conv_passed:
                    passed_convs += 1
                    print(f"Result for {filename}: PASSED")
                else:
                    print(f"Result for {filename}: FAILED")
                    
        print(f"\nTest Summary: {passed_convs} / {total_convs} conversations passed.")
        
    finally:
        stop_server(server_proc, log_file)

if __name__ == '__main__':
    run_replay_tests()
