import json
import re

transcript_path = r'C:\Users\Dell\.gemini\antigravity\brain\8727bfb8-e803-466a-b3b2-fa8a6250f9ca\.system_generated\logs\transcript_full.jsonl'

with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        if 'import streamlit' in line and 'app.py' in line and 'write_to_file' in line:
            try:
                data = json.loads(line)
                for call in data.get('tool_calls', []):
                    args = call.get('arguments', call.get('args', {}))
                    if args.get('TargetFile', '').endswith('app.py') and 'CodeContent' in args:
                        with open(r'C:\Users\Dell\.gemini\antigravity\scratch\cricket_shot_detection\app.py', 'w', encoding='utf-8') as out:
                            out.write(args['CodeContent'])
                        print('Done extracting app.py!')
                        exit(0)
            except Exception as e:
                pass

print('Could not find it cleanly. Trying regex fallback.')
with open(transcript_path, 'r', encoding='utf-8') as f:
    content = f.read()

# We look for a string that represents the app.py code
import ast
for line in content.splitlines():
    if 'import streamlit' in line and 'app.py' in line:
        try:
            d = json.loads(line)
            for call in d.get('tool_calls', []):
                args = call.get('arguments', call.get('args', {}))
                if 'CodeContent' in args and len(args['CodeContent']) > 1000:
                    with open(r'C:\Users\Dell\.gemini\antigravity\scratch\cricket_shot_detection\app.py', 'w', encoding='utf-8') as out:
                        out.write(args['CodeContent'])
                    print('Recovered using dictionary parsing.')
                    exit(0)
        except:
            pass

print('Failed.')
