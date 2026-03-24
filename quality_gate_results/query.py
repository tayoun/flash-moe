#!/usr/bin/env python3
"""Query the server and extract response from streaming chunks."""
import sys
import json
import requests

def query(port, prompt, output_file):
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 256,
        "stream": False
    }

    try:
        resp = requests.post(url, json=payload, timeout=300)
        text = resp.text

        # If it's streaming SSE, parse chunks
        if text.startswith("data:"):
            content_parts = []
            for line in text.split('\n'):
                line = line.strip()
                if line.startswith("data:") and not line.startswith("data: [DONE]"):
                    try:
                        chunk = json.loads(line[5:].strip())
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                content_parts.append(delta["content"])
                    except json.JSONDecodeError:
                        pass
            content = "".join(content_parts)
        else:
            # Regular JSON response
            data = json.loads(text)
            content = data["choices"][0]["message"]["content"]

        # Save full content
        with open(output_file, 'w') as f:
            json.dump({"content": content, "raw_length": len(text)}, f, indent=2)

        print(content[:500])  # Print preview
        return content

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        with open(output_file, 'w') as f:
            json.dump({"error": str(e)}, f)
        return None

if __name__ == "__main__":
    port = int(sys.argv[1])
    prompt = sys.argv[2]
    output = sys.argv[3]
    query(port, prompt, output)
