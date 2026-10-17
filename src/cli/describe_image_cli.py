import argparse
import mimetypes
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64


def main():
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    parser.add_argument("--query")
    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open("data/0LmrJI7-194x260 (1).jpeg" , "rb") as f:
        img = f.read()

    client  = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    system_prompt = '''Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary'''
    
    data_url = f"data:{mime};base64,{base64.b64encode(img).decode()}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt.strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": args.query.strip()},
            ],
        }
    ]
    response = client.chat.completions.create(messages=messages, model="openrouter/free")
    content = response.choices[0].message.content
    print(f"Rewritten query: {content.strip()}")
    if response.usage is not None:
        print(f"Total tokens:    {response.usage.total_tokens}")


if __name__=="__main__":
    main()
