"""
Gradio app.

UI Features:
- Chat interface for Q&A
- A "Debug" accordion so you can see:
  - retrieved chunks + distances
  - reranked chunks used in final answer
  - gating reason

This helps you verify the system is retrieving the right info.
"""

from dotenv import load_dotenv
load_dotenv()

import json
import os
from pathlib import Path

import gradio as gr

from core.rag_pipeline import run_rag


def ensure_vectordb():
    """
    On Hugging Face Spaces, the container can start fresh.
    If vectordb doesn't exist, build it from kb/ automatically.
    """
    if not Path("vectordb").exists():
        os.system("python ingest.py")


def format_debug(debug: dict) -> str:
    """
    Convert the debug dictionary into formatted, readable JSON in the UI.
    """
    return json.dumps(debug, indent=2, ensure_ascii=False)


def chat_fn(message, history):
    """
    Process user messages and return formatted responses.
    """
    result = run_rag(message)
    answer = result["answer"]
    sources = result["sources"]
    debug = result["debug"]

    sources_md = (
        "\n".join([f"- {s}" for s in sources]) if sources else "_No sources used._"
    )
    final_md = f"{answer}\n\n---\n**Sources used:**\n{sources_md}"

    return final_md, format_debug(debug)


with gr.Blocks(title="HealthierYou (RAG)") as demo:
    ensure_vectordb()

    gr.Markdown(
        "# HealthierYou Assistant (RAG)\nAsk questions about the internal knowledge base."
    )

    with gr.Row():
        with gr.Column(scale=2): # scale=2 means column takes 2/3 of the width
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Here to assist you. Ask a question")
            send = gr.Button("Send")

        with gr.Column(scale=1): # scale=1 means column takes 1/3 of the width
            with gr.Accordion("Debug (retrieval + reranking)", open=False): # Collapsible section (like a dropdown)
                debug_box = gr.Textbox(lines=25, label="Debug JSON")

    def respond(user_message, chat_history):
        """
        Handle user interactions and update the UI.
        
        This function:
        1. Gets the response from chat_fn (which runs RAG)
        2. Appends user message to chat history
        3. Appends assistant response to chat history
        4. Returns: empty string (clears input), updated history, debug info
        """
        response, debug_text = chat_fn(user_message, chat_history)
        
        # Append user message
        chat_history.append(
            {"role": "user", "content": user_message}
        )

        # Append assistant response
        chat_history.append(
            {"role": "assistant", "content": response}
        )

        # Return 3 values to update 3 UI components
        return "", chat_history, debug_text

    send.click(
        respond, 
        inputs=[msg, chatbot], 
        outputs=[msg, chatbot, debug_box]
    )
    msg.submit(
        respond, 
        inputs=[msg, chatbot], 
        outputs=[msg, chatbot, debug_box]
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Glass())
