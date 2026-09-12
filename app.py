import os
import urllib.parse

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Topic Explorer", page_icon="🔎", layout="centered")

# Groq hosts the open-weight "openai/gpt-oss-120b" model behind an
# OpenAI-compatible API, so we just point the OpenAI client at Groq's base URL.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "openai/gpt-oss-120b"


def get_client() -> OpenAI:
    """Build an OpenAI client pointed at Groq, using a key from secrets or env."""
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    if not api_key:
        st.error(
            "No API key found. Add GROQ_API_KEY to your Streamlit secrets "
            "(or as an environment variable) before running the app."
        )
        st.stop()
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def generate_content(client: OpenAI, topic: str) -> str:
    """Ask the model for a structured write-up on the given topic."""
    prompt = f"""You are a research assistant. A user wants to learn about: "{topic}"

Provide a well-organized response in Markdown with these sections:

## Overview
A clear, concise explanation of the topic (3-5 sentences).

## Key Points
4-6 bullet points covering the most important aspects.

## Recommended YouTube Searches
3-4 specific search phrases someone could paste into YouTube to find good videos.
Do not invent fake video titles or links, just strong search phrases.

## Recommended Websites / Resources
3-5 real, well-known websites or types of resources (official docs, Wikipedia,
reputable publications, etc.) relevant to this topic, each with a one-line note
on why it's useful.

Keep it factual. Do not make up specific URLs, video titles, or statistics you
are not confident about.
"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful, accurate research assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=1200,
    )
    return response.choices[0].message.content


def youtube_search_url(query: str) -> str:
    return "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)


# ---------- UI ----------

st.title("🔎 Topic Explorer")
st.caption(
    "Describe a topic and get an overview, key points, and places to learn "
    "more — powered by openai/gpt-oss-120b."
)

topic = st.text_input(
    "What topic do you want to explore?",
    placeholder="e.g. Quantum computing, French Revolution, Docker containers",
)

go = st.button("Explore", type="primary")

if go:
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        client = get_client()
        with st.spinner(f"Researching '{topic}'..."):
            try:
                content = generate_content(client, topic.strip())
            except Exception as e:
                st.error(f"Something went wrong calling the model: {e}")
                st.stop()

        st.markdown(content)

        st.divider()
        st.subheader("Quick link")
        st.markdown(f"[Search YouTube for '{topic}']({youtube_search_url(topic)})")

st.sidebar.header("About")
st.sidebar.write(
    "This app uses the **openai/gpt-oss-120b** model (served via Groq) to "
    "summarize a topic and suggest videos and resources to explore further."
)
st.sidebar.write("Add your Groq API key as `GROQ_API_KEY` in Streamlit secrets to run this app.")
