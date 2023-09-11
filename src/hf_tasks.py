from abc import ABC, abstractmethod

import requests
import streamlit as st


class HuggingFaceTask(ABC):
    """Abstract class for HuggingFace tasks."""

    def __init__(self, name):
        self.name = name
        self.api_token = st.secrets["HF_API_KEY"]
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
        self.api_root = "https://api-inference.huggingface.co/models/"
        self.api_url = None

    @abstractmethod
    def setup(self):
        """Abstract method."""

    @abstractmethod
    def run(self):
        """Abstract method."""


class HuggingFaceTaskMixin:
    """Mixin class."""

    def process_input(self, api_url, headers, text_lines, labels=None):
        """Query HF API."""
        output = []
        for row in text_lines:
            if labels is not None:
                payload = {
                    "inputs": row,
                    "parameters": {"candidate_labels": labels},
                    "options": {"wait_for_model": True},
                }
            else:
                payload = {
                    "inputs": row,
                    "options": {"wait_for_model": True},
                }

            try:
                response = requests.post(
                    api_url, headers=headers, json=payload, timeout=10
                )
            except requests.exceptions.Timeout:
                st.error("HTTP connection time out. Please try again!", icon="🚨")
                return []

            if response.status_code != 200:
                st.error(f"Query error code: {response.status_code}", icon="🚨")
                st.error(response.text, icon="🚨")
                return []

            output.append(response.json())

        st.success("Finished querying HuggingFace API successfully!", icon="✅")
        st.caption("")

        return output

    def get_text(self, sample_input, max_lines):
        """Get input text."""
        input_text = st.text_area("Enter input keyphrases", sample_input, height=150)

        text_lines = input_text.split("\n")  # a list of lines
        text_lines = list(dict.fromkeys(text_lines))  # remove dubplicates and empty
        text_lines = list(filter(None, text_lines))

        if len(text_lines) > max_lines:
            st.info(f"❄️  Only the first {max_lines} keyphrases are used.")
            text_lines = text_lines[:max_lines]

        return text_lines
