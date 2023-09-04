from abc import ABC, abstractmethod

import streamlit as st


class HuggingFaceTask(ABC):
    """Abstract class for HuggingFace tasks."""

    def __init__(self):
        self.api_token = st.secrets["API_KEY"]
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
        self.api_root = "https://api-inference.huggingface.co/models/"
        self.api_url = None

    @abstractmethod
    def setup(self):
        """Abstract method."""

    @abstractmethod
    def run(self):
        """Abstract method."""
