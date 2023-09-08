import requests
import pandas as pd
import streamlit as st

from .hf_tasks import HuggingFaceTask


class NamedEntityRecognition(HuggingFaceTask):
    """Named entity recognition."""

    # Adapted from:
    # https://huggingface.co/dslim/bert-base-NER

    def __init__(self, name: str) -> None:
        super().__init__()

        self.name = name
        self.api_url = None
        self.max_lines = 5

        sample_text = [
            "Hanoi is the capital of Vietnam.",
            "Google was founded by Larry Page and Sergey Brin in Menlo Park, California.",
        ]
        new_line = "\n"
        self.sample_input = f"{new_line.join(map(str, sample_text))}"

    def setup(self):
        """Setup function."""
        st.title(self.name)

        st.write("Identity named entities in keyphrases on-the-fly.")

        st.markdown("### Input")
        checkpoint = st.text_input("Model checkpoint", "dslim/bert-base-NER")
        self.api_url = self.api_root + checkpoint
        st.write("API endpoint: ", self.api_url)

        st.caption("")

    def _get_input(self):
        with st.form(key="my_form"):
            input_text = st.text_area(
                "Enter input keyphrases", self.sample_input, height=150
            )

            text_lines = input_text.split("\n")  # a list of lines
            text_lines = list(dict.fromkeys(text_lines))  # remove dubplicates and empty
            text_lines = list(filter(None, text_lines))

            if len(text_lines) > self.max_lines:
                st.info(f"❄️  Only the first {self.max_lines} keyphrases are used.")
                text_lines = text_lines[: self.max_lines]

            submit_button = st.form_submit_button(label="Submit")
            return submit_button, text_lines

    def _process_input(self, text_lines):
        output = []
        for row in text_lines:
            payload = {
                "inputs": row,
                "options": {"wait_for_model": True},
            }

            try:
                response = requests.post(
                    self.api_url, headers=self.headers, json=payload, timeout=10
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

    def _process_output(self, output):
        st.markdown("### Output")

        # convert a list of lists to a multiindex dataframe
        output = {
            (okey, ikey): values
            for okey, ilist in enumerate(output)
            for ikey, values in enumerate(ilist)
        }
        output = pd.DataFrame.from_dict(data=output, orient="index")
        output = output[["word", "entity_group", "score", "start", "end"]]
        output.index.names = ["keyphrase_id", "entity_id"]
        output = output.reset_index()
        st.write(output)

    def run(self):
        """Main entry function for HF task."""
        submit_button, text_lines = self._get_input()

        if submit_button:
            output = self._process_input(text_lines)

            if output:
                self._process_output(output)
