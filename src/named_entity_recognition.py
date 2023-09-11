import pandas as pd
import streamlit as st

from .hf_tasks import HuggingFaceTask, HuggingFaceTaskMixin


class NamedEntityRecognition(HuggingFaceTask, HuggingFaceTaskMixin):
    """Named entity recognition."""

    # Adapted from:
    # https://huggingface.co/dslim/bert-base-NER

    def __init__(self, name: str) -> None:
        super().__init__(name)

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
            text_lines = self.get_text(self.sample_input, self.max_lines)
            submit_button = st.form_submit_button(label="Submit")

            return submit_button, text_lines

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
            output = self.process_input(self.api_url, self.headers, text_lines, None)

            if output:
                self._process_output(output)
