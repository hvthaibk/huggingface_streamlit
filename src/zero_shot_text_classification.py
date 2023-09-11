from streamlit_tags import st_tags

from st_aggrid import AgGrid, GridUpdateMode, DataReturnMode
from st_aggrid.grid_options_builder import GridOptionsBuilder

import pandas as pd
import streamlit as st

from .hf_tasks import HuggingFaceTask, HuggingFaceTaskMixin


class ZeroShotTextClassification(HuggingFaceTask, HuggingFaceTaskMixin):
    """Zero-Shot Text Classification."""

    # Adapted from:
    # https://www.charlywargnier.com/post/how-to-create-a-zero-shot-learning-text-classifier-using-hugging-face-and-streamlit

    def __init__(self, name: str) -> None:
        super().__init__(name)

        self.max_lines = 5

        sample_text = [
            "I want to buy something in this store.",
            "This book is interesting.",
            "My internet connection is terribly slow.",
        ]
        new_line = "\n"
        self.sample_input = f"{new_line.join(map(str, sample_text))}"

    def setup(self):
        """Setup function."""
        st.title(self.name)

        st.write("Classify keyphrases on-the-fly. No ML training needed!")

        st.markdown("### Input")
        checkpoint = st.text_input("Model checkpoint", "valhalla/distilbart-mnli-12-3")
        self.api_url = self.api_root + checkpoint
        st.write("API endpoint: ", self.api_url)

        st.caption("")

    def _get_input(self):
        with st.form(key="my_form"):
            labels = st_tags(
                label="Create labels (e.g. `Positive`, `Negative`, `Neutral`)",
                text="Add labels - 3 max",
                value=["Positive", "Negative", "Neutral"],
                maxtags=3,
            )

            text_lines = self.get_text(self.sample_input, self.max_lines)
            submit_button = st.form_submit_button(label="Submit")

            return submit_button, text_lines, labels

    def _process_output(self, output):
        st.markdown("### Output")
        st.checkbox(
            "Widen layout",
            key="widen",
            help="Tick this box to toggle the layout to 'Wide' mode",
        )

        output = pd.DataFrame(data=output)

        # convert the decimals to percentages
        scores = [[f"{x:.2%}" for x in row] for row in output["scores"]]
        output["classification scores"] = scores
        output.drop("scores", inplace=True, axis=1)

        output.rename(columns={"sequence": "keyphrase"}, inplace=True)

        # for ag-grid
        go_builder = GridOptionsBuilder.from_dataframe(output)
        go_builder.configure_default_column(
            enablePivot=True, enableValue=True, enableRowGroup=True
        )
        go_builder.configure_selection(selection_mode="multiple", use_checkbox=True)
        go_builder.configure_side_bar()
        grid_opts = go_builder.build()

        _ = AgGrid(
            output,
            gridOptions=grid_opts,
            enable_enterprise_modules=True,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            height=300,
            fit_columns_on_grid_load=False,
            configure_side_bar=True,
        )

    def run(self):
        """Main entry function for HF task."""
        submit_button, text_lines, labels = self._get_input()

        if not submit_button and not st.session_state.valid_inputs_received:
            st.stop()

        elif submit_button and not text_lines:
            st.warning("❄️ There is no keyphrases to classify")
            st.session_state.valid_inputs_received = False
            st.stop()

        elif submit_button and not labels:
            st.warning("❄️ You have not added any labels, please add some! ")
            st.session_state.valid_inputs_received = False
            st.stop()

        elif submit_button and len(labels) == 1:
            st.warning("❄️ At least two labels are requierd for classification")
            st.session_state.valid_inputs_received = False
            st.stop()

        elif submit_button or st.session_state.valid_inputs_received:
            if submit_button:
                st.session_state.valid_inputs_received = True

            output = self.process_input(self.api_url, self.headers, text_lines, labels)
            if output:
                self._process_output(output)
