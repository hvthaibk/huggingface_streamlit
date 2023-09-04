# Import Streamlit and Pandas
import streamlit as st
from streamlit_option_menu import option_menu

from src.zero_shot_text_classification import ZeroShotTextClassification


def setup():
    """Streamlit setup."""

    task_dict = {
        "Zero-Shot Text Classification": ZeroShotTextClassification(),
        "Others": None,
    }

    # The code below is to control the layout width, title and logo of the app.
    if "widen" not in st.session_state:
        layout = "centered"
    else:
        layout = "wide" if st.session_state.widen else "centered"

    st.set_page_config(layout=layout, page_title="HuggingFace serving", page_icon="🤗")

    # Set up session state so app interactions don't reset the app.
    if "valid_inputs_received" not in st.session_state:
        st.session_state["valid_inputs_received"] = False

    st.sidebar.image("data/logo.png")
    st.write("")

    # The code below is to display the menu bar.
    with st.sidebar:
        st.session_state["task"] = option_menu("", list(task_dict.keys()))

    # The block of code below is to display information about Streamlit.
    st.sidebar.markdown("---")
    st.sidebar.header("About")

    st.sidebar.markdown(
        """App created by [Thai Hoang](https://www.linkedin.com/in/thai-hoang-b7012637/) """
        """using [Streamlit](https://streamlit.io/) and """
        """[HuggingFace](https://huggingface.co/inference-api)."""
    )

    return task_dict[st.session_state["task"]]


if __name__ == "__main__":
    task = setup()

    if task is not None:
        task.setup()
        task.run()
    else:
        st.title("To Be Implemented!")
