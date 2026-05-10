from __future__ import annotations

import pandas as pd
import streamlit as st


def show_table(rows: list[dict], empty_message: str) -> None:
    if not rows:
        st.info(empty_message)
        return
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=360)
