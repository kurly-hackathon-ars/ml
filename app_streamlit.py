import csv
import os
import tempfile

import pandas as pd
import streamlit as st

from app import deps, service, vector
from app.milvus import MilvusHelper

TEST_DATA_DIR = "./data"


def main():
    st.set_page_config(layout="wide")

    st.title("Kurly Recommend System Demo")
    selected_option = st.sidebar.selectbox(
        "Select View",
        [
            "View Data",
            "Recommend By Vector",
        ],
    )
    ############################################################################
    # Upload Data
    ############################################################################
    st.sidebar.subheader("Upload data")
    uploaded_items = st.sidebar.file_uploader("Upload items", type=["csv"])

    if st.sidebar.button("Upload", disabled=not uploaded_items) and uploaded_items:
        with tempfile.TemporaryDirectory() as tmp_dir:
            items_fp = os.path.join(tmp_dir, "items.csv")

            with open(items_fp, mode="wb") as f:
                f.write(uploaded_items.read())

            deps.setup_sample_items_from_csv(items_fp)

    uploaded_filter_dictionary = st.sidebar.file_uploader(
        "Upload filter keyword dictionary", type=["csv"]
    )

    if (
        st.sidebar.button(
            "Upload filter keyword dictionary", disabled=not uploaded_filter_dictionary
        )
        and uploaded_filter_dictionary
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fp = os.path.join(tmp_dir, "file.csv")
            with open(fp, mode="wb") as f:
                f.write(uploaded_filter_dictionary.read())

            df = pd.read_csv(fp)
            for _, row in df.iterrows():
                deps.upsert_item_filter_dictionary(row["keyword"])

            st.info(f"Uploaded {df.shape[0]} filter keywords")

    mysql_limit = st.sidebar.number_input("Limit", value=200, step=1)
    if st.sidebar.button("Load sample dataset from backend"):
        deps.setup_sample_items_from_mysql("kurly_products", int(mysql_limit or 200))

    st.sidebar.subheader("App Commands")
    if st.sidebar.button("Drop items (Disabled for demo)", disabled=True):
        MilvusHelper.drop()

    if st.sidebar.button("Build items"):
        service.build_items()

    if selected_option == "View Data":
        view_data()
    elif selected_option == "Recommend By Vector":
        recommend_by_vector()
    else:
        raise ValueError("Unknown selected_option: %s", selected_option)


def view_data():
    st.subheader("Items")
    st.dataframe(pd.DataFrame([item.dict() for item in deps.get_items()]))

    items = deps.get_items()
    categories = set()
    for item in items:
        categories.add(item.category)

    st.subheader("Generate filter keywords for vector search build")
    if st.button("Generate"):
        keywords = service.generate_item_filter_keywords_from_items()
        with tempfile.TemporaryDirectory() as tmp_dir:
            fp = os.path.join(tmp_dir, "download.csv")
            with open(fp, mode="w", encoding="utf8") as f:
                writer = csv.DictWriter(f, ["keyword"])
                writer.writeheader()
                writer.writerows([{"keyword": k} for k in keywords])

            with open(fp) as f:
                st.download_button(
                    "Download Generated Filter Keywords",
                    f,
                    "filter-dictionary-generated.csv",
                )


def recommend_by_vector():
    st.subheader("Train")
    uploaded_training = st.file_uploader("Training Data", type=["csv"])

    st.subheader("Filter Keyword Dictionary")
    filter_keyword = st.text_input("Add filter keyword to dictionary").strip()
    if filter_keyword:
        deps.upsert_item_filter_dictionary(filter_keyword)
        st.info(f"{filter_keyword} is added to dictionary.")

    dicts = deps.get_item_filter_dictionaries()
    dicts_df = pd.DataFrame([each.dict() for each in dicts])
    st.dataframe(dicts_df)
    with tempfile.TemporaryDirectory() as tmp_dir:
        fp = os.path.join(tmp_dir, "data.csv")
        dicts_df.to_csv(fp, index=False)
        with open(fp) as f:
            st.download_button(
                "Download ItemFilterDictionary", f, "filter-dictionary.csv"
            )

    if uploaded_training and st.button("Start Train", disabled=True):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fp = os.path.join(tmp_dir, "data.csv")
            with open(fp, mode="wb") as f:
                f.write(uploaded_training.read())

            # TODO Train

    st.subheader("Test")
    query = st.text_input("Query")

    if not query.strip():
        st.stop()

    st.dataframe(
        pd.DataFrame([each.dict() for each in service.recommend_by_vector(query)])
    )


if __name__ == "__main__":
    main()
