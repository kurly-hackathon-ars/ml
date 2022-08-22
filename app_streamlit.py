import os
import pickle
import tempfile

import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from app import deps, service
from app.milvus import MilvusHelper

TEST_DATA_DIR = "./data"


def main():
    st.set_page_config(layout="wide")

    st.title("Kurly Recommend System Demo")
    selected_option = st.sidebar.selectbox(
        "Select View", ["View Data", "Recommend By Vector", "Recommend By Activity"]
    )

    st.sidebar.subheader("Upload data")
    uploaded_items = st.sidebar.file_uploader("Upload items", type=["csv"])
    uploaded_ratings = st.sidebar.file_uploader("Upload ratings", type=["csv"])

    if (
        st.sidebar.button("Upload", disabled=not uploaded_items or not uploaded_ratings)
        and uploaded_items
        and uploaded_ratings
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            items_fp = os.path.join(tmp_dir, "items.csv")
            ratings_fp = os.path.join(tmp_dir, "ratings.csv")

            with open(items_fp, mode="wb") as f:
                f.write(uploaded_items.read())

            with open(ratings_fp, mode="wb") as f:
                f.write(uploaded_ratings.read())

            deps.setup_sample_items(items_fp, ratings_fp)

    if st.sidebar.button("Upload sample dataset"):
        deps.setup_sample_items()

    st.sidebar.subheader("Commands")

    if st.sidebar.button("Drop items"):
        MilvusHelper.drop()

    if st.sidebar.button("Build items"):
        deps.insert_entities(deps.get_items())

    if selected_option == "View Data":
        view_data()
    elif selected_option == "Recommend By Vector":
        recommend_by_vector()
    elif selected_option == "Recommend By Activity":
        recommend_by_activity()
    else:
        ...


def view_data():
    st.dataframe(pd.DataFrame([item.dict() for item in deps.get_items()]))
    st.dataframe(pd.DataFrame([item.dict() for item in deps.get_activities()]))


def recommend_by_vector():
    st.subheader("Test")
    query = st.text_input("Query")

    if not query.strip():
        st.stop()

    st.dataframe(
        pd.DataFrame([each.dict() for each in service.recommend_by_vector(query)])
    )


def recommend_by_activity():

    # Load dataset from Movie...
    items = pd.read_csv(os.path.join(TEST_DATA_DIR, "movies.csv"))
    st.dataframe(items)
    ratings = pd.read_csv(os.path.join(TEST_DATA_DIR, "ratings.csv"))
    st.dataframe(ratings)

    item_count = max(items["itemId"])
    user_count = max(ratings["userId"])
    st.text(item_count)
    st.text(user_count)

    st.title("Dataset")
    dataset = ratings.pivot(index="itemId", columns="userId", values="rating")
    dataset.fillna(0, inplace=True)
    st.title("Dataset #1")
    st.dataframe(dataset)
    st.title("Dataset Target")
    st.dataframe(items[items["itemId"] == 68])
    st.dataframe(ratings[ratings["itemId"] == 68])
    st.dataframe(ratings[ratings["userId"] == 26])

    no_user_voted = ratings.groupby("itemId")["rating"].agg("count")
    no_movies_voted = ratings.groupby("userId")["rating"].agg("count")
    st.dataframe(no_movies_voted)  # movie 를 추천한 유저 수
    st.dataframe(no_user_voted)  # 유저가 추천한 영화수

    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    plt.scatter(no_user_voted.index, no_user_voted)
    plt.xlabel("itemId")  # TODO: Item id
    plt.ylabel("No. of users voted")  # TODO: No. of users saw
    st.pyplot(fig)

    # dataset: pd.DataFrame = dataset.loc[
    #     [idx for idx in no_user_voted[no_user_voted > 1000].index if idx in dataset.index],
    #     :,
    # ]
    # dataset: pd.DataFrame = dataset.loc[
    #     no_user_voted[no_user_voted > 10].index,
    #     :,
    # ]
    st.title("Dataset #2!")
    st.dataframe(dataset)

    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    plt.scatter(no_movies_voted.index, no_movies_voted)
    plt.xlabel("UserId")
    plt.ylabel("No. of votes by user")
    st.pyplot(fig)

    # dataset = dataset.loc[
    #     :,
    #     [
    #         idx
    #         for idx in no_movies_voted[no_movies_voted > 50].index
    #         if idx in dataset.columns
    #     ],
    # ]
    st.dataframe(dataset)
    dataset: pd.DataFrame
    csr_dataset = csr_matrix(dataset.values)
    st.text(csr_dataset)
    st.dataframe(dataset)
    dataset.reset_index(inplace=True)
    st.title("Final Dataset")
    st.dataframe(dataset)
    if os.path.exists("./model.pkl"):
        st.text("Load from model.pkl")
        with open("./model.pkl", mode="rb") as f:
            knn = pickle.load(f)
    else:
        st.text("Train from scratch")
        knn = NearestNeighbors(
            metric="cosine", algorithm="brute", n_neighbors=20, n_jobs=-1
        )
        knn.fit(csr_dataset)
        s = pickle.dumps(knn)
        with open("./model2.pkl", "wb") as f:
            f.write(s)

    n_movies_to_recommend = 20
    movie_list = items[items["title"].str.contains("Toy Story")]
    if not len(movie_list):
        st.stop()

    st.dataframe(movie_list)
    movie_idx = movie_list.iloc[0]["itemId"]
    movie_idx = dataset[dataset["itemId"] == movie_idx]
    st.dataframe(movie_idx)
    movie_idx = movie_idx.index[0]
    st.text(movie_idx)

    distances, indices = knn.kneighbors(
        csr_dataset[movie_idx], n_neighbors=n_movies_to_recommend + 1
    )
    st.text(distances)
    st.text(indices)
    rec_movie_indices = sorted(
        list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
        key=lambda x: x[1],
        reverse=True,
    )[:0:-1]
    recommend_frame = []
    for val in rec_movie_indices:
        movie_idx = dataset.iloc[val[0]]["itemId"]
        st.text(movie_idx)
        idx = items[items["itemId"] == movie_idx].index
        st.text(f"idx ==> {idx}")
        recommend_frame.append(
            {"Title": items.iloc[idx]["title"].values[0], "Distance": val[1]}
        )
    df = pd.DataFrame(recommend_frame, index=range(1, n_movies_to_recommend + 1))
    st.dataframe(df)


if __name__ == "__main__":
    main()
