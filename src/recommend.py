import argparse
import os

import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from utils import *


def recommend(
    username: str,
    how: str = "tfidf",
    root: os.PathLike = "../data",
    nr=10,
    nt=20,
) -> pd.DataFrame:
    """
    Recommends the top n posts for a given profile

    Parameters
    ----------
    username : str
        Username of the profile to recommend posts for
    vectorizer : str, optional
        Vectorizer to use, by default "tfidf"
    date : Union[str, datetime], optional
        Date to recommend posts for, by default datetime.today()
    root : os.PathLike, optional
        Path to the folder where the posts are stored, by default "../data"
    nr : int, optional
        Number of posts to recommend, by default 10
    nt : int, optional
        Number of trends to consider, by default 20

    Returns
    -------
    pd.DataFrame
        Dataframe containing the top n posts
    """
    download_profile(username, root=root)
    vectorizer = fit_vectorizer(username, root=root, how=how)
    trends = get_trends(count=nt)
    posts = get_posts(username, root=root)

    tqdm.pandas(desc="Embed captions")
    posts["embeddings"] = posts["caption"].progress_apply(vectorizer.transform)

    tqdm.pandas(desc="Embed trends")
    trends["embeddings"] = trends["text"].progress_apply(vectorizer.transform)

    recent_embedding = posts.iloc[-1]["embeddings"]

    trends["similarity"] = trends["embeddings"].apply(
        lambda x: cosine_similarity(recent_embedding, x)[0][0]
    )
    trends.sort_values("similarity", ascending=False, inplace=True)

    return trends.head(nr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommend posts for a given profile")
    parser.add_argument(
        "-u",
        "--username",
        type=str,
        help="Username of the profile to recommend posts for",
    )
    parser.add_argument(
        "-nr",
        "--num_recommend",
        type=int,
        default=10,
        help="Number of posts to recommend",
    )
    parser.add_argument(
        "-nt", "--num_trends", type=int, default=20, help="Number of trends to consider"
    )
    parser.add_argument(
        "-r",
        "--root",
        type=str,
        default="../data",
        help="Path to the folder where the posts are stored",
    )
    parser.add_argument(
        "-v",
        "--vectorizer",
        type=str,
        default="tfidf",
        help="Vectorizer to use",
        choices=["tfidf", "bow", "bert"],
    )
    args = parser.parse_args()

    print(
        recommend(
            args.username,
            nr=args.num_recommend,
            nt=args.num_trends,
            root=args.root,
            how=args.vectorizer,
        )
    )
