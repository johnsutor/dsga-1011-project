import json
import os
import re
from datetime import datetime
from typing import List, Tuple, Union

import instaloader
import numpy as np
import pandas as pd
import requests
import torch
from dateutil import parser as dateparser
from newspaper import Article
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def download_profile(
    usernames: Union[str, List[str]], root: os.PathLike = "../data", **kwargs
):
    """
    Downloads all posts of a given publicly-accessible profile.
    Does not download images or videos, only metadata

    Parameters
    ----------
    usernames : str, List[str]
        Username(s) of the profile to download
    root : str
        Path to the folder where the posts will be stored
    """

    loader = instaloader.Instaloader(
        dirname_pattern=os.path.join(root, "{profile}"),
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=True,
        compress_json=False,
    )

    if isinstance(usernames, str):
        usernames = [usernames]

    profiles = [
        instaloader.Profile.from_username(loader.context, username)
        for username in usernames
    ]

    latest_stamps = instaloader.LatestStamps(os.path.join(root, "latest_timestamp.txt"))

    loader.download_profiles(
        profiles,
        fast_update=True,
        profile_pic=False,
        igtv=False,
        latest_stamps=latest_stamps,
        stories=False,
        highlights=False,
        tagged=False,
    )


class Vectorizer:
    """
    A class for vectorizing text inputs
    """

    def __init__(self, how: str = "tfidf", ngram_range: Tuple[int, int] = (1, 1)):
        """
        Initializes the vectorizer

        Parameters
        ----------
        how : str
            How to vectorize the content. Can be either "tfidf", "bow" (bag of words), or "bert"
        ngram_range : Tuple[int, int]
            Range of ngrams to use for tfidf or count vectorization
        """
        self.how = how
        self.ngram_range = ngram_range
        if self.how in ["tfidf", "bow"]:
            self.vectorizer = (
                TfidfVectorizer if self.how == "tfidf" else CountVectorizer
            )(
                input="filename",
                strip_accents="unicode",
                ngram_range=self.ngram_range,
            )
        elif self.how == "bert":
            self.vectorizer = AutoModel.from_pretrained("vinai/bertweet-base")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "vinai/bertweet-base", use_fast=False
            )
        else:
            raise NotImplementedError("how must be either tfidf, bow or bert")
        self.trained = False

    def fit_transform(
        self, text_files: List[os.PathLike], batch_size: int = 8
    ) -> np.ndarray:
        """
        Fits the vectorizer to the given text files, and returns the vectors

        Parameters
        ----------
        text_files : List[os.PathLike]
            List of paths to the text files to fit the vectorizer to
        batch_size : int
            Batch size for bert vectorization

        Returns
        -------
        np.ndarray
            Array containing the vectors
        """
        if self.how in ["tfidf", "bow"]:
            vectors = self.vectorizer.fit_transform(text_files).toarray()
            self.vectorizer.input = "content"
        elif self.how == "bert":
            all_embeddings = []

            for i in tqdm(range(0, len(text_files), 8), desc="Bert vectorization"):
                batch_contents = [
                    open(file_path, "r", encoding="utf-8").read()
                    for file_path in text_files[i : i + 8]
                    if os.path.exists(file_path)
                ]

                tokens = self.tokenizer(
                    batch_contents, padding=True, truncation=True, return_tensors="pt"
                )

                with torch.no_grad():
                    outputs = self.vectorizer(**tokens)
                    embeddings = [
                        o.numpy() for o in outputs.last_hidden_state
                    ]  # This contains the embeddings for each token in the input
                    all_embeddings.extend(embeddings)

            vectors = np.array(all_embeddings)
        else:
            raise NotImplementedError("how must be either tfidf, bow or bert")

        self.trained = True
        return vectors

    def fit(self, text_files: List[os.PathLike]) -> None:
        """
        Fits the vectorizer to the given text files

        Parameters
        ----------
        text_files : List[os.PathLike]
            List of paths to the text files to fit the vectorizer to
        """
        if self.how in ["tfidf", "bow"]:
            self.vectorizer.fit(text_files)
            self.vectorizer.input = "content"
        elif self.how == "bert":
            pass
        else:
            raise NotImplementedError("how must be either tfidf, bow or bert")

        self.trained = True

    def transform(self, text: str) -> np.ndarray:
        """
        Transforms the given text into a vector

        Parameters
        ----------
        text : str
            Text to transform

        Returns
        -------
        np.ndarray
            Array containing the vector
        """
        if not self.trained:
            raise ValueError("Vectorizer must be trained first")
        if self.how in ["tfidf", "bow"]:
            vector = self.vectorizer.transform([text])
        elif self.how == "bert":
            tokens = self.tokenizer(
                [text], padding=True, truncation=True, return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self.vectorizer(**tokens)
                vector = outputs.last_hidden_state[0].numpy()

        return vector


def fit_vectorizer(
    username: str,
    root: str = "../data",
    how: Union[str, "tfidf", "bow", "bert"] = "tfidf",
    ngram_range: Tuple[int, int] = (1, 1),
    fit_before: Union[str, datetime] = datetime.today(),
    batch_size: int = 8,
) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Vectorizes the content of a given profile. Assumes the download
    has already been done, and the directory is full of posts. Directory
    is expected to contain a folder named after the profile, which contains
    the text files and json metadata files for each post.

    Parameters
    ----------
    username : str
        Username of the profile to vectorize
    root : str
        Path to the folder where the posts are stored
    how : str
        How to vectorize the content. Can be either "tfidf", "bow" (bag of words), or "bert"
    return_vectorizer : bool
        Whether to return the vectorizer object or not
    ngram_range : Tuple[int, int]
        Range of ngrams to use for tfidf or count vectorization
    fit_before : Union[str, datetime]
        Date to fit the vectorizer before. Can be either a datetime object
    batch_size : int
        Batch size for bert vectorization

    Returns
    -------
    Tuple[np.ndarray, Union[TfidfVectorizer, None]]
        Tuple containing the vectors and the vectorizer object if
        return_vectorizer is True, None otherwise
    """
    profile_path = os.path.join(root, username)
    text_files = [
        os.path.join(profile_path, file)
        for file in os.listdir(profile_path)
        if file.endswith(".txt")
        and datetime.fromtimestamp(os.path.getmtime(os.path.join(profile_path, file)))
        < fit_before
    ]

    vectorizer = Vectorizer(how=how, ngram_range=ngram_range)
    vectorizer.fit(text_files)

    # Necessary to work with raw text inputs after training on documents
    return vectorizer


def remove_control_characters(html: str) -> str:
    """
    Strip invalid XML characters that `lxml` cannot parse.
    See: https://github.com/html5lib/html5lib-python/issues/96

    The XML 1.0 spec defines the valid character range as:
    Char ::= #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]

    We can instead match the invalid characters by inverting that range into:
    InvalidChar ::= #xb | #xc | #xFFFE | #xFFFF | [#x0-#x8] | [#xe-#x1F] | [#xD800-#xDFFF]

    Sources:
    https://www.w3.org/TR/REC-xml/#charsets,
    https://lsimons.wordpress.com/2011/03/17/stripping-illegal-characters-out-of-xml-in-python/

    Parameters
    ----------
    html : str
        HTML string to clean

    Returns
    -------
    str
        Cleaned HTML string
    """

    def strip_illegal_xml_characters(s, default, base=10):
        # Compare the "invalid XML character range" numerically
        n = int(s, base)
        if (
            n in (0xB, 0xC, 0xFFFE, 0xFFFF)
            or 0x0 <= n <= 0x8
            or 0xE <= n <= 0x1F
            or 0xD800 <= n <= 0xDFFF
        ):
            return ""
        return default

    # We encode all non-ascii characters to XML char-refs, so for example "💖" becomes: "&#x1F496;"
    # Otherwise we'd remove emojis by mistake on narrow-unicode builds of Python
    html = html.encode("ascii", "xmlcharrefreplace").decode("utf-8")
    html = re.sub(
        r"&#(\d+);?",
        lambda c: strip_illegal_xml_characters(c.group(1), c.group(0)),
        html,
    )
    html = re.sub(
        r"&#[xX]([0-9a-fA-F]+);?",
        lambda c: strip_illegal_xml_characters(c.group(1), c.group(0), base=16),
        html,
    )
    # A regex matching the "invalid XML character range"
    html = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1F\uD800-\uDFFF\uFFFE\uFFFF]").sub(
        "", html
    )
    return html


def get_article(url: str) -> str:
    """
    Obtains the likely text from an article based on the newspaper library

    Parameters
    ----------
    url: str
        URL of the article to fetch

    Returns
    -------
    str
        Most likely article text
    """

    text = ""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4644.45 Safari/537.36",
            "Connection": "keep-alive",
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        article = Article(url)
        article.download(input_html=remove_control_characters(response.text))
        article.parse()
        text = article.text

    except requests.exceptions.HTTPError as http_err:
        pass

    return text


def get_trends(
    hl="en-US",
    geo="US",
    tz=360,
    count=20,
    date: Union[datetime, str] = datetime.today(),
) -> pd.DataFrame:
    """
    Fetch Google Trends realtime data

    Parameters
    ----------
    hl : str
        Language
    geo : str
        Country
    tz : int
        Timezone
    count : int
        Number of results
    date : Union[datetime, str]
        Date to fetch data from. Can be either a datetime object or a string
        in the format YYYY-MM-DD (fuzzy parsing is enabled, but not recommended)

    Returns
    -------
    pd.DataFrame
        DataFrame containing the results
    """
    if isinstance(date, str):
        date = dateparser.parse(date, fuzzy=True)

    response = requests.get(
        "https://trends.google.com/trends/api/dailytrends",
        params={
            "hl": hl,
            "tz": tz,
            "ed": date.strftime("%Y%m%d"),
            "geo": geo,
            "ns": count,
        },
    )

    response.raise_for_status()
    if response.status_code != 204:
        data = response.text.split(")]}',\n")[1]
        data = json.loads(data)["default"]["trendingSearchesDays"][0][
            "trendingSearches"
        ]

    dfs = pd.concat(
        [pd.DataFrame(trend["articles"]) for trend in data], ignore_index=True
    )

    dfs["text"] = dfs["url"].apply(get_article)

    return dfs


def read_metadata_json(fp: os.PathLike):
    """
    Reads a json file containing the metadata of an Instagram post
    and returns a dictionary with the relevant information. Expects
    the file to be named as YYYY-MM-DD_HH-MM-SS_UTC.json, which is
    the default for instaloader.

    Parameters
    ----------
    fp : os.PathLike
        Path to the json file

    Returns
    -------
    dict
        Dictionary containing the relevant metadata
    """
    with open(fp, "r") as f:
        metadata = json.loads(f.read())["node"]

        dt = datetime.strptime(os.path.basename(fp), "%Y-%m-%d_%H-%M-%S_UTC.json")

        clean_metadata = {
            "dt": dt,
            "likes": metadata["edge_media_preview_like"]["count"],
            "comments": metadata["edge_media_to_comment"]["count"],
            "caption": metadata["edge_media_to_caption"]["edges"][0]["node"]["text"]
            if metadata["edge_media_to_caption"]["edges"]
            else "",
            "comments_disabled": metadata["comments_disabled"],
            "is_video": metadata["is_video"],
            "tagged_users": metadata["edge_media_to_tagged_user"],
        }

        return clean_metadata


def get_posts(
    username: str,
    root: os.PathLike = "../data",
) -> pd.DataFrame:
    """
    Reads all the posts of a given profile and returns a DataFrame
    with the relevant information

    Parameters
    ----------
    username : str
        Username of the profile to read
    root : os.PathLike
        Path to the folder where the posts are stored

    Returns
    -------
    pd.DataFrame
        DataFrame containing the relevant metadata
    """
    profile_path = os.path.join(root, username)
    json_files = [
        os.path.join(profile_path, file)
        for file in os.listdir(profile_path)
        if file.endswith("UTC.json")
    ]
    metadata = [read_metadata_json(file) for file in json_files]
    df = pd.DataFrame(metadata)
    df.set_index("dt", inplace=True)
    df.sort_index(inplace=True)
    return df


def ema(data: pd.Series, alpha: float = 0.99) -> pd.Series:
    """
    Calculates the exponential moving average of a given series

    Parameters
    ----------
    data : pd.Series
        Series to calculate the ema for
    alpha : float
        Alpha parameter for the ema calculation

    Returns
    -------
    pd.Series
        Series containing the ema values
    """
    ema = []
    ema_value = None

    for value in data:
        if ema_value is None:
            ema_value = value
        else:
            ema_value = (value - ema_value) * alpha + ema_value
        ema.append(ema_value)

    return ema
