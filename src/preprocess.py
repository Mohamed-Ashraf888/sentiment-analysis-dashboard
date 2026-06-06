import re
import string
from typing import Iterable

STOPWORDS = set("""
a an the and or but if while with without of to in on at for from by about as into like through after over between out against during before under around among
is am are was were be been being have has had do does did can could should would may might must will just not no nor so than too very
this that these those i you he she it we they me him her us them my your his its our their mine yours ours theirs
""".split())


def clean_text(text: str) -> str:
    """Clean review text: lowercase, remove URLs, numbers, punctuation, and extra spaces."""
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS and len(word) > 1]
    return " ".join(tokens)


def preprocess_text(text: str) -> str:
    return remove_stopwords(clean_text(text))


def preprocess_series(texts: Iterable[str]):
    return [preprocess_text(t) for t in texts]
