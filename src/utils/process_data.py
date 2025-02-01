import logging
import os
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tika import parser


class EPUBProcessor:
    """
    A class for processing EPUB files, extracting text, and performing text preprocessing
    such as removing stopwords, URLs, and lemmatization.

    Attributes:
        cfg (dict): Configuration dictionary containing preprocessing settings.
        logger (logging.Logger, optional): Logger instance for logging messages.
        combined_text (str): The combined text from all EPUB files.
    """

    def __init__(self, cfg: dict, logger: logging.Logger = None) -> None:
        self.cfg = cfg
        self.logger = logger
        self.combined_text = ""

    def load_epub_text(self) -> None:
        """
        Initializes the EPUBProcessor with the given configuration and optional logger.

        Args:
            cfg (dict): Configuration dictionary containing preprocessing settings.
            logger (logging.Logger, optional): Logger instance for logging messages. Defaults to None.
        """
        books_dir = os.path.join(os.getcwd(), self.cfg["preprocessing"]["path"])

        if not os.path.isdir(books_dir):
            raise FileNotFoundError(f"Directory not found: {books_dir}")

        epub_files = [
            os.path.join(books_dir, book)
            for book in os.listdir(books_dir)
            if book.endswith(".epub")
        ]

        if not epub_files:
            raise FileNotFoundError(f"No epub files were found in {books_dir}")

        extracted_books = []

        for book in epub_files:
            book_name = os.path.splitext(os.path.basename(book))[0]

            if self.logger:
                self.logger.info(f"Reading {book_name}")

            extracted_text = parser.from_file(
                filename=book,
            )["content"]
            extracted_books.append(extracted_text.strip().lower())

        self.combined_text = "\n\n".join(extracted_books)

    def clean_data(self) -> str:
        """
        Cleans the extracted text by removing URLs, non-alphanumeric characters,
        stopwords, and applying lemmatization.

        The method performs the following steps:
        1. Removes URLs from the text.
        2. Removes non-alphanumeric characters (except spaces).
        3. Downloads necessary NLTK resources if not available.
        4. Tokenizes the text and removes stopwords.
        5. Applies lemmatization to the words.

        Returns:
            str: The cleaned and preprocessed text.

        Raises:
            ValueError: If `combined_text` is empty.
        """
        if not self.combined_text:
            raise ValueError("No text to clean, please check.")

        text_without_urls = re.sub(r"www.\S+", "", self.combined_text)
        text_alpha_numeric = re.sub(r"[^A-Za-z0-9\s]", "", text_without_urls)

        for model in self.cfg["preprocessing"]["nltk"]:
            nltk.download(model)

        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))
        words = word_tokenize(text_alpha_numeric)

        processed_text = " ".join(
            lemmatizer.lemmatize(word) for word in words if word not in stop_words
        )

        return processed_text
