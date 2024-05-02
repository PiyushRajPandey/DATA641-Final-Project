import codecs
import gzip
import json
import re

import jsonlines
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split


def csv_to_jsonl_gz(file, destination):

    text_col = None

    if "personality" in file:
        text_col = "STATUS"
    else:
        text_col = "TEXT"

    # Read CSV file
    df = pd.read_csv(file, encoding="latin1")

    # Select required columns
    df_selected = df[["#AUTHID", text_col, "cNEU"]]

    # Write to JSONL file
    with gzip.open(destination, "wt") as jsonl_file:
        writer = jsonlines.Writer(jsonl_file)

        for index, row in df_selected.iterrows():
            data = {
                "#AUTHID": row["#AUTHID"],
                "STATUS": row[text_col],
                "cNEU": row["cNEU"]
            }

            writer.write(data)

        print("file processed successfully")


def read_and_clean_lines(infile, verbatim):
    statuses = []
    labels = []

    with gzip.open(infile, 'rt') as f:
        # for line in tqdm(f):
        for line in f:
            data = json.loads(line)
            text = re.sub(r'\s+', ' ', data["STATUS"])

            statuses.append(text)
            labels.append(data["cNEU"])

    if verbatim:
        print("Read {} documents and labels".format(len(statuses)))
        print("Read {} labels".format(len(labels)))

    return statuses, labels


def split_training_set(lines, labels, test_size, random_seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        lines, labels,
        test_size=test_size, random_state=random_seed
    )

    return X_train, X_test, y_train, y_test


# TODO: experiment with different stop word modules
def load_stopwords(filename, words_list):
    if words_list == "standard":

        # NLTK
        # nltk.download('stopwords')
        # stopwords = list(nltk.corpus.stopwords.words('english'))

        # SpaCy
        # nlp = spacy.load("en_core_web_sm")
        # stopwords = nlp.Defaults.stop_words

        # SciKit-Learn
        stopwords = list(ENGLISH_STOP_WORDS)

        return stopwords

    elif words_list == "custom":
        stopwords = []
        with codecs.open(filename, 'r', encoding='ascii', errors='ignore') as fp:
            stopwords = fp.read().split('\n')
        return list(stopwords)


def remove_stop_words(stop_words, sentences):
    cleaned_sentences = []
    for sentence in sentences:
        words = sentence.split()
        cleaned_words = [word for word in words if word.lower() not in stop_words]
        cleaned_sentence = ' '.join(cleaned_words)
        cleaned_sentences.append(cleaned_sentence)
    return cleaned_sentences


def words_with_apostrophe(sentences):
    apostrophe_words = []
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            if "'" in word:
                apostrophe_words.append(word)
    return list(set(apostrophe_words))


def preprocess_sentences(sentences):
    preprocessed_sentences = []
    for sentence in sentences:
        # Lowercase every word
        sentence = sentence.lower()

        # Reduce multiple exclamation and question marks to just one
        sentence = re.sub(r'(\!{2,}|\?{2,}|\.{2,})', r'\1', sentence)

        # Automatically add a space after each punctuation
        sentence = re.sub(r'([.!?])', r'\1 ', sentence)

        # Remove all "*" characters
        sentence = sentence.replace('*', ' ')

        # Remove all double and single quotes
        sentence = sentence.replace('"', ' ')

        # Remove dashes '-'
        sentence = sentence.replace('-', ' ')

        sentence = sentence.replace(',', ' ')

        # Remove parentheses, brackets, and braces
        sentence = re.sub(r'[\(\)\[\]\{\}<>]', ' ', sentence)

        preprocessed_sentences.append(sentence)
    return preprocessed_sentences


def expand_contractions(sentence):
    contractions_dict = {
        "here's": " here is ",
        "that's": " that is ",
        "it's": " it is ",
        "can't": " cannot ",
        "c'mon": " come on ",
        "don't": " do not ",
        "doesn't": " does not ",
        "won't": " will not ",
        "shouldn't": " should not ",
        "aren't": " are not ",
        "isn't": " is not ",
        "weren't": " were not ",
        "wouldn't": " would not ",
        "haven't": " have not ",
        "couldn't": " could not ",
        "hadn't": " had not ",
        "didn't": " did not ",
        "hasn't": " has not ",
        "wasn't": " was not ",
        "let's": " let us ",
        "i'll": " i will ",
        "she'll": " she will ",
        "he'll": " he will ",
        "we'll": " we will ",
        "they'll": " they will ",
        "i've": " i have ",
        "you've": " you have ",
        "we've": " we have ",
        "they've": " they have ",
        "i'd": " i would ",
        "you'd": " you would ",
        "he'd": " he would ",
        "she'd": " she would ",
        "it'd": " it would ",
        "we'd": " we would ",
        "they'd": " they would ",
        "i'm": " i am ",
        "you're": " you are ",
        "he's": " he is ",
        "she's": " she is ",
        "we're": " we are ",
        "they're": " they are ",
        "you'll": " you will ",
        "y'all": " you all ",
        "ya'll": " you all "
    }

    pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b')
    expanded_sentence = pattern.sub(lambda x: contractions_dict[x.group()], sentence)

    return expanded_sentence