import os
import sys
import tarfile

import nltk
import numpy as np
import pandas as pd
import pyprind
from nltk.corpus import stopwords

nltk.download('stopwords')
stop = stopwords.words('english')

from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

porter_stop = None

df = None


def get_porter_stemmer_stop_word_list():
    global porter_stop
    global stop
    global porter
    if not porter_stop:
        porter_stop = [porter.stem(word) for word in stop]
        porter_stop.extend(stop)
        porter_stop = list(set(porter_stop))
        porter_stop.append('becau')
    return porter_stop


porter_stop = get_porter_stemmer_stop_word_list()


def extract_tar_gz(tar_gz_file):
    """
    This function extracts the tar.gz file
    :param tar_gz_file:
    :return:  None
    """
    with tarfile.open(tar_gz_file, 'r:gz') as tar:
        tar.extractall(filter=tarfile.data_filter)


def create_appended_csv_from_acl_imdb_data(csv_file):
    """
    This function reads the aclImdb data and creates a csv file with the review and sentiment

    :param csv_file: The csv file to be created
    :return:  The dataframe created from the aclImdb data
    """

    df = pd.DataFrame()
    label_mapping = {'pos': 1, 'neg': 0}
    base_path = 'aclImdb'
    progress_bar = pyprind.ProgBar(50000, stream=sys.stdout)
    for folder in ['train/pos', 'train/neg', 'test/pos', 'test/neg']:
        label = folder.split('/')[1]
        for file in os.listdir(os.path.join(base_path, folder)):
            with (open(os.path.join(base_path, folder, file), 'r', encoding='utf-8') as f):
                txt = f.read()
                df = pd.concat([df, pd.DataFrame([[txt, label_mapping[label]]])])
            progress_bar.update()
    # ensure the dataframe rows are random and shuffled No 5 rows show gave the same value
    df = df.sample(frac=1).reset_index(drop=True)

    df.to_csv(csv_file, index=False, encoding='utf-8')
    return df


def get_acl_imdb_df(data_file='movie_data.csv'):
    global df
    if not df:
        generate_acl_imdb_csv(data_file)
    if not df:
        df = pd.read_csv(data_file, encoding='utf-8')
    df = df.rename(columns={'0': 'review', '1': 'sentiment'})
    return df


def get_bag_of_words_vectors(documents_sentiment_df):
    """
    This function returns the bag of words vectors
    Bag of words is a simple and commonly used way to represent text data. They are based
    on the frequency of words in the document. The bag of words vectors are sparse vectors
    where each element represents the frequency of a word in the document.

    :param documents_sentiment_df:
    :return:  bag of words vectors
    """
    from sklearn.feature_extraction.text import CountVectorizer
    count = CountVectorizer()
    bag_of_words = count.fit_transform(documents_sentiment_df['review'].values)
    return bag_of_words , count


def get_tfidf_vectors(docs_sentiments_df):
    """
    This function returns the tfidf vectors
    Term frequency inverse document frequency (tfidf) is a way to represent text data. It is
    based on the frequency of words in the document and the frequency of words in the corpus.
    It is a way to represent the importance of words in the document. The more common a word
    is in the document and the less common it is in the corpus, the higher the tfidf value.
    The smaller the tfidf value, the less important the word is.

    :param docs_sentiments_df:
    :return:  tfidf vectors
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
    tfidf_vectors = tfidf.fit_transform(docs_sentiments_df['review'].values)
    return tfidf_vectors


def generate_acl_imdb_csv(data_file):
    global df
    if not os.path.exists('aclImdb'):
        extract_tar_gz('aclImdb_v1.tar.gz')
    if not os.path.exists(data_file):
        df = create_appended_csv_from_acl_imdb_data(data_file)
    return df


def remove_html_tags_using_bs4(text):
    """
    This function removes the html tags from the text
    :param text:
    :return:  text without html tags
    """
    # check if text has html or markup tags using regex
    import re
    if re.search('<.*?>', text):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, 'html.parser')
        return_value = soup.get_text()
    else:
        return_value = text
    # find Emoticons in text and extract them using regex and replace them with their meanings
    emoticons = re.findall(r"(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)

    # Replace all non-word characters with a space and convert text to lowercase
    cleaned_text = re.sub(r"[\W]+", ' ', return_value.lower())

    # Join emoticons into a single string, separated by spaces, and remove hyphens
    emoticons_str = ' '.join(emoticons).replace('-', '')

    # Combine the cleaned text and emoticons string
    return_value = cleaned_text + emoticons_str
    return return_value


def tokenize(text):
    """
    This function tokenizes the text
    :param text:
    :return:  tokenized text
    """
    return text.split()


def tokenize_remove_stop(text):
    """
    This function tokenizes the text and removes the stop words
    :param text:
    :return:  tokenized text
    """
    global stop
    return [word for word in tokenize(text) if word not in stop]


def tokenize_and_stem_remove_stop(text):
    """
    This function tokenizes the text and stems the words while removing the stop words
    :param text:
    :return:  tokenized and stemmed text
    """
    global porter
    global porter_stop
    if not porter_stop:
        porter_stop = get_porter_stemmer_stop_word_list()

    return [porter.stem(word) for word in tokenize(text) if porter.stem(word) not in porter_stop]


def stream_acl_imdb_df(data_file='movie_data.csv'):
    """
    This function streams the aclImdb data line by line
    :return:  generator
    """
    # if file data_file does not exist call generate_acl_imdb_csv(data_file)
    if not os.path.exists(data_file):
        generate_acl_imdb_csv(data_file)

    with open(data_file, 'r', encoding='utf-8') as f:
        next(f)  # skip the header
        for line in f:
            # last text from back starting with , and ending with \n
            last_comma = line.rfind(',')
            label_start = last_comma + 1
            text_end = last_comma - len(line)
            label = line[label_start:-1]
            text = line[:text_end]
            yield text, int(label)


def get_mini_batch_from_stream(stream_generator, batch_size=10):
    """
    This function generates a mini-batch from a stream
    :param stream_generator:  The stream generator
    :param batch_size:  The batch size
    :return:  The mini-batch
    """
    mini_batch_docs = []
    mini_batch_labels = []
    for _ in range(batch_size):
        try:
            line_doc, line_label = next(stream_generator)
            mini_batch_docs.append(remove_html_tags_using_bs4(line_doc))
            mini_batch_labels.append(line_label)
        except StopIteration:
            return None, None
    return mini_batch_docs, mini_batch_labels


if __name__ == '__main__':
    # if aclImdb folder already exists skip extract all
    file_path = 'movie_data.csv'
    generator = stream_acl_imdb_df(file_path)
    all_lines = []
    for i in range(2):
        lines = get_mini_batch_from_stream(generator, batch_size=5)
        print(lines)
        all_lines.extend(lines)
    acl_imdb_df = get_acl_imdb_df()
    acl_imdb_df['review'] = acl_imdb_df['review'].apply(remove_html_tags_using_bs4)
    acl_imdb_df['review'] = acl_imdb_df['review'].apply(tokenize_and_stem_remove_stop)
    print(acl_imdb_df.head())
    print(acl_imdb_df.shape)
