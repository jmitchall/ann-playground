from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from text_processor import get_acl_imdb_df, remove_html_tags_using_bs4, tokenize, tokenize_remove_stop, \
    tokenize_and_stem_remove_stop, get_porter_stemmer_stop_word_list

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None, token_pattern=None)


def get_linear_regression_model_pipeline(vectorizer):
    """
    This function generates the logistic regression model pipeline
    :param vectorizer:  vectorizer
    :return:  logistic regression model pipeline
    """
    lr_pipeline = Pipeline([('vect', vectorizer), ('clf', LogisticRegression(solver='liblinear'))])
    return lr_pipeline


porter_stop = get_porter_stemmer_stop_word_list()


def get_parameter_grid():
    """
    This function generates the parameter grid.
    :return:   parameter grid
    """
    global porter_stop
    param_grid = [{'vect__ngram_range': [(1, 1)],
                   'vect__stop_words': [None, porter_stop],
                   'vect__tokenizer': [tokenize, tokenize_remove_stop, tokenize_and_stem_remove_stop],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]
                   },
                  {'vect__ngram_range': [(1, 1)],
                   'vect__stop_words': [None, porter_stop],
                   'vect__tokenizer': [tokenize, tokenize_remove_stop, tokenize_and_stem_remove_stop],
                   'vect__use_idf': [False],
                   'vect__norm': [None],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]
                   },
                  {'vect__ngram_range': [(1, 1)],
                   'vect__stop_words': [None, porter_stop],
                   'vect__tokenizer': [tokenize, tokenize_remove_stop, tokenize_and_stem_remove_stop],
                   'vect__use_idf': [False],
                   'vect__norm': ['l1', 'l2'],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]
                   },
                  {'vect__ngram_range': [(1, 1)],
                   'vect__stop_words': [None, porter_stop],
                   'vect__tokenizer': [tokenize, tokenize_remove_stop, tokenize_and_stem_remove_stop],
                   'vect__use_idf': [True],
                   'vect__norm': ['l1', 'l2'],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]
                   }
                  ]
    return param_grid


def get_grid_search_cross_validation_results(pipeline, param_grid, x_train, y_train):
    """
    This function performs the grid search cross validation
    :param pipeline:  pipeline
    :param param_grid:  parameter grid
    :param x_train:  training data
    :param y_train:  training labels
    :return:  grid search cross validation results
    """
    from sklearn.model_selection import GridSearchCV
    gs = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='accuracy', cv=5,
                      verbose=2, n_jobs=-1)
    gs = gs.fit(x_train, y_train)
    return gs


def get_random_search_cross_validation_results(pipeline, param_grid, x_train, y_train):
    """
    This function performs the random search cross validation
    :param pipeline:  pipeline
    :param param_grid:  parameter grid
    :param x_train:  training data
    :param y_train:  training labels
    :return:  random search cross validation results
    """
    from sklearn.model_selection import RandomizedSearchCV
    rs = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, scoring='accuracy', cv=5,
                            n_iter=10, verbose=2, n_jobs=-1)
    rs = rs.fit(x_train, y_train)
    return rs


if __name__ == '__main__':
    # if aclImdb folder already exists skip extract all
    acl_imdb_df = get_acl_imdb_df()

    # remove html tags from the review column
    acl_imdb_df['review'] = acl_imdb_df['review'].apply(remove_html_tags_using_bs4)

    # Perform random search cross validation
    print("Performing Random Search Cross Validation")
    result_random_search = get_random_search_cross_validation_results(get_linear_regression_model_pipeline(tfidf),
                                                                      get_parameter_grid(),
                                                                      acl_imdb_df['review'],
                                                                      acl_imdb_df['sentiment'])

    print("Best score: ", result_random_search.best_score_)
    print("Best parameters: ", result_random_search.best_params_)
    print("Best estimator: ", result_random_search.best_estimator_)
    """
    Best score:  0.8805799999999999
    Best parameters:  {
    'vect__use_idf': False, 
    'vect__tokenizer': <function tokenize_remove_stop at 0x000002005405D080>, 
    'vect__stop_words': ['to', 'here', 'through', 'same', 'mightn', 'being', 'its', 'few', 'and', 'as', 'do', 'very', 'ani', 'there', 'is', 'ma', 'yours', 'any', 'other', 'didn', 'what', 'on', 't', 'was', 'about', 'when', "hasn't", 'this', 'yourselves', 'don', 'themselv', 'did', 'doesn', 'those', 'that', 'nor', "couldn't", 'yourself', 'an', 'then', "haven't", 'onc', 'these', 'me', 'than', 'your', 'them', 'off', 'each', 'has', 'but', 'just', "needn't", 'why', 'now', 'aren', 'i', "won't", 'isn', 'won', "shouldn't", 'she', "should'v", "you've", 'over', 'abov', 'against', 'how', 'weren', "mightn't", 'his', 'myself', 'ha', 'their', 'had', 'can', 'not', "should've", 'hers', 'couldn', 's', 'shan', 'in', 'because', 'hasn', 'been', "wasn't", 'dure', 'whi', 'veri', "weren't", 've', "hadn't", 'her', 'themselves', 'hi', "that'll", 'whom', 'only', 'up', 'herself', 'onli', 'wouldn', 'ours', 'which', 'into', 'under', 'yourselv', 'doing', 'more', 'too', "doesn't", 'the', "aren't", 'once', 'be', 'so', 'o', 'it', 'or', 'some', "you'v", 'such', 'they', 'he', 'all', 'll', 'at', 'does', 'who', "isn't", 'own', 'were', 'am', 'until', 'after', "shan't", 'most', 're', 'from', 'of', 'further', 'himself', 'haven', 'down', "mustn't", "wouldn't", "you're", 'having', 'have', 'needn', 'you', "you'll", 'where', 'a', 'below', 'd', 'him', 'wasn', 'are', 'befor', 'ain', "it's", 'our', 'becaus', 'out', 'itself', 'while', 'should', 'ourselv', 'hadn', 'my', 'shouldn', "it'", "didn't", 'thi', 'ourselves', 'during', "she's", "you'd", 'mustn', "you'r", 'for', 'both', 'wa', 'between', 'theirs', 'with', 'doe', "she'", "don't", 'again', 'y', 'if', 'by', 'm', 'will', 'no', 'above', 'we', 'before', 'becau'], 
    'vect__norm': 'l1', 
    'vect__ngram_range': (1, 1), 
    'clf__penalty': 'l1', 'clf__C': 10.0}
    Best estimator:  Pipeline(steps=[('vect',
                 TfidfVectorizer(lowercase=False, norm='l1',
                                 stop_words=['to', 'here', 'through', 'same',
                                             'mightn', 'being', 'its', 'few',
                                             'and', 'as', 'do', 'very', 'ani',
                                             'there', 'is', 'ma', 'yours',
                                             'any', 'other', 'didn', 'what',
                                             'on', 't', 'was', 'about', 'when',
                                             "hasn't", 'this', 'yourselves',
                                             'don', ...],
                                 token_pattern=None,
                                 tokenizer=<function tokenize_remove_stop at 0x000002005405D080>,
                                 use_idf=False)),
                ('clf',
                 LogisticRegression(C=10.0, penalty='l1', solver='liblinear'))])
    """
    # Perform grid search cross validation
    print("Performing Grid Search Cross Validation")
    results = get_grid_search_cross_validation_results(get_linear_regression_model_pipeline(tfidf),
                                                       get_parameter_grid(),
                                                       acl_imdb_df['review'],
                                                       acl_imdb_df['sentiment'])
    print("Best score: ", results.best_score_)
    print("Best parameters: ", results.best_params_)
    print("Best estimator: ", results.best_estimator_)
