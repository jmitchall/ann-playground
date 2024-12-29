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

def detect_imbalanced_labels(y_data, imbalance_threshold=0.15):
    # if minority class is less than 15% of the total data, then the data is imbalanced
    print(f"Imbalance threshold: {imbalance_threshold * 100}%")
    print(f"Class Count: {y_data.value_counts()}")
    minority_class = y_data.value_counts().min()
    imbalance_threshold_data_count = imbalance_threshold * y_data.shape[0]
    imbalance = minority_class < imbalance_threshold_data_count
    if imbalance:
        print(f"Data is imbalanced because minority class count {minority_class} < {imbalance_threshold_data_count}  "
              f"is less than {imbalance_threshold * 100}% of the total data {y_data.shape[0]}"
              "the statistical / probabilistic arithmetic gets quite ugly, quite quickly, with unbalanced data."
              "Solving unbalanced data is basically intentionally biasing your data to get interesting results "
              "instead of accurate results. All methods are vulnerable although SVM and logistic regressions "
              "tend to be a little less vulnerable while decision trees are very vulnerable.\n"
              "I DON'T CARE) You are purely interested in accurate prediction and you think your data is "
              "representative.\nIn this case you do not have to correct at all\n "
              "I DO CARE) Interested in Prediction, You know your source is balanced but your current data is not.\n"
              "Correction needed.\n"
              "I care about rare cases and I want to make sure rare cases are predicted accurately.\n"
              "data imbalance is a problem if \n"
              "a) your model is misspecified, and \n"
              "b) you're either\n "
              "interested in good performance on a minority class or "
              "interested in the model itself. Boosting algorithms ( e.g AdaBoost, XGBoost,…), "
              "because higher weight is given to the minority class at each successive iteration. "
              "during each interation in training the weights of misclassified classes are adjusted."
              " other effective methods are: \n"
              "1) Resampling techniques: Oversampling the minority class or undersampling the majority class. "
              "2) Synthetic data generation: SMOTE (Synthetic Minority Over-sampling Technique) "
              "3) Cost-sensitive learning: Assigning higher costs to misclassifications of the minority class. "
              "4) Anomaly detection: Identifying outliers in the minority class. "
              "5) Ensemble methods: Combining multiple models to improve performance. "
              "6) Transfer learning: Using knowledge from a related task to improve performance. "
              "7) Active learning: Selecting the most informative samples for labeling. "
              "8) Semi-supervised learning: Using a combination of labeled and unlabeled data. "
              "9) Clustering: Grouping similar instances together. "
              "10) Feature selection: Identifying the most relevant features. "
              "11) Data augmentation: Increasing the size of the training set. "
              "12) Model evaluation: Using appropriate metrics to evaluate performance. "
              "13) Model interpretation: Understanding how the model makes predictions. "
              )
    else:
        print(f"Data is balanced because because minority class count {minority_class} > "
              f"{imbalance_threshold_data_count} is greater than {imbalance_threshold * 100}% of the total "
              f"data {y_data.shape[0]}.\n"
              f"IT IS BEST FOR PREDICTION: If you are purely interested in accurate prediction and you\n"
              f"think your data is representative, then you do not have to correct at all.\n"
              f"Many classical models simplify neatly under the assumption of balanced data, especially for\n"
              f"methods like ANOVA that are closely related to experimental design—a traditional / original\n"
              f"motivation for developing statistical methods\n"
              f"https://stats.stackexchange.com/questions/283170/when-is-unbalanced-data-really-a-problem-in-machine-learning")
    return imbalance

if __name__ == '__main__':
    # if aclImdb folder already exists skip extract all
    acl_imdb_df = get_acl_imdb_df()

    # remove html tags from the review column
    acl_imdb_df['review'] = acl_imdb_df['review'].apply(remove_html_tags_using_bs4)
    data_copy= acl_imdb_df.copy()
    # detect if the data is imbalanced
    print(detect_imbalanced_labels(data_copy['sentiment']))

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
    """
    Best score:  0.90382
    Best parameters:  {'clf__C': 10.0, 
    'clf__penalty': 'l2', 
    'vect__ngram_range': (1, 1), 
    'vect__stop_words': None, 
    'vect__tokenizer': <function tokenize at 0x0000029C116F9300>}
    Best estimator:  Pipeline(steps=[('vect',
                     TfidfVectorizer(lowercase=False, token_pattern=None,
                                     tokenizer=<function tokenize at 0x0000029C116F9300>)),
                    ('clf', LogisticRegression(C=10.0, solver='liblinear'))])
    """