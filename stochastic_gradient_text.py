import numpy as np
from pyprind import ProgBar
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

from text_processor import get_mini_batch_from_stream, stream_acl_imdb_df, tokenize_remove_stop

if __name__ == '__main__':
    iterations = 45
    file_path = 'movie_data.csv'
    vect = HashingVectorizer(decode_error='ignore', n_features=2 ** 21, preprocessor=None,
                             tokenizer=tokenize_remove_stop)
    clf = SGDClassifier(loss='log_loss', random_state=1)
    generator = stream_acl_imdb_df(file_path)
    prog_bar = ProgBar(iterations)
    classes = np.array([0, 1])
    for i in range(iterations):
        x_train, y_train = get_mini_batch_from_stream(generator, batch_size=1000)
        if not x_train:
            break
        x_train = vect.transform(x_train)
        clf.partial_fit(x_train, y_train, classes=classes)
        prog_bar.update()

    x_test, y_test = get_mini_batch_from_stream(generator, batch_size=5000)

    # are all values in y_test the same?
    if all([y == y_test[0] for y in y_test]):
        print(f'All values in y_test are the same {y_test[0]}')
    else:
        print('Not all values in y_test are the same')

    x_test = vect.transform(x_test)
    print('Accuracy: %.3f' % clf.score(x_test, y_test))

    import matplotlib.pyplot as plt
    # plot confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    y_pred = clf.predict(x_test)

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.legend(['Negative', 'Positive'], fontsize="x-large")
    plt.show()

    # plot precision score
    from sklearn.metrics import precision_score

    print('Precision Score: %.3f' % precision_score(y_test, y_pred))

    # plot recall score
    from sklearn.metrics import recall_score

    print('Recall Score: %.3f' % recall_score(y_test, y_pred))

    # plot balanced accuracy score
    from sklearn.metrics import balanced_accuracy_score

    print('Balanced Accuracy Score: %.3f' % balanced_accuracy_score(y_test, y_pred))

    # plot precision recall curve
    from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.legend(['Precision-Recall Curve'], fontsize="x-large")
    plt.show()

    # plot average precision score
    from sklearn.metrics import average_precision_score

    print('Average Precision Score: %.3f' % average_precision_score(y_test, y_pred))

    # plot zero one loss
    from sklearn.metrics import zero_one_loss

    print('Zero One Loss: %.3f' % zero_one_loss(y_test, y_pred))

    # plot f1 score
    from sklearn.metrics import f1_score

    print('F1 Score: %.3f' % f1_score(y_test, y_pred))

    # plot classification report
    from sklearn.metrics import classification_report

    print(classification_report(y_test, y_pred))

    # plot matthews correlation coefficient
    from sklearn.metrics import matthews_corrcoef

    print('Matthews Correlation Coefficient: %.3f' % matthews_corrcoef(y_test, y_pred))

    # plot brier score loss
    from sklearn.metrics import brier_score_loss

    print('Brier Score Loss: %.3f' % brier_score_loss(y_test, y_pred))

    # plot jaccard score
    from sklearn.metrics import jaccard_score

    print('Jaccard Score: %.3f' % jaccard_score(y_test, y_pred))
