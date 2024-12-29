from sklearn.decomposition import LatentDirichletAllocation

from text_processor import get_acl_imdb_df, get_bag_of_words_vectors

if __name__ == '__main__':
    # if aclImdb folder already exists skip extract all
    acl_imdb_df = get_acl_imdb_df()
    x , count =get_bag_of_words_vectors(acl_imdb_df)
    feature_names = count.get_feature_names_out()
    lda = LatentDirichletAllocation(n_components=10, learning_method='batch', random_state=123)
    x_topic = lda.fit_transform(x)
    print(lda.components_.shape)
    n_top_words = 5
    for topic_idx, topic in enumerate(lda.components_):
        indices_of_topic_sorted_ascending = topic.argsort()
        top_n_words_of_reversed_sorted_indices = indices_of_topic_sorted_ascending[:-n_top_words - 1:-1]
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in top_n_words_of_reversed_sorted_indices]))
