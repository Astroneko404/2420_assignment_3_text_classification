from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy
import pandas as pd
from random import Random
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score


def train_test_split(toxicity_shuffled_in, text_vector_in, i):
    train_X, train_Y, test_X, test_Y = [], [], [], []

    for label in range(1, 5):  # Toxicity score values only have 1, 2, 3 and 4
        for idx in toxicity[label]:
            if idx not in toxicity_shuffled_in[label][i]:
                train_X.append(text_vector_in[idx])
                train_Y.append(label)
            else:
                test_X.append(text_vector_in[idx])
                test_Y.append(label)

    return train_X, train_Y, test_X, test_Y


if __name__ == '__main__':

    ############################
    # Preprocess the dataframe
    ############################
    excel_file = 'data/SFUcorpus.xlsx'
    corpus = pd.read_excel(excel_file)
    toxicity = {1: [], 2: [], 3: [], 4: []}  # For distribution calculation

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    vectorizer_bow = CountVectorizer()
    vectorizer_tfidf = TfidfVectorizer()
    # vector_model = KeyedVectors.load_word2vec_format(
    #     'model/GoogleNews-vectors-negative300.bin',
    #     binary=True,
    #     limit=150000,  # Reduce loading time
    #     unicode_errors='ignore'
    # )

    # Remove useless columns
    corpus = corpus.drop('comment_counter', axis=1)
    corpus = corpus.drop('globe_url', axis=1)
    corpus = corpus.drop('url', axis=1)
    corpus = corpus.drop('is_constructive', axis=1)
    corpus = corpus.drop('is_constructive:confidence', axis=1)
    corpus = corpus.drop('toxicity_level:confidence', axis=1)
    corpus = corpus.drop('did_you_read_the_article', axis=1)
    corpus = corpus.drop('did_you_read_the_article:confidence', axis=1)
    corpus = corpus.drop('annotator_comments', axis=1)
    corpus = corpus.drop('expert_is_constructive', axis=1)
    corpus = corpus.drop('expert_toxicity_level', axis=1)
    corpus = corpus.drop('expert_comments', axis=1)

    # Get the ground truth value for 'toxicity_level'
    for idx, cell in corpus['toxicity_level'].iteritems():
        cell_split = [i for i in re.split(r'\\n|\'', repr(cell)) if i]
        corpus.at[idx, 'toxicity_level'] = int(cell_split[0])
        # print(idx, corpus.at[idx, 'toxicity_level'])

    # Preprocess each comment text (in column F)
    for idx, text in corpus['comment_text'].iteritems():
        ## Tokenize
        text_split = [i for i in
                      re.split(' |\n|\t|,|\.|!|\\?|;|:|-|–|—|~|%|_|\\|/|/|<|>|\^|\(|\)|\[|\]|\\|\'|\'|`|"', text)
                      if i]

        ## Lower case
        text_split = [i.lower() for i in text_split]

        ## Get rid off stop words
        ### Note: Remember to run this line in terminal first:
        ### import nltk
        ### nltk.download('stopwords')
        text_split = [i for i in text_split if len(i) > 1 if i not in stop_words]

        ## Lemmatize words (nouns only)
        text_split = [lemmatizer.lemmatize(i) for i in text_split]

        ## Put the preprocessed text back to the dataframe
        corpus.at[idx, 'comment_text'] = ' '.join(text_split)

    ###############################
    # Toxicity score distribution
    ###############################
    for idx, t_score in corpus['toxicity_level'].iteritems():
        toxicity[int(t_score)].append(idx)

    ######################
    # Train & test split
    ######################
    k = 5  # Cross validation parameter
    toxicity_shuffled = toxicity.copy()
    for label, idx_list in toxicity_shuffled.items():
        Random(0).shuffle(idx_list)
        idx_list_shuffled = numpy.array_split(numpy.array(idx_list), k)
        toxicity_shuffled[label] = [sub_list.tolist() for sub_list in idx_list_shuffled]

    ################
    # Bag of words
    ################
    text_list = corpus['comment_text'].tolist()
    text_vector_bow = vectorizer_bow.fit_transform(text_list).toarray()
    recall_micro_total = 0
    recall_macro_total = 0
    mvote_total = 0

    for i in range(k):  # Cross validation
        train_X, train_Y, test_X, test_Y = train_test_split(toxicity_shuffled, text_vector_bow, i)

        # Predict and calculate the recall
        model = LogisticRegression(
            multi_class='multinomial',
            random_state=0,
            solver='lbfgs'
        ).fit(train_X, train_Y)
        pred_Y = model.predict(test_X).tolist()
        major_Y = [1 for _ in range(len(pred_Y))]
        recall_micro_total += recall_score(test_Y, pred_Y, average='micro')
        recall_macro_total += recall_score(test_Y, pred_Y, average='macro')
        mvote_total += recall_score(test_Y, major_Y, average='micro')

    recall_micro_total /= float(k)
    recall_macro_total /= float(k)
    mvote_total /= float(k)

    print('Majority vote:\n', mvote_total)
    print()
    print('Bag of words:\n',
          'Recall (micro average):', recall_micro_total,
          '\tRecall (macro average):', recall_macro_total)  # Not suitable for skewed class
    print()

    ###################
    # TF-IDF (Sparse)
    ###################
    text_list = corpus['comment_text'].tolist()
    recall_micro_total = 0
    recall_macro_total = 0

    for i in range(k):  # Cross validation
        train_X, train_Y, test_X, test_Y = [], [], [], []

        for label in range(1, 5):
            for idx in toxicity[label]:
                if idx not in toxicity_shuffled[label][i]:
                    train_X.append(corpus.at[idx, 'comment_text'])
                    train_Y.append(label)
                else:
                    test_X.append(corpus.at[idx, 'comment_text'])
                    test_Y.append(label)

        train_X = vectorizer_tfidf.fit_transform(train_X)
        test_X = vectorizer_tfidf.transform(test_X)

        # Predict and calculate the recall
        model = LogisticRegression(
            multi_class='multinomial',
            random_state=0,
            solver='lbfgs'
        ).fit(train_X, train_Y)
        pred_Y = model.predict(test_X).tolist()
        major_Y = [1 for _ in range(len(pred_Y))]
        recall_micro_total += recall_score(test_Y, pred_Y, average='micro')
        recall_macro_total += recall_score(test_Y, pred_Y, average='macro')

    recall_micro_total /= float(k)
    recall_macro_total /= float(k)
    print('TF_IDF:\n',
          'Recall (micro average):', recall_micro_total,
          '\tRecall (macro average):', recall_macro_total)  # Not suitable for skewed class
    print()

    ############
    # Word2Vec
    ############
