from gensim.models import KeyedVectors
from nltk.corpus import stopwords
import pandas as pd
import re

if __name__ == '__main__':
    
    ############################
    # Preprocess the dataframe
    ############################
    excel_file = 'data/SFUcorpus.xlsx'
    corpus = pd.read_excel(excel_file)
    stop_words = set(stopwords.words('english'))
    vector_model = KeyedVectors.load_word2vec_format(
        'model/GoogleNews-vectors-negative300.bin',
        binary=True,
        limit=150000,  # Reduce loading time
        unicode_errors='ignore'
    )

    # Get the ground truth value for 'toxicity_level'
    for idx, cell in corpus['toxicity_level'].iteritems():
        cell_split = [i for i in re.split(r'\\n|\'', repr(cell)) if i]
        corpus.at[idx, 'toxicity_level'] = int(cell_split[0])
        # print(idx, corpus.at[idx, 'toxicity_level'])

    # Preprocess each comment text (in column F)
    for _, text in corpus['comment_text'].iteritems():
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

        # Calculate vectors
