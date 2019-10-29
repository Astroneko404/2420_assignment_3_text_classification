from gensim.models import Word2Vec
import pandas as pd
import re

if __name__ == '__main__':
    excel_file = 'data/SFUcorpus.xlsx'

    corpus = pd.read_excel(excel_file)

    ############################
    # Preprocess the dataframe
    ############################
    # Get the ground truth value for 'toxicity_level'
    for idx, cell in corpus['toxicity_level'].iteritems():
        cell_split = [i for i in re.split(r'\\n|\'', repr(cell)) if i]
        corpus.at[idx, 'toxicity_level'] = int(cell_split[0])
        print(idx, corpus.at[idx, 'toxicity_level'])
    # print(corpus.iloc[0]['toxicity_level'])
