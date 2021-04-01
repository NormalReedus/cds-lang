# standard library
import sys,os
sys.path.append('..')
from pprint import pprint
import json

# data and nlp
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])

# visualisation
import seaborn as sns
from matplotlib import rcParams
# figure size in inches
rcParams['figure.figsize'] = 20,10


# LDA tools
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from utils import lda_utils

# warnings
import logging, warnings
warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

def main():
    # Unfortunately newlines have been parsed as nothing instead of spaces
    # but the script will work just the same
    with open('data/all_series_lines.json') as file:
        content = file.read()
        line_dict = json.loads(content)


    episodes = {}

    for series_name, series in line_dict.items():
        for episode_name, episode in series.items():
            episode_string = ''

            for character_lines in episode.values():
                lines = ' '.join(character_lines)
            
                # Avoid adding just spaces
                if len(lines) != 0:
                    episode_string += ' ' + lines

            # Add the string containing all lines from the episode to our dict
            episode_key = series_name + '_' + episode_name.split()[1]
            episodes[episode_key] = episode_string

    # explicitly convert to a list for processing
    episode_lines = list(episodes.values())

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(episode_lines, min_count=10, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[episode_lines], threshold=100)  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Tokenize, remove stopwords etc
    processed_lines = lda_utils.process_words(episode_lines, nlp, bigram_mod, trigram_mod, allowed_postags=["NOUN"])

    # Convert every token to an id
    id2word = corpora.Dictionary(processed_lines)

    # Count frequencies of the tokens (ids) collocation within an episode
    corpus = [id2word.doc2bow(episode_lines) for episode_lines in processed_lines]

    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                            id2word=id2word,num_topics=12, 
                                            random_state=420,
                                            chunksize=10,
                                            passes=10,
                                            iterations=100,
                                            per_word_topics=True, 
                                            minimum_probability=0.0)

    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, 
                                        texts=processed_lines, 
                                        dictionary=id2word, 
                                        coherence='c_v')

    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    # topic overview
    pprint(lda_model.print_topics())

    # generate plot of topics over episodes
    values = list(lda_model.get_document_topics(corpus))

    split = []
    for entry in values:
        topic_prevelance = []
        for topic in entry:
            topic_prevelance.append(topic[1])
        split.append(topic_prevelance)

    df = pd.DataFrame(map(list,zip(*split)))

    topic_plot = sns.lineplot(data=df.T.rolling(20).mean())
    topic_plot.figure.savefig('topics.png')



if __name__ == '__main__':
    main()