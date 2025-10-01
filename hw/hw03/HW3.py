import nltk
from transformers import AutoTokenizer
import math

def get_ngrams(text:list, n):
    """
    Params:
        text: tokenized text in a list
        n: the n of the ngram

    Returns:
        list of all ngrams
    """
    return list(nltk.ngrams(text, n))

def get_ngramFreqs(text: list, n: int):
    """
    Params:
        text: text, split into list of tokens 
        n: the n for ngram

    Returns:
        Frequnency dictionary
    """

    ngrams = get_ngrams(text, n)
    freq_dict = nltk.probability.FreqDist(ngrams)

    return freq_dict

def preprocess(textfname: list, lower, tokenizer, **kwargs):
    """
    Params:
        textfname: path to text file. 
        tokenizer: tokenizing function 
        **kwargs: other kwargs for the tokenizer

    Returns: 
        List of tokens in the text

    """
    tokens = []
    print(f'Reading {textfname}')
    with open(textfname, 'r') as f:
        text = f.readlines()

    for i,line in enumerate(text):
        if lower:
            line = line.lower()
        tokens.extend(tokenizer(line, kwargs))
        if i%100 == 0:
            print(f'Tokenized {i+1} lines')

    return tokens

def hf_tokenize(text:str, kwargs):
    """
    Params: 
        text: string of text
        kwargs: dictionary with kwargs. Should include key 'modelname' which specifies the hf modelname
    Returns: 


    """
    modelname = kwargs['modelname']

    tokenizer = AutoTokenizer.from_pretrained(modelname)
    tokenized_output = tokenizer(text)
    words = tokenizer.convert_ids_to_tokens(tokenized_output['input_ids'])
    return words

def get_hf_vocab(modelname):
    """
    Params:
        modelname: string of hf modelname
    Returns:
        The vocabulary used by the huggingface model 

    """
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    return tokenizer.vocab

# compute bigram and trigram probabilities with addk smoothing, and evaluate LM performance.
#THINGS TO DO
# For Bigram: -get bigrams and their frequencies; write function that computes bigram probabilities.  

def get_bigram_prob(bigram: tuple, bigram_dict: dict, word_freq: dict, smoothing: str, vocab_len: int):
    """__returns the probability of a bigram__
    Args:
        bigram: the words to compute probability for
        bigram_dict: a dictionary of bigrams and their frequencies
        word_freq: a dictionary of words and their frequencies
    """
    k = float(smoothing[4:])
    bigram_count = bigram_dict[bigram]
    prev_word_count = word_freq[bigram[0]]
    return (bigram_count + k)/(prev_word_count + (vocab_len*k))


def get_trigram_prob(trigram: tuple, trigram_dict: dict, bigram_dict: dict, smoothing: str, vocab_len: int):
    """__returns the probability of a bigram__
    Args:
        trigram: the trigram whose probability is computed
        trigram_dict: a dictionary of trigrams and their frequencies
        bigram_dict: a dictionary of bigrams and their frequencies
    """
    k = float(smoothing[4:])
    trigram_count = trigram_dict[trigram]
    prev_bigram_count = bigram_dict[(trigram[0], trigram[1])]
    return (trigram_count + k)/(prev_bigram_count + (vocab_len*k))


def evaluate_model(prob_dict: dict, ngrams: tuple, corpus_size: int):
    """__Calculates the perplexity of the model__
    Args:
        prob_dict (dict):__a dictionary of either bigrams or trigrams with their probabilities__
        ngrams (tuple): __ a list of tuples of the model for which we want to calculate -log(prob)
        corpus_size (int): _number of probabilities to include in the surprisal calculations_
    """
    neg_log_sum = 0
    for ngram in ngrams:
        prob = prob_dict[ngram]
        neg_log_sum += -math.log2(prob)
    return pow(2, neg_log_sum/corpus_size)
    

def tests():
    
    text = preprocess('data/test.txt', True, hf_tokenize, modelname='distilgpt2')

    bigram_freqs = get_ngramFreqs(text, 2)

    bigrams = get_ngrams(text, 2)

    for bigram in bigrams:
        if bigram_freqs[bigram] !=1:
            print(bigram, bigram_freqs[bigram])

    print(len(get_hf_vocab('distilgpt2')))

#tests()

def main():
    text = preprocess('data/alice_in_wonderland.txt', True, hf_tokenize, modelname='distilgpt2')
    vocab_len = len(get_hf_vocab('distilgpt2'))
    smoothing = "add-0.001"
    
    print(vocab_len)
    
    #BIGRAM MODEL
    bigram_freq_dict = get_ngramFreqs(text, 2)
    bigrams = get_ngrams(text, 2)
    word_freq = get_ngramFreqs(text, 1) #words and their respective frequencies
    
    #TRIGRAM MODEL
    trigram_freq_dict = get_ngramFreqs(text, 3)
    trigrams = get_ngrams(text, 3)
    
    bigram_prob_dict = {}
    trigram_prob_dict = {}
    
    #for each bigram / trigram compute probability
    for bigram in bigrams:
        bigram_prob_dict[bigram] = get_bigram_prob(bigram, bigram_freq_dict, word_freq, smoothing, vocab_len)
        
    for trigram in trigrams:
        trigram_prob_dict[trigram] = get_trigram_prob(trigram, trigram_freq_dict, bigram_freq_dict, smoothing, vocab_len)
    
    N = 1 #what is N supposed to be: the size of the trained data, or the size of the data being evaluated on    
    bigram_eval = evaluate_model(bigram_prob_dict, bigrams, N)
    trigram_eval = evaluate_model(trigram_prob_dict, trigrams, N)
    

    
    


# if __name__ == "__main__":
#     tests()
main()