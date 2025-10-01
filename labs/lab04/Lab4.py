import math
import doctest
import nltk
nltk.download('punkt_tab')

def getVocab(vocab_fname: str) -> set: 
    """
    Args: 
        vocab_fname: filepath to vocab file. Each line has a new vocab item

    Returns: 
        A set of all the vocabulary items in the file plus three additional tokens: 
            - [UNK] : to represent words in the text not in the vocab
            - [BOS] : to represent the beginning of sentences. 
            - [EOS] : to represent the end of sentences. 
        If you run this function on the glove vocab, it should return set with 400003 items.

    >>> len(getVocab('data/glove_vocab.txt'))
    400003
    """
    try:
        with open(vocab_fname,'r') as text_file:
            my_vocab = {'[UNK]', '[BOS]', '[EOS]'}
            for line in text_file.readlines():
                for word in line.strip().split():
                    my_vocab.add(word)
        return my_vocab
        
    except FileNotFoundError:
        print("File not found!!")
        

def preprocess(textfname:str, mark_ends: bool) -> list:
    """
    Args: 
        text: some text

        mark_ends: indicates whether sentences should start with [BOS] and end with [EOS]

    Returns: 
        A list of lists where each sublist consists of tokens from each sentence. 
        Use existing nltk functions to first divide the text into sentences, and then into words. 

    >>> preprocess('data/test.txt', mark_ends=True)
    [['[BOS]', 'one', 'thing', 'was', 'certain', ',', 'that', 'the', '_white_', 'kitten', 'had', 'had', 'nothing', 'to', 'do', 'with', 'it', ':', '—it', 'was', 'the', 'black', 'kitten', '’', 's', 'fault', 'entirely', '.', '[EOS]'], ['[BOS]', 'for', 'the', 'white', 'kitten', 'had', 'been', 'having', 'its', 'face', 'washed', 'by', 'the', 'old', 'cat', 'for', 'the', 'last', 'quarter', 'of', 'an', 'hour', '(', 'and', 'bearing', 'it', 'pretty', 'well', ',', 'considering', ')', ';', 'so', 'you', 'see', 'that', 'it', '_couldn', '’', 't_', 'have', 'had', 'any', 'hand', 'in', 'the', 'mischief', '.', '[EOS]']]

    >>> preprocess('data/test.txt', mark_ends=False)
    [['one', 'thing', 'was', 'certain', ',', 'that', 'the', '_white_', 'kitten', 'had', 'had', 'nothing', 'to', 'do', 'with', 'it', ':', '—it', 'was', 'the', 'black', 'kitten', '’', 's', 'fault', 'entirely', '.'], ['for', 'the', 'white', 'kitten', 'had', 'been', 'having', 'its', 'face', 'washed', 'by', 'the', 'old', 'cat', 'for', 'the', 'last', 'quarter', 'of', 'an', 'hour', '(', 'and', 'bearing', 'it', 'pretty', 'well', ',', 'considering', ')', ';', 'so', 'you', 'see', 'that', 'it', '_couldn', '’', 't_', 'have', 'had', 'any', 'hand', 'in', 'the', 'mischief', '.']]

    """
    try:
        with open(textfname,'r') as text_file:
            all_words = []
            file_text = text_file.read()
            sentences = nltk.sent_tokenize(file_text)
            for sent in sentences:
                tokens = nltk.word_tokenize(sent)
                tokens = [item.lower() for item in tokens]           #make all lower case
                if mark_ends:
                    tokens.insert(0, '[BOS]')
                    tokens.append('[EOS]')
                all_words.append(tokens)
            return all_words
        
    except FileNotFoundError:
        print("File not found!!")


def TestBigramFreqs(freq_dict, print_non1 = False):
    """
    Helper function to use in doctest of getBigramFreqs

    """
    inverse = {}
    # inverse2 = {}
    for key,val in freq_dict.items():
        # inverse[val] = inverse.get(val, 0) + 1
        inverse[val] = inverse.get(val, []) 
        inverse[val].append(key)

    if print_non1:
        return {key: val for key, val in inverse.items() if key !=1}
    else:
        return {key: len(val) for key, val in inverse.items()}

def getBigramFreqs(preprocessed_text:list, vocab:set) -> dict:
    """
    Args: 
        preprocessed_text: text that has been divided into sentences and tokens


    Returns: 
        dictionary with all bigrams that occur in the text along with frequencies. 
        Each key should be a tuple of strings of the format (first_token, second_token). 

    >>> TestBigramFreqs(getBigramFreqs(preprocess('data/test.txt', mark_ends=True), getVocab('data/glove_vocab.txt')))
    {1: 70, 2: 3}

    >>> TestBigramFreqs(getBigramFreqs(preprocess('data/test.txt', mark_ends=True), getVocab('data/glove_vocab.txt')), print_non1=True)
    {2: [('kitten', 'had'), ('.', '[EOS]'), ('for', 'the')]}

    """
    #preprocessed_text is a 2d array
    my_bigram = {}
    for sent in preprocessed_text:
        idx = 0
        while idx < len(sent) - 1: # loop to the last but one index 
            curr_word = sent[idx]
            if curr_word not in vocab:
                curr_word = '[UNK]'
            next_word = sent[idx+1]
            if next_word not in vocab:
                next_word = '[UNK]'
            if (curr_word, next_word) in my_bigram:
                my_bigram[(curr_word, next_word)] += 1
            else:
                my_bigram[(curr_word, next_word)] = 1
            idx += 1
    return my_bigram


def get_word_freq_dict(sentences: list, vocabs: set):
    """counts the number of times a given word appears in a preprocessed corpus
    Args:
        sentences (list): a 2d array with each sentence as a sublist in the bigger list
        word (_type_): the word we are looking to count
    """
    word_freq = {"[UNK]": 0, '[EOS]': 0, '[BOS]': 0} #'[UNK]', '[BOS]', '[EOS]'}
    for sent in sentences:
        for word in sent:
            if word == '[EOS]':
                word_freq['[EOS]'] += 1
            elif word == '[BOS]':
                word_freq['[BOS]'] += 1
                
            elif word not in vocabs:
                word_freq['[UNK]'] += 1 
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    return word_freq


def corpus_size(sentences):
    count = 0
    for sent in sentences:
        count += len(sent)
    return count


def getBigramProb(bigram: tuple, smooth: str, **kwargs):
    """
    Args:
        bigram: the tuple of the bigram you want the prob of
        smooth: MLE (no smoothing), add-k where you add k to all bigram counts. Returns -1 if invalid smooth is entered. 
        **kwargs: other parameters you might want. 

        Hint: think about what parameters do you want to pass in so you minimize redundant computation. 

    Returns:
        float with prob. 
        Return -1.0 if invalid smoothing value is entered. 

        Here are the probabilities for some bigrams from data/test.txt
        
        MLE: 
            ('one', 'thing') 1.0
            ('kitten', 'had') 0.6666666666666666
            ('cat', 'had') 0.0
            ('had', 'had') 0.25
            ('on', 'the') 0.0
            ('held', 'a') 0.0
            ('zzzzzzz', 'the') 0.0

        add-1:
            ('one', 'thing') 4.999950000499995e-06
            ('kitten', 'had') 7.499887501687475e-06
            ('cat', 'had') 2.4999750002499977e-06
            ('had', 'had') 4.999912501531223e-06
            ('on', 'the') 2.499981250140624e-06
            ('held', 'a') 2.499981250140624e-06
            ('zzzzzzz', 'the') 2.4999562507656116e-06

    """
    k = 0 #no smoothing by default
    if smooth != "MLE" and smooth[:4] != "add-": #invalid smoothing value
        return -1
    
    #get the dictionary from kwargs
    bigram_dict = kwargs['bigram_dict'] #gives freq dict
    vocabs = kwargs['vocabs'] #the vocabulary of the model
    word_freqs = kwargs['dict_word_freq'] #dictionary of trained courpus words along with their frequiencies
    
    bigram_count = 0 #for cases where bigram is not present in my bigram dictionary
    if bigram in bigram_dict:
        bigram_count = bigram_dict[bigram] #update bigram count to how many times bigram occured
    
    #if the word is not in training corpus, use frequency of unknown words
    prev_word = bigram[0]
    if prev_word not in word_freqs:
        word_freq = word_freqs['[UNK]']
    else:
        word_freq = word_freqs[prev_word] #use dictionary to retrieve freq of word
        
    if smooth != "MLE":  #retrieve k if smooth is add-k
        k = float(smooth[4:]) 
        
    return (bigram_count + k) / (word_freq + len(vocabs)*k)

#code to evaluate model: we use the perplexity formula to evaluate the model; 
#N is the size of the corpus; we defined eps to address the problem of computing log(0). 
def evaluate_model(N, sentences, smoothing, vocab, bigram_freq, word_freqs):
    total_log_sum = 0
    eps = 0.0001 
    for sent in sentences:
        for i in range(1, len(sent)):                
            prob = getBigramProb((sent[i-1],sent[i]), smoothing, bigram_dict = bigram_freq, vocabs = vocab, dict_word_freq = word_freqs) 
            if prob == 0: 
                prob += eps #this is to prevent calciulating log 0
            total_log_sum += -math.log2(prob)  
    return pow(2, total_log_sum/N)     

def main():
    if __name__ == "__main__":
        doctest.testmod()
    vocab = getVocab('data/glove_vocab.txt')
    sentences = preprocess('data/test.txt', mark_ends=True)
    bigram_freq = getBigramFreqs(sentences, vocab)
    N = corpus_size(sentences)
    word_freq = get_word_freq_dict(sentences, vocab) #dictionary with words and their frequencies
    #print(word_freq)
        
    #print(evaluate_model(N, sentences, "add-0.0001", vocab, bigram_freq, word_freq))
    
    
    #clean sentences of punctuations and use actual words
    #print(getBigramProb(('one','thing'), "MLE", bigram_dict = bigram_freq, vocabs = vocab, dict_word_freq = word_freq))
    print(getBigramProb(('on','the'), "add-1", bigram_dict = bigram_freq, vocabs = vocab, dict_word_freq = word_freq))
    print(getBigramProb(('held','a'), "add-1", bigram_dict = bigram_freq, vocabs = vocab, dict_word_freq = word_freq))
    #print(getBigramProb(('zzzzzzz','the'), "MLE", bigram_dict = bigram_freq, vocabs = vocab, dict_word_freq = word_freq))    
    #print(evaluate_model(N, sentences, "MLE", vocab, bigram_freq))
    
main() 