import numpy as np
import nltk
import re
import random
import math
nltk.download('punkt')

def create_pcfg(fname:str):
    with open(fname, 'r') as f:
        rules = f.readlines()

    rules = [rule for rule in rules if len(rule.strip()) > 0 and rule[0] != '#'] # remove empty lines and comments

    pcfg = {}
    for rule in rules:
        lhs, rhs =  rule.split('->')
        lhs = lhs.strip()
        curr_rules = rhs.split('|') # in case there are multiple
        for curr_rule in curr_rules:
            weight = re.findall(r'\[.*\]', curr_rule)[0]
            weight = float(weight.replace('[', '').replace(']', ''))
            curr_rhs = re.sub(r'\[.*\]', '', curr_rule).strip()
            pcfg[lhs] = pcfg.get(lhs, []) + [(curr_rhs, weight)]

    return pcfg


def pick_random(list_of_tuples):
    rules = [x[0] for x in list_of_tuples]
    probs = [x[1] for x in list_of_tuples]
    choice = random.choices(population=list_of_tuples, 
                            weights = probs,
                            k= 1)
    return list(choice)[0]

    # return list(np.random.choice(rules, 1, probs))

def generate(rule, logprob, sentence, grammar):
    sub_rules = rule.split()

    if len(sub_rules) == 1 and sub_rules[0] not in grammar: #reached terminal
        sentence.append(rule)
    else:
        for curr_rule in sub_rules:
            next_rule, next_prob = pick_random(grammar[curr_rule])

            logprob.append(math.log2(next_prob))
            generate(next_rule, logprob, sentence, grammar)

    sentence = [x.replace("'", "") for x in sentence]
    return(' '.join(sentence), np.sum(logprob))

def generate_sentences(grammar: dict, num_sents: int) -> list:
    """
    Generates num_sents, sampled probabilistically from grammar
    """

    sents = []

    for i in range(num_sents):
        sents.append(generate('ROOT', [], [], grammar))
    return sents

def parse_sentence(grammar_fname: str, sentence: str):
    with open(grammar_fname, 'r') as f:
        grammar = nltk.PCFG.fromstring(f.read())
    try:
        parser = nltk.parse.InsideChartParser(grammar)

        return list(parser.parse(sentence.split()))
    except:
        return []

def print_parses(sent, grammar_fname):
    parses = parse_sentence(grammar_fname, sent )

    for tree in parses:
        tree.pretty_print()

def is_grammatical(grammar_fname, sentence):
    return len(parse_sentence(grammar_fname, sentence)) != 0

def write_sents(grammar: dict, num: int, output_f1: str, output_f2: str):
    sentences = generate_sentences(grammar, num)
    
    with open(output_f1, 'w') as train_file:
        next = 0
        train_file.write("text\n")
        while next < len(sentences) * 0.9:
            train_file.write(sentences[next][0] + '\n')
            next += 1
    
    with open(output_f2, 'w') as validation_file:
        next = int(len(sentences) * 0.9) 
        validation_file.write("text\n") 
        while next < len(sentences):
            validation_file.write(sentences[next][0] + '\n')
            next += 1      
    
def main():
    grammar = create_pcfg('hw2_grammar_yoda.txt')
    write_sents(grammar, 1000, 'train_file.tsv', 'validation_file.tsv')   
    
main()
    





