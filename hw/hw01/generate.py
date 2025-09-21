import pandas as pd
import csv

def read_file(filename):
    """
    Reads a TSV file and returns a list of rows, where each row is a list of columns.
    """
    with open(filename, "r") as f:
        lines = f.read().strip().split("\n")[1:]  # Skip the header line
    rows = [line.split('\t') for line in lines if line]
    #print(rows)
    return rows

def generate_sentences(all_verbs):
    """
    generate sentences and write them into a output file
    selected 10 verbs from the tsv file 
    selects based on word frequency; ranges to use 0-20; 21-100; 101 - 300; 301 - 500; > 500
    """
    #verbs = read_file(fname)
    i = 0
    verb_sentences = []
    while i < len(all_verbs): 
        sentence1 = "Sally " + str(all_verbs[i][0]) + " Mary because Sally did it." #expected
        sentence2 = "Mary " + str(all_verbs[i][0]) + " Sally because Sally did it." #unexpected
        sentence3 = "Sally " + str(all_verbs[i][0]) + " Mary because Mary did it." #expected
        sentence4 = "Mary " + str(all_verbs[i][0]) + " Sally because Mary did it." #unexpected
        sentence5 = "Ben " + str(all_verbs[i][0]) + " John because Ben did it." #expected
        sentence6 = "John " + str(all_verbs[i][0]) + " Ben because Ben did it." #unexpected
        sentence7 = "Ben " + str(all_verbs[i][0]) + " John because John did it." #expected
        sentence8 = "John " + str(all_verbs[i][0]) + " Ben because John did it." #unexpected
        verb_sentences.extend([sentence1, sentence2, sentence3, sentence4, sentence5, sentence6, sentence7, sentence8])
        i += 1
    return verb_sentences
    
#csv write_data function boilerplate code was generated using gemini AI. It was modified to meet the specific requirements of this particular program  
def write_data(filename, sentences, verbs):
    """
    Generates a CSV file with a specific format and a given number of rows.
    Args:
        filename (str): The name of the output CSV file.
    """
    # Define the header for the CSV file
    header = ['sentid', 'pairid', 'comparison', 'sentence', 'ROI', 'freq', 'wordfreq']
    
    # Possible values for the 'comparison' column
    comp_options = ['expected', 'unexpected']
    roi = 4 #same for every sentence
    curr_verb = 0
    try:
        # Open the file in write mode ('w') with newline='' to prevent blank rows
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            # Create a writer object to write to the CSV file
            csv_writer = csv.writer(csvfile)

            # 1. Write the header row
            csv_writer.writerow(header)

            # 2. Write the data rows
            for i in range(1, len(sentences)+1):                    
                pairid = (i + 1) // 2
                comp_idx = (i + 1) % 2
                # Write the complete row to the file
                csv_writer.writerow([i, pairid,  comp_options[comp_idx], sentences[i-1], roi, verbs[curr_verb][1], verbs[curr_verb][2]])
                if i % 8 == 0:
                    curr_verb += 1

        print(f"Successfully generated '{filename}' with {len(sentences)} rows.")

    except IOError as e:
        print(f"Error writing to file '{filename}': {e}")
    

def main():
    file_name = 'my_data.tsv'
    all_verbs = read_file(file_name)
    all_sentences = generate_sentences(all_verbs)
    write_data('data.csv', all_sentences,all_verbs)    

main()
