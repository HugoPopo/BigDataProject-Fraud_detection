import csv
import json
from sys import argv

# dictionary containing the number of occurences of frauds and non-frauds
collections = {}
collections[0] = 0
collections[1] = 0

# Parses the file line by line and counts the number of each occurence
def read_csv(file):
    fieldnames = ['Transaction', 'Montant', 'compteS','Solde source avant', 'Solde source après', 'compteD','Solde dest avant', 'Solde dest après', 'fraude']
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile, fieldnames)
        title = reader.fieldnames
        for row in reader:
            # test if the current transaction is a fraud
            isFraud = int(row.pop('fraude'))
            collections[isFraud] += 1

read_csv(argv[1])
# display final results
print("Number of frauds:", collections[1]);
print("Number of allowed transactions:", collections[0]);

