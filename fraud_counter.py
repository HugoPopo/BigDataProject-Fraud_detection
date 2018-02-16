import csv
import json
from sys import argv

collections = {}
collections[0] = 0
collections[1] = 0

def read_csv(file):
    csv_rows = []
    json_res = ''
    fieldnames = ['Transaction', 'Montant', 'compteS','Solde source avant', 'Solde source après', 'compteD','Solde dest avant', 'Solde dest après', 'fraude']
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile, fieldnames)
        title = reader.fieldnames
        for row in reader:
            isFraud = int(row.pop('fraude'))
            collections[isFraud] += 1

read_csv(argv[1])
print("Number of frauds:", collections[1]);
print("Number of allowed transactions:", collections[0]);

