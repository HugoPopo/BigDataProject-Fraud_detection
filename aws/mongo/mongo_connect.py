from pymongo import MongoClient
import csv
import json
from sys import argv

def read_csv(file):
    csv_rows = []
    json_res = ''
    fieldnames = ['Transaction', 'Montant', 'compteS','Solde source avant', 'Solde source après', 'compteD','Solde dest avant', 'Solde dest après', 'fraude']
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile, fieldnames)
        title = reader.fieldnames
        for row in reader:
            print(row['fraude'])
            process_row(row)

def process_row(row):
    isFraud = int(row.pop('fraude'))
    row['Montant'] = float(row['Montant'])
    row['Solde source avant'] = float(row['Solde source avant'])
    row['Solde source après'] = float(row['Solde source après'])
    row['Solde dest avant'] = float(row['Solde dest avant'])
    row['Solde dest après'] = float(row['Solde dest après'])
    collections[isFraud].save(row)
    print("Saved transaction", row["compteS"], "to", row["compteS"])
    
MONGO_HOST = "78.121.111.237"
MONGO_PORT = 27017
MONGO_DB = "frauds"
MONGO_USER = "Batman"
MONGO_PASS = "Robin<3"
con = MongoClient(MONGO_HOST, MONGO_PORT)
collections = {}

print("Client created")
db = con[MONGO_DB]
print("DB loaded")
db.authenticate(MONGO_USER, MONGO_PASS)
print("authentication successful")
nonFrauds = con.frauds.nonFrauds
collections[0] = nonFrauds
frauds = con.frauds.frauds
collections[1] = frauds

print("Connected as "+MONGO_USER)
            
read_csv(argv[1])
