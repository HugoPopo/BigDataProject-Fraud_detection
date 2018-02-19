from pymongo import MongoClient
import csv
import json
from sys import argv
# Use: launch the script with the CSV file you want to persist in argument

# Reads the csv file line by line
def read_csv(csvFile):
    fieldnames = ['Transaction', 'Montant', 'compteS','Solde source avant', 'Solde source après', 'compteD','Solde dest avant', 'Solde dest après', 'fraude']
    with open(csvFile) as currentFile:
        reader = csv.DictReader(currentFile, fieldnames)
        title = reader.fieldnames
        for row in reader:
            print(row['fraude'])
            process_row(row)

# Saves a transaction into the remote Mongo database 
def process_row(row):
    # test if fraud or not
    isFraud = int(row.pop('fraude'))
    # cast numerical string values into floats
    row['Montant'] = float(row['Montant'])
    row['Solde source avant'] = float(row['Solde source avant'])
    row['Solde source après'] = float(row['Solde source après'])
    row['Solde dest avant'] = float(row['Solde dest avant'])
    row['Solde dest après'] = float(row['Solde dest après'])
    # save ito the database
    collections[isFraud].save(row)
    print("Saved transaction", row["compteS"], "to", row["compteS"])
    
# MongoDB settings
MONGO_HOST = "78.121.111.237" # /!\ put here the domain name or IP address of the server (in that case, be sure to enable port forwarding to this machine on port 27017)
MONGO_PORT = 27017
MONGO_DB = "frauds"
MONGO_USER = "Batman"
MONGO_PASS = "Robin<3"
con = MongoClient(MONGO_HOST, MONGO_PORT)
# dictionary containing the collections frauds & nonFrauds
collections = {}

# connection
print("Client created")
db = con[MONGO_DB]
print("DB loaded")
db.authenticate(MONGO_USER, MONGO_PASS)
print("authentication successful")
# access to the two collections
nonFrauds = con.frauds.nonFrauds
collections[0] = nonFrauds
frauds = con.frauds.frauds
collections[1] = frauds

print("Connected as "+MONGO_USER)

# begin the parsing of the file
read_csv(argv[1])
