import pandas as pd
import csv
import pickle
from sklearn import preprocessing


predict = pd.read_csv('predict.csv',names = ['Transaction', 'Montant', 'compteS','Solde source avant', 'Solde source après', 'compteD','Solde dest avant', 'Solde dest après'])

filename = "neural_model.sav"

tree = pickle.load(open(filename, 'rb'))

for column in predict.columns:
    if predict[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        predict[column] = le.fit_transform(predict[column])

fraude_colonne = tree.predict(predict)
fraude_colonne_df= pd.DataFrame(fraude_colonne, columns=['fraude'])
print(fraude_colonne_df[:10])
predict['fraude']=fraude_colonne_df



print(predict[:10])

predict.to_csv('result_neural.csv', sep=',', encoding='utf-8', header=False, index=False)

#writer = csv.writer(open("result.csv", 'w'))
#for row in predict.rows:
#        writer.writerow(row)
