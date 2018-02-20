# BigDataProject-Fraud_detection

## Introduction
The aim of the project is to automate <strong>fraud detection</strong> on banking operations thanks to [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning).
 To do this, we use [Hadoop](http://hadoop.apache.org/), [algorithms for data analysis](https://docs.microsoft.com/en-us/sql/analysis-services/data-mining/data-mining-algorithms-analysis-services-data-mining)
 , [AWS](https://aws.amazon.com/) cloud services and a NoSql database, [MongoDB](https://www.mongodb.com/).
 
## Hadoop file system
The first dataset, containing train, test and predict data, is sent to the [HDFS](https://hadoop.apache.org/docs/r1.2.1/hdfs_design.html)
component of Hadoop, on an [HortonWorks](https://hortonworks.com/) distribution on a VM. 
It is then imported back in localhost and sent to the AWS platform (cf. [this script](https://github.com/HugoPopo/BigDataProject-Fraud_detection/blob/master/hadoop/hadoop2aws.py)).

## Dataset on AWS
For the actual data processing, we use an [EC2](https://aws.amazon.com/ec2/) 
t2 micro instance with the following [Python packages](https://docs.python.org/3/installing/index.html)
installed:<ul>
<li>numpy</li>
<li>matplotlib</li>
<li>pandas</li>
<li>seaborn</li>
<li>sklearn</li>
<li>graphviz</li>
<li>mpl_toolkits</li>
</ul>

### Data analysis
For data processing, we use [decision tree](https://en.wikipedia.org/wiki/Decision_tree_learning) for classification. 
The model [trains](https://github.com/HugoPopo/BigDataProject-Fraud_detection/blob/master/aws/training.py) 
with a dedicated CSV file and is tested on another special file 
([K-nearest neighbors](https://github.com/HugoPopo/BigDataProject-Fraud_detection/blob/master/aws/k-neighbors-training.py)
and [neural networks](https://github.com/HugoPopo/BigDataProject-Fraud_detection/blob/master/aws/neural_training.py)
are also tested). 
Actual predictions [are made](https://github.com/HugoPopo/BigDataProject-Fraud_detection/blob/master/aws/treatment.py)
with a third dedicated file.
<br>The [output file](https://github.com/HugoPopo/BigDataProject-Fraud_detection/blob/master/aws/result.csv)
 is a CSV file indicating if a transaction is a fraud (1) or not (0).


## Storage in MongoDB
In order to make the results persistent, they are [stored](https://github.com/HugoPopo/BigDataProject-Fraud_detection/blob/master/aws/mongo/mongo_connect.py)
in a remote Mongo database. There are two distinct collections: <em>frauds</em> & <em>nonFrauds</em>.

## Details
The overall reasoning (files reading, preprocessing, data vizualisation, algorithms & benchmarking,...)
is exposed in this [file](https://github.com/HugoPopo/BigDataProject-Fraud_detection/blob/master/data-analysis/Big%20Data%20Project%20Etape%203.ipynb).
(You must run it with [Jupyter notebook](http://jupyter.readthedocs.io/en/latest/install.html)).

## See also
[Lespes, Robin, et al. “Ces algorithmes chasseurs de fraudeurs.” <em>Quantmetry | Data Consulting | Big Data | Data Science</em>, Quantmetry, 11 May 2017.](https://www.quantmetry.com/single-post/2017/05/11/Ces-algorithmes-chasseurs-de-fraudeurs)
