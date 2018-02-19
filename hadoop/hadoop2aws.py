import sys, paramiko, os
from boto3.s3.transfer import S3Transfer
import boto3

## From Hadoop
host_ip = "127.0.0.1"
port = 2222
user = "root"
pw = "hadooppwd"

## Files path
path = 'C:\\Users\\Sam\\Desktop'
source_train= 'train.csv'
source_test= 'test.csv'
source_predict= 'predict.csv'
full_key_name_train = os.path.join(path, source_train)
full_key_name_test = os.path.join(path, source_test)
full_key_name_predict = os.path.join(path, source_predict)

## AWS access CSV
access_key = 'AKIAI2F5E23QCQAOHEQQ'
secret_key = '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'

## From Hadoop to localhost
try:
    # SSH => get files from HDFS
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host_ip, port=port, username=user, password=pw)
    stdin, stdout, stderr = client.exec_command('hadoop fs -get /tmp/maria_dev/train.csv')
    stdin, stdout, stderr = client.exec_command('hadoop fs -get /tmp/maria_dev/predict.csv')
    stdin, stdout, stderr = client.exec_command('hadoop fs -get /tmp/maria_dev/test.csv')

    # SCP => import files in localhost
    scp = paramiko.Transport((host_ip, port))
    scp.connect(username=user, password=pw)
    sftp = paramiko.SFTPClient.from_transport(scp)
    sftp.get(source_train, full_key_name_train)
    sftp.get(source_predict, full_key_name_predict)
    sftp.get(source_test, full_key_name_test)

except:
    print("Hadoop import not working")
    
finally:
    client.close()
    scp.close()

 
## AWS send files
bucket_name = 'bigdatazbra'
try:
    client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    transfer = S3Transfer(client)
    transfer.upload_file(full_key_name_train, bucket_name, source_train)
    transfer.upload_file(full_key_name_test, bucket_name, source_test)
    transfer.upload_file(full_key_name_predict, bucket_name, source_predict)

except:
    print("S3 not working")
finally:
    print("done")

    
## delete local files
os.remove(dest_train)
os.remove(dest_test)
os.remove(dest_predict)


    
# get the files on linux instance :
#  wget https://bigdatazbra.s3.amazonaws.com/test.csv
#  wget https://bigdatazbra.s3.amazonaws.com/predict.csv
#  wget https://bigdatazbra.s3.amazonaws.com/train.csv

