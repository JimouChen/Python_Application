"""
# @Time    :  2020/8/10
# @Author  :  Jimou Chen
"""
import os
from google.oauth2 import service_account
# credentials = service_account.Credentials.from_service_account_file(r"D:\App\API\wellnet.json")

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"D:\App\API\wellnet.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:/App/API/wellnet.json"

# Imports the Google Cloud client library
from google.cloud import storage

# Instantiates a client
storage_client = storage.Client()

# The name for the new bucket
bucket_name = "my-new-bucket"

# Creates the new bucket
bucket = storage_client.create_bucket(bucket_name)

print("Bucket {} created.".format(bucket.name))
