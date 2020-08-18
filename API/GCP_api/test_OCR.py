"""
# @Time    :  2020/8/10
# @Author  :  Jimou Chen
"""

import base64
import json
import os

from google.cloud import pubsub_v1
from google.cloud import storage
from google.cloud import translate_v2 as translate
from google.cloud import vision
from ipykernel.tests.test_message_spec import validate_message
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:/App/API/wellnet.json"

#
# def detect_text(bucket, filename):
#     print('Looking for text in image {}'.format(filename))
#
#     futures = []
#
#     text_detection_response = vision_client.text_detection({
#         'source': {'image_uri': 'gs://{}/{}'.format(bucket, filename)}
#     })
#     annotations = text_detection_response.text_annotations
#     if len(annotations) > 0:
#         text = annotations[0].description
#     else:
#         text = ''
#     print('Extracted text {} from image ({} chars).'.format(text, len(text)))
#
#     detect_language_response = translate_client.detect_language(text)
#     src_lang = detect_language_response['language']
#     print('Detected language {} for text {}.'.format(src_lang, text))
#
#     # Submit a message to the bus for each target language
#     to_langs = os.environ['TO_LANG'].split(',')
#     for target_lang in to_langs:
#         topic_name = os.environ['TRANSLATE_TOPIC']
#         if src_lang == target_lang or src_lang == 'und':
#             topic_name = os.environ['RESULT_TOPIC']
#         message = {
#             'text': text,
#             'filename': filename,
#             'lang': target_lang,
#             'src_lang': src_lang
#         }
#         message_data = json.dumps(message).encode('utf-8')
#         topic_path = publisher.topic_path(project_id, topic_name)
#         future = publisher.publish(topic_path, data=message_data)
#         futures.append(future)
#     for future in futures:
#         future.result()
#
#
# def process_image(file, context):
#     """Cloud Function triggered by Cloud Storage when a file is changed.
#     Args:
#         file (dict): Metadata of the changed file, provided by the triggering
#                                  Cloud Storage event.
#         context (google.cloud.functions.Context): Metadata of triggering event.
#     Returns:
#         None; the output is written to stdout and Stackdriver Logging
#     """
#     bucket = validate_message(file, 'bucket')
#     name = validate_message(file, 'name')
#
#     detect_text(bucket, name)
#
#     print('File {} processed.'.format(file['name']))


# def translate_text(event, context):
#     if event.get('data'):
#         message_data = base64.b64decode(event['data']).decode('utf-8')
#         message = json.loads(message_data)
#     else:
#         raise ValueError('Data sector is missing in the Pub/Sub message.')
#
#     text = validate_message(message, 'text')
#     filename = validate_message(message, 'filename')
#     target_lang = validate_message(message, 'lang')
#     src_lang = validate_message(message, 'src_lang')
#
#     print('Translating text into {}.'.format(target_lang))
#     translated_text = translate_client.translate(text,
#                                                  target_language=target_lang,
#                                                  source_language=src_lang)
#     topic_name = os.environ['RESULT_TOPIC']
#     message = {
#         'text': translated_text['translatedText'],
#         'filename': filename,
#         'lang': target_lang,
#     }
#     message_data = json.dumps(message).encode('utf-8')
#     topic_path = publisher.topic_path(project_id, topic_name)
#     future = publisher.publish(topic_path, data=message_data)
#     future.result()


vision_client = vision.ImageAnnotatorClient()
translate_client = translate.Client()
publisher = pubsub_v1.PublisherClient()
storage_client = storage.Client()

# project_id = os.environ['GCP_PROJECT']

print(storage_client)
