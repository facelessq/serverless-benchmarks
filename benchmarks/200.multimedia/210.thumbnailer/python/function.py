import os
import sys
import uuid
from urllib.parse import unquote_plus
from PIL import Image

from storage import storage
client = storage.get_instance()

def resize_image(image_path, resized_path, w, h):
    with Image.open(image_path) as image:
        image.thumbnail((w,h))
        image.save(resized_path)

def handler(event):
  
    input_bucket = event.get('bucket').get('input')
    output_bucket = event.get('bucket').get('output')
    key = unquote_plus(event.get('object').get('key'))
    width = event.get('object').get('width')
    height = event.get('object').get('height')
    # UUID to handle multiple calls
    download_path = '/tmp/{}-{}'.format(uuid.uuid4(), key)
    upload_path = '/tmp/resized-{}'.format(key)
    client.download(input_bucket, key, download_path)
    resize_image(download_path, upload_path, width, height)
    client.upload(output_bucket, key, upload_path)
