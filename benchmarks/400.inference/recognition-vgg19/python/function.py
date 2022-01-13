
import datetime, json, os, uuid

from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50

from _mman_cffi import ffi,lib
import sys

# from . import storage
import storage
client = storage.storage.get_instance()

SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
class_idx = json.load(open(os.path.join(SCRIPT_DIR, "imagenet_class_index.json"), 'r'))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
model = None
pagesize = 4096

def handler(event):
    handler_begin = datetime.datetime.now()
    model_bucket = event.get('bucket').get('model')
    input_bucket = event.get('bucket').get('input')
    key = event.get('object').get('input')
    model_key = event.get('object').get('model')
    download_path = '/tmp/{}-{}'.format(key, uuid.uuid4())

    image_download_begin = datetime.datetime.now()
    image_path = download_path
    client.download(input_bucket, key, download_path)
    image_download_end = datetime.datetime.now()

    global model
    if not model:
        model_download_begin = datetime.datetime.now()
        model_path = os.path.join('/tmp', model_key)
        client.download(model_bucket, model_key, model_path)
        model_download_end = datetime.datetime.now()
        model_process_begin = datetime.datetime.now()
        model = torch.load(model_path)
        model.eval()
        model_process_end = datetime.datetime.now()
    else:
        model_download_begin = datetime.datetime.now()
        model_download_end = model_download_begin
        model_process_begin = datetime.datetime.now()
        model_process_end = model_process_begin
    
    def test(model):
        for key, param in model._modules.items():
            if param is not None:
                if hasattr(param, 'weight'):
                    if param.weight is not None:
                        addr = param.weight.data_ptr()
                        addr_rounded = param.weight.data_ptr() & ~(pagesize - 1)
                        size = param.weight.size().numel() * 4
                        # size_total += size
                        size_round_up = (size + pagesize - 1) & ~(pagesize - 1)
                        # print("addr", hex(addr), "addr_rounded", hex(addr_rounded), key, ".", "weight", param.weight.size(), size, "/", size_round_up, size_round_up/4096, "page(s)")
                        # madvise_page += size_round_up/4096
                        ret = lib.usm_madvise(addr_rounded, size_round_up)
                        if ret < 0:
                            print("madvise ret < 0")
                if hasattr(param, 'bias'):
                    if param.bias is not None:
                        addr = param.bias.data_ptr() & ~(pagesize - 1)
                        size = param.bias.size().numel() * 4
                        # size_total += size
                        size_round_up = (size + pagesize - 1) & ~(pagesize - 1)
                        # print(key, ".", "bias", param.bias.size(), size, "/", size_round_up, size_round_up/4096, "page(s)")
                        # madvise_page += size_round_up/4096
                        ret = lib.usm_madvise(addr, size_round_up)
                        if ret < 0:
                            print("madvise ret < 0")
                if hasattr(param, 'running_mean'):
                    if param.running_mean is not None:
                        addr = param.running_mean.data_ptr() & ~(pagesize - 1)
                        size = param.running_mean.size().numel() * 4
                        # size_total += size
                        size_round_up = (size + pagesize - 1) & ~(pagesize - 1)
                        # print(key, ".", "running_mean", param.running_mean.size(), size, "/",  size_round_up, size_round_up/4096, "page(s)")
                        # madvise_page += size_round_up/4096
                        ret = lib.usm_madvise(addr, size_round_up)
                        if ret < 0:
                            print("madvise ret < 0")
                if hasattr(param, 'running_var'):
                    if param.running_var is not None:
                        addr = param.running_var.data_ptr() & ~(pagesize - 1)
                        size = param.running_var.size().numel() * 4
                        # size_total += size
                        size_round_up = (size + pagesize - 1) & ~(pagesize - 1)
                        # print(key, ".", "running_var", param.running_var.size(), size, "/", size_round_up, size_round_up/4096, "page(s)")
                        # madvise_page += size_round_up/4096
                        ret = lib.usm_madvise(addr, size_round_up)
                        if ret < 0:
                            print("madvise ret < 0")
                if hasattr(param, 'num_batches_tracked'):
                    if param.num_batches_tracked is not None:
                        addr = param.num_batches_tracked.data_ptr() & ~(pagesize - 1)
                        size = param.num_batches_tracked.size().numel() * 4
                        # size_total += size
                        size_round_up = (size + pagesize - 1) & ~(pagesize - 1)
                        # print(key, ".", "num_batches_tracked", param.num_batches_tracked.size(), size, "/", size_round_up, size_round_up/4096, "page(s)")
                        # madvise_page += size_round_up/4096
                        ret = lib.usm_madvise(addr, size_round_up)
                        if ret < 0:
                            print("madvise ret < 0")
                test(param)
    
    madvise_begin = datetime.datetime.now()
    test(model)
    madvise_end = datetime.datetime.now()
    process_begin = datetime.datetime.now()
    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model 
    output = model(input_batch)
    _, index = torch.max(output, 1)
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    prob = torch.nn.functional.softmax(output[0], dim=0)
    _, indices = torch.sort(output, descending = True)
    ret = idx2label[index]
    process_end = datetime.datetime.now()

    download_time = (image_download_end- image_download_begin) / datetime.timedelta(microseconds=1)
    model_download_time = (model_download_end - model_download_begin) / datetime.timedelta(microseconds=1)
    model_process_time = (model_process_end - model_process_begin) / datetime.timedelta(microseconds=1)
    process_time = (process_end - process_begin) / datetime.timedelta(microseconds=1)
    madvise_time = (madvise_end - madvise_begin) / datetime.timedelta(microseconds=1)
    handler_time = (process_end - handler_begin) / datetime.timedelta(microseconds=1)
    return {
            'result': {'idx': index.item(), 'class': ret},
            'measurement': {
                'download_time': download_time + model_download_time,
                'compute_time': process_time + model_process_time,
                'model_time': model_process_time,
                'model_download_time': model_download_time,
                'madvise_time': madvise_time,
                'handler_time': handler_time
            }
        }

