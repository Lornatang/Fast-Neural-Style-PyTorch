# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
    Reference to https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
"""
from PIL import Image


def load_image(filename, size=None, scale=None):
    image = Image.open(filename)
    if size is not None:
        image = image.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        image = image.resize((int(image.size[0] / scale), int(image.size[1] / scale)), Image.ANTIALIAS)
    return image


def save_image(filename, data):
    image = data.clone().clamp(0, 255).numpy()
    image = image.transpose(1, 2, 0).astype("uint8")
    image = Image.fromarray(image)
    image.save(filename)


def gram_matrix(y):
    (n, c, h, w) = y.size()
    features = y.view(n, c, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (c * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std
