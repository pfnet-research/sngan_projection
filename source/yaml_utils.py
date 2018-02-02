# !/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import sys
import time

import yaml


# Copy from tgans repo.
class Config(object):
    def __init__(self, config_dict):
        self.config = config_dict

    def __getattr__(self, key):
        if key in self.config:
            return self.config[key]
        else:
            raise AttributeError(key)

    def __getitem__(self, key):
        return self.config[key]

    def __repr__(self):
        return yaml.dump(self.config, default_flow_style=False)


def load_dataset(config):
    dataset = load_module(config.dataset['dataset_fn'],
                          config.dataset['dataset_name'])
    return dataset(**config.dataset['args'])


def load_module(fn, name):
    mod_name = os.path.splitext(os.path.basename(fn))[0]
    mod_path = os.path.dirname(fn)
    sys.path.insert(0, mod_path)
    return getattr(__import__(mod_name), name)


def load_model(model_fn, model_name, args=None):
    model = load_module(model_fn, model_name)
    if args:
        return model(**args)
    return model()


def load_updater_class(config):
    return load_module(config.updater['fn'], config.updater['name'])



