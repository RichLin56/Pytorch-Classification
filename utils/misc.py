#!/usr/bin/python
import importlib
import inspect
import json
import logging
import os
from datetime import datetime


def load_config_json(config_file, config_group=None):
    __logger = logging.getLogger(name=__name__)
    __logger.info(
        'Loading config file from this path: {}...'.format(config_file))
    config_dict = {}
    with open(config_file, 'r') as f:
        j_doc = json.load(f)

        # iterate over groups
        for group, group_dict in j_doc.items():
            group_dict = iterate_dict(group_dict)

            config_dict[group] = group_dict

    if config_group is not None:
        __logger.info('Loaded Config_Group: {}:\n{}'.format(
            config_group, json.dumps(config_dict[config_group], indent=4)))
        return config_dict[config_group]
    else:
        __logger.info('Loaded Config_Dict:\n{}'.format(
            json.dumps(config_dict, indent=4)))
        return config_dict

def iterate_dict(dictionary):
    for key, val in dictionary.items():
        if isinstance(val, dict):
            dictionary[key] = iterate_dict(dictionary[key])
        else:
            # set attributes with value 'None' to None
            if val == 'None':
                dictionary[key] = None
            if val == 'True':
                dictionary[key] = True
            if val == 'False':
                dictionary[key] = False
    return dictionary

def find_class_in_module_by_name(module, class_name):
    class_members = [(obj, obj_name) for obj_name,
                     obj in inspect.getmembers(module) if inspect.isclass(obj)]
    class_members_name = [obj_name for obj, obj_name in class_members]
    class_members_with_class_name = [
        (obj, obj_name) for obj, obj_name in class_members if obj_name.lower() == class_name.lower()]
    assert len(class_members_with_class_name) != 0, "check config file or args!\nclass_name=\'{}\' not found in module={}\navailable class_names={}".format(
        class_name.lower(), module, class_members_name)
    assert len(class_members_with_class_name) <= 1, "multiple matches of class_name=\'{}\' found in module={}".format(
        class_name.lower(), module)
    obj = class_members_with_class_name[0][0]
    return obj

def find_module_in_package_by_name(package, module_name):
    modules = [file for file in os.listdir(package.__path__[0]) if (
        file.endswith('.py') and file != '__init__.py')]
    assert module_name.lower() + '.py' in modules, "check config file or args!\nmodule_name=\'{}\' not found in package={}\navailable modules={}".format(module_name.lower(), package, modules)
    sub_module = importlib.import_module(
        '{}.{}'.format(package.__name__, module_name.lower()))
    return sub_module

def flatten_list(flat_me, flat_into=[]):
    for i in flat_me:
        if isinstance(i, list):
            flatten_list(i, flat_into)
        else:
            flat_into.append(i)
    return flat_into

def time_now(format="%Y_%m_%d-%I_%M_%S_%p"):
    return datetime.now().strftime(format)

