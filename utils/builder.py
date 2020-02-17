#!/usr/bin/python
import inspect

import abc
import utils.misc as misc


class Builder(object):
    def __init__(self, group_name, group_cfg, package=None, module=None):
        assert package != None or module != None
        if 'name' in group_cfg:
            self.name = group_cfg['name']
        else:
            self.name = group_name
        if 'ACTIVATE' in group_cfg:
            if group_cfg['ACTIVATE'] == False:
                return None
        if package is not None:
            module = misc.find_module_in_package_by_name(
                package=package, module_name=self.name)
        self.__build = misc.find_class_in_module_by_name(
            module=module, class_name=self.name)

    @abc.abstractmethod
    def build():
        """Return the desired object.
        """


class OptimizerBuilder(Builder):
    def __init__(self, group_name, group_cfg, package, model_params_to_update):
        super().__init__(group_name=group_name, group_cfg=group_cfg, package=package)
        self.params_from_dict = dict(item for item in group_cfg.items(
        ) if item[0] in inspect.getargspec(self._Builder__build.__init__).args)
        self.model_params_to_update = model_params_to_update
    
    def build(self):
        return self._Builder__build(self.model_params_to_update, **self.params_from_dict)


class AugmentationBuilder(Builder):
    def __init__(self, group_name, group_cfg, module):
        super().__init__(group_name=group_name, group_cfg=group_cfg, module=module)
        self.params_from_dict = dict(item for item in group_cfg.items(
        ) if item[0] in inspect.getargspec(self._Builder__build.__init__).args)
    
    def build(self):
        return self._Builder__build(**self.params_from_dict)



class SchedulerBuilder(Builder):
    def __init__(self, group_name, group_cfg, module, optimizer):
        super().__init__(group_name=group_name, group_cfg=group_cfg, module=module)
        self.params_from_dict = dict(item for item in group_cfg.items(
        ) if item[0] in inspect.getargspec(self._Builder__build.__init__).args)
        self.optimizer = optimizer

    def build(self):
        return self._Builder__build(self.optimizer, **self.params_from_dict)


