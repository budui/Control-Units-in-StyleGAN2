import inspect
import warnings
from copy import deepcopy
from typing import MutableMapping

from loguru import logger


class _Registry:
    def __init__(self, name):
        self._name = name

    def get(self, key):
        raise NotImplemented

    #
    # def keys(self):
    #     raise NotImplemented
    #
    # def __len__(self):
    #     len(self.keys())

    def __contains__(self, key):
        return self.get(key) is not None

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self._name})"

    def build_with(self, cfg, default_args=None):
        """Build a module from config dict.
        Args:
            cfg (MutableMapping, or type str): Config dict. It should at least contain the key "_type".
            default_args (dict, optional): Default initialization arguments.
        Returns:
            object: The constructed object.
        """
        if isinstance(cfg, MutableMapping):
            if '_type' in cfg:
                # {"_type": "CLASS_NAME", ...arguments}
                args = deepcopy(cfg)
                obj_type = args.pop('_type')
                args = dict(args.items())
            elif len(cfg) == 1:
                # {"CLASS_NAME": {...arguments}}
                obj_type, args = list(cfg.items())[0]
                args = dict(args.items())
            else:
                raise ValueError(f"Invalid cfg. the cfg dict must contain the type info, but got {cfg}")
        elif isinstance(cfg, str):
            obj_type = cfg
            args = dict()
        else:
            raise TypeError(f'cfg must be `MutableMapping` or a str, but got {type(cfg)}')

        for invalid_key in [k for k in args.keys() if k.startswith("_")]:
            warnings.warn(f"got param start with `_`: {invalid_key}, will remove it")
            args.pop(invalid_key)

        if not (isinstance(default_args, dict) or default_args is None):
            raise TypeError('default_args must be a dict or None, '
                            f'but got {type(default_args)}')

        if isinstance(obj_type, str):
            obj_cls = self.get(obj_type)
            if obj_cls is None:
                raise KeyError(f'{obj_type} is not in the {self.name} registry')
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(f'type must be a str or valid type, but got {type(obj_type)}')

        if default_args is not None:
            for name, value in default_args.items():
                args.setdefault(name, value)
        try:
            obj = obj_cls(**args)
        except TypeError as e:
            logger.error(e)
            raise TypeError(f"invalid argument in {args} when try to build {obj_cls}\n build with {cfg}\n") from e
        return obj


class Registry(_Registry):
    """A registry to map strings to classes.
    Args:
        name (str): Registry name.
    """

    def __init__(self, name):
        super().__init__(name)
        self._module_dict = dict()

    def keys(self):
        return tuple(self._module_dict.keys())

    def __len__(self):
        len(self.keys())

    def get(self, key):
        """
        Get the registry record.
        :param key: The class name in string format.
        :return: The corresponding class.
        """
        return self._module_dict.get(key, None)

    def _register_module(self, module_class, module_name=None, force=False):
        if module_name is None:
            module_name = module_class.__name__
        if not force and module_name in self._module_dict:
            if self._module_dict[module_name] == module_class:
                warnings.warn(f'{module_name} is already registered in {self.name}, but is the same class')
                return
            if module_class.__module__ == "__main__":
                warnings.warn(f"{module_name} is already registered in {self.name}, but registered again in __main__")
                return
            raise KeyError(f'{module_name}:{self._module_dict[module_name]} is already registered in {self.name}'
                           f'so {module_class} can not be registered')
        self._module_dict[module_name] = module_class

    def register_module(self, name=None, force=False, module=None):
        """Register a module.
        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.
        Args:
            name (str | None): The module name to be registered. If not
                specified, the class name will be used.
            force (bool, optional): Whether to override an existing class with
                the same name. Default: False.
            module (type): Module class to be registered.
        """
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(
                module_class=module, module_name=name, force=force)
            return module

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)):
            raise TypeError(f'name must be a str, but got {type(name)}')

        # use it as a decorator: @x.register_module()
        def _register(cls):
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls

        return _register


class ModuleRegistry(_Registry):
    def __init__(self, module):
        super(ModuleRegistry, self).__init__(module.__name__)
        self.module = module

    def get(self, key):
        return getattr(self.module, key, None)
