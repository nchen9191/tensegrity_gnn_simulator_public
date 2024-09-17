from typing import Optional, Union

from torch import nn, device
from flatten_dict import flatten, unflatten
from torch.nn.modules.module import T

from utilities.misc_utils import DEFAULT_DTYPE


class BaseStateObject(nn.Module):

    def __init__(self,
                 name: str,
                 dtype=DEFAULT_DTYPE,
                 device='cpu'):
        super().__init__()
        self.name = name
        self.dtype = dtype
        self.device = device

    def set_attribute(self, attr_name, value):
        """
        - means index into list
        : means key into dict
        all means same param for all elements in list
        """
        attr_names = None
        if isinstance(attr_name, list):
            attr_names = [a for a in attr_name[1:]] if len(attr_name) > 2 else attr_name[1]
            attr_name = attr_name[0]

        index, key = None, None
        if "-" in attr_name:
            attr_name, index = attr_name.split('-')
        elif ":" in attr_name:
            attr_name, key = attr_name.split(':')

        if attr_names:
            if index:
                attr = getattr(self, attr_name)
                if index == 'all':
                    for i in range(len(attr)):
                        attr[i].set_attribute(attr_names, value)
                else:
                    attr[int(index)].set_attribute(attr_names, value)
            elif key:
                attr = getattr(self, attr_name)
                if key == 'all':
                    for k in attr.keys():
                        attr[k].set_attribute(attr_names, value)
                else:
                    attr[key].set_attribute(attr_names, value)
            else:
                attr = getattr(self, attr_name)
                attr.set_attribute(attr_names, value)
        else:
            if index:
                attr = getattr(self, attr_name)
                if index == 'all':
                    for i in range(len(attr)):
                        attr[i].set_attribute(attr_names, value)
                else:
                    attr[int(index)].set_attribute(attr_names, value)
            elif key:
                attr = getattr(self, attr_name)
                if key == 'all':
                    for k in attr.keys():
                        attr[k].set_attribute(attr_names, value)
                else:
                    attr[key].set_attribute(attr_names, value)
            else:
                setattr(self, attr_name, value)

    def set_dict_attribute(self, attr_name, value):
        attr_names = attr_name.split("-")
        param_obj, key_lst = attr_names[0], tuple(attr_names[1:])
        dict_attr = getattr(self, param_obj)

        dict_attr = flatten(dict_attr)
        dict_attr[key_lst] = value
        dict_attr = unflatten(dict_attr)

        setattr(self, param_obj, dict_attr)

    def to(self, device):
        super().to(device)
        self.device = device

        return self

    def cuda(self: T, device: Optional[Union[int, device]] = None) -> T:
        super().cuda()
        self.to('cuda')

        return self

    def cpu(self: T) -> T:
        super().cpu()
        self.to('cpu')

        return self