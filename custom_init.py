from mxnet.initializer import Xavier, Initializer
from mxnet import random


class CustomInit(Initializer):
    """
    https://mxnet.incubator.apache.org/api/python/optimization.html#mxnet.initializer.register
    Create and register a custom initializer that
    Initialize the weight and bias with custom requirements

    """
    weightMethods = ["normal", "uniform", "orthogonal", "xavier"]
    biasMethods = ["costant"]

    def __init__(self, kwargs):
        self._kwargs = kwargs
        super(CustomInit, self).__init__(**kwargs)

    def _init_weight(self, name, arr):
        if name in self._kwargs.keys():
            init_params = self._kwargs[name]
            for (k, v) in init_params.items():
                if k.lower() == "normal":
                    random.normal(0, v, out=arr)
                elif k.lower() == "uniform":
                    random.uniform(-v, v, out=arr)
                elif k.lower() == "orthogonal":
                    raise NotImplementedError("Not support at the moment")
                elif k.lower() == "xavier":
                    xa = Xavier(v[0], v[1], v[2])
                    xa(name, arr)
        else:
            raise NotImplementedError("Not support")

    def _init_bias(self, name, arr):
        if name in self._kwargs.keys():
            init_params = self._kwargs[name]
            for (k, v) in init_params.items():
                if k.lower() == "costant":
                    arr[:] = v
        else:
            raise NotImplementedError("Not support")