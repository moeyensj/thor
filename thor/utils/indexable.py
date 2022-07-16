import numpy as np
import pandas as pd
from copy import deepcopy
from typing import (
    Optional,
    Union
)
from astropy.time import Time
from collections import OrderedDict

__all__ = [
    "Indexable",
]

UNSLICEABLE_DATA_STRUCTURES = (str, int, float, dict, bool, set, OrderedDict)

class Indexable:
    """
    Class that enables indexing and slicing of itself and its members.
    If an Indexable subclass has members that are `~numpy.ndarray`s, `~numpy.ma.core.MaskedArray`s,
    lists, or `~astropy.time.core.Time`s then these members are appropriately sliced and indexed along their first axis.
    Any members that are dicts, OrderedDicts, floats, integers or strings are not indexed and left unchanged.
    """
    def __init__(self, index: Optional[np.ndarray] = None):

        if isinstance(index, np.ndarray):
            self._index = index
        else:
            self._index = np.arange(0, len(self), dtype=int)

        return

    def _handle_index(self, i: Union[int, slice, tuple, list, np.ndarray]):

        if isinstance(i, int):
            if i < 0:
                _i = i + len(self)
            else:
                _i = i
            ind = slice(_i, _i+1)

        elif isinstance(i, tuple):
            ind = list(i)

        elif isinstance(i, (slice, np.ndarray, list)):
            ind = i
        else:
            raise IndexError("Index should be either an int or a slice.")

        if isinstance(i, slice) and ind.start is not None and ind.start >= len(self):
            raise IndexError(f"Index {ind.start} is out of bounds.")

        unique_ind = pd.unique(self._index)
        ind = np.in1d(self._index, unique_ind[i])

        return ind

    def __len__(self):
        err = (
            "Length is not defined for this class."
        )
        raise NotImplementedError(err)

    @property
    def index(self):
        return self._index

    def __getitem__(self, i: Union[int, slice, tuple, list, np.ndarray]):

        ind = self._handle_index(i)
        copy = deepcopy(self)

        for k, v in self.__dict__.items():
            if isinstance(v, (np.ndarray, np.ma.masked_array, list, Time, Indexable)):
                copy.__dict__[k] = v[ind]
            elif isinstance(v, UNSLICEABLE_DATA_STRUCTURES):
                copy.__dict__[k] = v
            elif v is None:
                pass
            else:
                err = (
                    f"{type(v)} are not supported."
                )
                raise NotImplementedError(err)

        return copy

    def __delitem__(self, i: Union[int, slice, tuple, list, np.ndarray]):

        ind = self._handle_index(i)

        for k, v in self.__dict__.items():
            if isinstance(v, np.ma.masked_array):
                self.__dict__[k] = np.delete(v, np.s_[ind], axis=0)
                self.__dict__[k].mask = np.delete(v.mask, np.s_[ind], axis=0)
            elif isinstance(v, np.ndarray):
                self.__dict__[k] = np.delete(v, np.s_[ind], axis=0)
            elif isinstance(v, Time):
                self.__dict__[k] = Time(
                    np.delete(v.mjd, np.s_[ind], axis=0),
                    scale=v.scale,
                    format="mjd"
                )
            elif isinstance(v, (list, Indexable)):
                del v[ind]
            elif isinstance(v, UNSLICEABLE_DATA_STRUCTURES):
                self.__dict__[k] = v
            elif v is None:
                pass
            else:
                err = (
                    f"{type(v)} are not supported."
                )
                raise NotImplementedError(err)
        return

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def yield_chunks(self, chunk_size):
        for c in range(0, len(self), chunk_size):
            yield self[c : c + chunk_size]

    def append(self, other):

        assert type(self) == type(other)

        for k, v in other.__dict__.items():

            self_v = self.__dict__[k]

            if isinstance(v, np.ma.masked_array):
                self.__dict__[k] = np.ma.concatenate([self_v, v])

            elif isinstance(v, np.ndarray):
                self.__dict__[k] = np.concatenate([self_v, v])

            elif isinstance(v, Time):
                self.__dict__[k] = Time(
                    np.concatenate([self_v.mjd, v.mjd]),
                    scale=v.scale,
                    format="mjd"
                )

            elif isinstance(v, (list, Indexable)):
                self_v.append(v)

            elif isinstance(v, UNSLICEABLE_DATA_STRUCTURES):
                assert v == self_v

            elif v is None:
                self.__dict__[k] = None

            else:
                err = (
                    f"{type(v)} are not supported."
                )
                raise NotImplementedError(err)
        return
