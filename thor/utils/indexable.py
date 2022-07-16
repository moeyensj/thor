import numpy as np
import pandas as pd
from copy import deepcopy
from typing import (
    List,
    Optional,
    Union
)
from astropy.time import Time
from collections import OrderedDict

__all__ = [
    "Indexable",
    "concatenate"
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

def concatenate(indexables: List[Indexable]) -> Indexable:
    """
    Concatenate a list of Indexables.

    Parameters
    ----------
    indexables : list
        List of instances of Indexables.

    Returns
    -------
    indexable : Indexable
        Indexable with each sliceable attribute concatenated.
    """
    # Create a deepcopy of the first class in the list
    copy = deepcopy(indexables[0])

    # For each attribute in that class, if it is an array-like object
    # that can be concatenated add it to the dictionary as a list
    # If it is not a data structure that should be concatenated, simply
    # add a copy of that data structure to the list.
    data = {}

    # Astropy time objects concatenate slowly and very poorly so we convert them to
    # numpy arrays and track which attributes should be time objects.
    time_attributes = []
    time_scales = {}
    time_formats = {}
    for k, v in indexables[0].__dict__.items():
        if isinstance(v, (np.ndarray, np.ma.masked_array, list, Indexable)):
            data[k] = [deepcopy(v)]
        elif isinstance(v, Time):
            time_attributes.append(k)
            time_scales[k] = v.scale
            time_formats[k] = v.format

            data[k] = [v.mjd]

        elif isinstance(v, UNSLICEABLE_DATA_STRUCTURES):
            data[k] = deepcopy(v)
        else:
            data[k] = None


    # Loop through each indexable and add their attributes to lists in data
    # For unsupported data structures insure they are equal
    for indexable_i in indexables[1:]:
        for k, v in indexable_i.__dict__.items():
            if isinstance(v, (np.ndarray, np.ma.masked_array, list, Indexable)) and k not in time_attributes:
                data[k].append(v)
            elif k in time_attributes:
                assert time_scales[k] == v.scale
                data[k].append(v.mjd)
            elif isinstance(v, UNSLICEABLE_DATA_STRUCTURES):
                assert data[k] == v
            else:
                pass

    for k, v in data.items():
        if isinstance(v, list):
            if isinstance(v[0], np.ma.masked_array):
                copy.__dict__[k] = np.ma.concatenate(v)
                copy.__dict__[k].fill_value = np.NaN
            elif isinstance(v[0], np.ndarray) and k not in time_attributes:
                copy.__dict__[k] = np.concatenate(v)
            elif isinstance(v[0], Indexable):
                copy.__dict__[k] = concatenate(v)
            elif k in time_attributes:
                copy.__dict__[k] = Time(
                    np.concatenate(v),
                    scale=time_scales[k],
                    format="mjd"
                )
        elif isinstance(v, UNSLICEABLE_DATA_STRUCTURES):
            pass
        else:
            pass

    if "_index" in copy.__dict__.keys():
        index = copy.__dict__["_index"]
        if issubclass(index.dtype.type, (np.str_, np.string_)):
            copy.__dict__["_index"] = index
        elif issubclass(index.dtype.type, np.int_) and (len(pd.unique(index)) == len(index)):
            copy.__dict__["_index"] = index
        else:
            copy.__dict__["_index"] = None

    return copy