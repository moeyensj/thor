import numpy as np
from copy import deepcopy
from typing import Union
from astropy.time import Time

class Indexable:
    """
    Class that enables indexing and slicing of itself and its members.
    If an Indexable sublclass has members that are `~numpy.ndarray`s, `~numpy.ma.core.MaskedArray`s,
    these members are appropriately sliced and indexed along their first axis.
    Any members that are lists, floats, integers or strings are not indexed and left unchanged.
    """
    def _handle_index(self, i: Union[int, slice]):
        if isinstance(i, int):
            if i < 0:
                _i = i + len(self)
            else:
                _i = i
            ind = slice(_i, _i+1)

        elif isinstance(i, slice):
            ind = i
        else:
            raise IndexError("Index should be either an int or a slice.")

        if ind.start >= len(self):
            raise IndexError(f"Index {ind.start} is out of bounds.")

        return ind

    def __len__(self):
        err = (
            "Length is not defined for this class."
        )
        raise NotImplementedError(err)

    def __getitem__(self, i: Union[int, slice]):

        ind = self._handle_index(i)
        copy = deepcopy(self)

        for k, v in self.__dict__.items():
            if isinstance(v, (np.ndarray, np.ma.masked_array, Time, Indexable)):
                copy.__dict__[k] = v[ind]
            elif isinstance(v, (str, int, float, list, dict)):
                copy.__dict__[k] = v
            elif v is None:
                pass
            else:
                err = (
                    f"{type(v)} are not supported."
                )
                raise NotImplementedError(err)

        return copy

    def __delitem__(self, i: Union[int, slice]):

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
                    format=v.format
                )
            elif isinstance(v, Indexable):
                del v[ind]
            elif isinstance(v, (int, float, str, list, dict)):
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