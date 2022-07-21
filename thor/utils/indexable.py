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
    def __init__(self, index: Union[str, np.ndarray]):
        self.set_index(index)
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

        if isinstance(ind, slice) and ind.start is not None and ind.start >= len(self):
            raise IndexError(f"Index {ind.start} is out of bounds.")

        unique_ind = self.index.unique(level="class_index")
        return self.index.get_locs([unique_ind[ind], slice(None)])

    def __len__(self):
        if self.index is not None:
            return len(self.index.unique(level="class_index"))
        else:
            err = (
                "Length is not defined for this class."
            )
            raise NotImplementedError(err)

    @property
    def index(self):
        return self._index

    def set_index(self, index: Union[str, np.ndarray]):

        if isinstance(index, str):
            class_index = getattr(self, index)
            self._index_attribute = index
            array_index = np.arange(0, len(class_index), dtype=int)

        elif isinstance(index, np.ndarray):
            class_index = index
            self._index_attribute = None
            array_index = np.arange(0, len(index), dtype=int)

        else:
            err = ("index must be a str, numpy.ndarray")
            raise ValueError(err)

        self._index = pd.MultiIndex.from_arrays(
            [class_index, array_index],
            names=["class_index", "array_index"]
        )

        return

    def __getitem__(self, i: Union[int, slice, tuple, list, np.ndarray, pd.MultiIndex]):

        ind = self._handle_index(i)
        copy = deepcopy(self)

        for k, v in self.__dict__.items():
            if isinstance(v, (np.ndarray, np.ma.masked_array, list, Time, Indexable, pd.MultiIndex)):
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

    def __delitem__(self, i: Union[int, slice, tuple, list, np.ndarray, pd.MultiIndex]):

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
            elif isinstance(v, pd.MultiIndex):
                self.__dict__[k] = v.delete(ind)
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
                self.__dict__[k].mask = np.concatenate([self_v.mask, v.mask])
                self.__dict__[k].fill_value = self_v.fill_value

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

            elif isinstance(v, pd.MultiIndex):
                self.set_index(index=self._index_attribute)

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

    def sort_values(self,
            by: Union[str, List[str]],
            inplace: bool = False,
            ascending: Union[bool, List[bool]] = True,
        ):
        """
        Sort by values. Values can be contained by this class itself or any attribute
        that is also an Indexable. For example, if an attribute of this class is an Indexable
        with attribute "a", then this function will first search all attributes of this class,
        if no attribute is found then it will search all Indexable attributes of this class
        for the attribute "a".

        Parameters
        ----------
        by : {str, list}
            Sort values using this class attribute or class attributes.
        inplace : bool
            If True will sort the class inplace, if False will return a sorted
            copy of the class.
        ascending : {bool, list}
            Sort columns in ascending order or descending order. If by is a list
            then each attribute can have a separate sort order by passing a list.

        Returns
        -------
        cls : If inplace is False.
        """
        if isinstance(ascending, list) and isinstance(by, list):
            assert len(ascending) == len(by)
            ascending_ = ascending
            by_ = by

        elif isinstance(ascending, bool) and isinstance(by, list):
            ascending_ = [ascending for i in range(len(by))]
            by_ = by

        elif isinstance(ascending, bool) and isinstance(by, str):
            ascending_ = [ascending]
            by_ = [by]

        elif isinstance(ascending, list) and isinstance(by, str):
            ascending_ = ascending
            by_ = [by]

        else:
            pass

        attributes = []
        for by_i in by_:
            found = False
            try:
                attribute_i = getattr(self, by_i)
                attributes.append(attribute_i)
                found = True
            except AttributeError as e:
                for k, v in self.__dict__.items():
                    if isinstance(v, Indexable):
                        try:
                            attribute_i = getattr(v, by_i)
                            attributes.append(attribute_i)
                            found = True
                        except AttributeError as e:
                            pass

                if not found:
                    err = (f"{by_i} attribute could not be found.")
                    raise AttributeError(err)

        data = {}
        for by_i, attribute_i in zip(by_, attributes):
            if isinstance(attribute_i, np.ma.masked_array):
                data[by_i] = deepcopy(attribute_i.filled())
            elif isinstance(attribute_i, Time):
                data[by_i] = deepcopy(attribute_i.mjd)
            else:
                data[by_i] = deepcopy(attribute_i)

        df = pd.DataFrame(data)
        df_sorted = df.sort_values(
            by=by_,
            ascending=ascending_,
            inplace=False,
            ignore_index=False
        )

        sorted_indices = df_sorted.index.values
        index_attribute = deepcopy(self._index_attribute)
        # Reset index to integer values so that sorted can be cleanly
        # achieved
        index = np.arange(0, len(self.index), 1)
        self.set_index(index)

        copy = deepcopy(self[sorted_indices])
        if index_attribute is not None:
            copy.set_index(index_attribute)
            self.set_index(index_attribute)
        else:
            copy.set_index(index)
            self.set_index(index)

        if inplace:
            self.__dict__.update(copy.__dict__)
        else:
            return copy
        return

def concatenate(
        indexables: List[Indexable],
    ) -> Indexable:
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

    N = 0
    for k, v in data.items():
        if isinstance(v, list):
            if isinstance(v[0], np.ma.masked_array):
                copy.__dict__[k] = np.ma.concatenate(v)
                copy.__dict__[k].fill_value = np.NaN
                N = len(copy.__dict__[k])
            elif isinstance(v[0], np.ndarray) and k not in time_attributes:
                copy.__dict__[k] = np.concatenate(v)
                N = len(copy.__dict__[k])
            elif isinstance(v[0], Indexable):
                copy.__dict__[k] = concatenate(v)
                N = len(copy.__dict__[k])
            elif k in time_attributes:
                copy.__dict__[k] = Time(
                    np.concatenate(v),
                    scale=time_scales[k],
                    format="mjd"
                )
                N = len(copy.__dict__[k])
        elif isinstance(v, UNSLICEABLE_DATA_STRUCTURES):
            pass
        else:
            pass

    if "_index" in copy.__dict__.keys():
        if copy._index_attribute is not None:
            copy.set_index(copy._index_attribute)
        else:
            copy.set_index(np.arange(0, N, 1))

    return copy