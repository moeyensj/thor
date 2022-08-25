import logging
import numpy as np
import pandas as pd
from copy import copy, deepcopy
from typing import (
    List,
    Union
)
from astropy.time import Time
from collections import OrderedDict

__all__ = [
    "Indexable",
    "concatenate"
]

logger = logging.getLogger(__name__)

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

    def query_index(self, class_ind: Union[int, slice, tuple, list, np.ndarray]):
        """
        Given a integer, slice, list, `~numpy.ndarray`, appropriately slice
        the class index and return the correct index for this class's underlying
        members.

        Parameters
        ----------
        class_ind : Union[int, slice, tuple, list, np.ndarray]
            Slice of the class index.

        Returns
        -------
        member_ind : np.ndarray
            Slice into this class's members.
        """
        if isinstance(class_ind, int):
            if class_ind < 0:
                _i = class_ind + len(self._class_index_unique)
            else:
                _i = class_ind

            # If the class index is an array of slices then
            # lets not convert class_ind to a slice yet.
            if self._class_index_is_slice:
                ind = _i
            else:
                ind = slice(_i, _i+1)

        elif isinstance(class_ind, tuple):
            ind = list(class_ind)

        elif isinstance(class_ind, (slice, np.ndarray, list)):
            ind = class_ind
        else:
            raise IndexError("Index should be either an int or a slice.")

        if isinstance(ind, slice) and ind.start is not None and ind.start >= len(self):
            raise IndexError(f"Index {ind.start} is out of bounds.")

        if isinstance(ind, int) and self._class_index_is_slice:
            member_ind = self._class_index_unique[ind]

        elif isinstance(ind, slice) and self._class_index_is_slice:

            # Check if the array of slices are consecutive and share
            # the same step size. If so, create a single slice that
            # combines all of the slices.
            slices = self._class_index_unique[ind]
            is_consecutive = True
            for i, s_i in enumerate(slices[:-1]):
                if s_i.stop != slices[i + 1].start:
                    is_consecutive = False
                    break
                if s_i.step is not None and (s_i.step != slices[i + 1].step):
                    is_consecutive = False
                    break

            if is_consecutive:
                logger.debug(
                    "Slices are consecutive and share the same step. " \
                    f"Combining slices a single slice with start {slices[0].start}, end {slices[-1].stop} and step {slices[0].step}."
                )
                member_ind = slice(slices[0].start, slices[-1].stop, slices[0].step)
            else:
                logger.debug(
                    "Slices are not consecutive. " \
                    f"Combining slices a concatenating the members index for each slice."
                )
                member_ind = np.concatenate([self._class_index_unique[s] for s in slices])

        elif self._use_class_index:
            logger.debug("Using class index to index member arrays.")
            member_ind = self._class_index_unique[ind]
        else:
            logger.debug("Using unique class index to index member arrays with np.isin.")
            member_ind = self._member_index[
                np.isin(
                    self._class_index,
                    self._class_index_unique[ind],
                )
            ]

        return member_ind, ind

    def __len__(self):
        if self.index is not None:
            return len(self._class_index_unique)
        else:
            err = (
                "Length is not defined for this class."
            )
            raise NotImplementedError(err)

    @property
    def index(self):
        return self._class_index

    def set_index(self, index: Union[str, np.ndarray]):

        # Assume by default that the class index is not going
        # to be an array of slices.
        self._class_index_is_slice = False
        self._use_class_index = False

        if isinstance(index, str):
            class_index = getattr(self, index)
            self._index_attribute = index
            member_index = np.arange(0, len(class_index), dtype=int)

        elif isinstance(index, np.ndarray):
            class_index = index
            self._index_attribute = None
            member_index = np.arange(0, len(index), dtype=int)

        else:
            err = ("index must be a str, numpy.ndarray")
            raise ValueError(err)

        # If the index is to be set on an attribute that has an object dtype (like a string)
        # then we map the unique values of the index to integers. This will make querying the
        # index significantly faster.
        if isinstance(class_index, np.ndarray) and isinstance(class_index, (object, float)):
            df = pd.DataFrame({"class_index_object" : class_index})
            df_unique = df.drop_duplicates(keep="first").copy()
            df_unique["class_index"] = np.arange(0, len(df_unique))
            class_index = df.merge(df_unique, on="class_index_object", how="left")["class_index"].values

        self._class_index = class_index
        self._class_index_unique = pd.unique(class_index)
        self._member_index = member_index

        # If the class index is monotonically increasing and the class index and member index
        # are identical then we can convert the class index into an array of slices into the
        # array members. This will make __getitem__ significantly faster.
        is_sorted = np.all((self._class_index_unique[1:] - self._class_index_unique[:-1]) == 1)
        is_same = np.all(self._class_index_unique == member_index)
        if is_sorted and not is_same:
            logger.debug(
                "Class index is sorted and not the same as the member index. " \
                "Converting class index to an array of slices."
            )
            slices = []
            slice_start = 0
            for i in self._class_index_unique:
                mask = class_index[class_index == i]
                slices.append(slice(slice_start, slice_start + len(mask)))
                slice_start += len(mask)

            self._class_index_unique = np.array(slices)
            self._class_index_is_slice = True

        if is_same:
            self._use_class_index = True

        return

    def __getitem__(self, class_ind: Union[int, slice, tuple, list, np.ndarray]):

        member_ind, class_ind_ = self.query_index(class_ind)
        copy_self = copy(self)

        for k, v in copy_self.__dict__.items():
            if k != "_class_index_unique":
                if isinstance(v, (np.ndarray, np.ma.masked_array, list, Time, Indexable)):
                    copy_self.__dict__[k] = v[member_ind]
                elif isinstance(v, UNSLICEABLE_DATA_STRUCTURES):
                    copy_self.__dict__[k] = v
                elif v is None:
                    pass
                else:
                    err = (
                        f"{type(v)} are not supported."
                    )
                    raise NotImplementedError(err)
            else:
                if self._use_class_index:
                    copy_self.__dict__[k] = v[member_ind]
                else:
                    copy_self.__dict__[k] = v[class_ind_]

        copy_self.__dict__["idx"] = 0
        return copy_self

    def __delitem__(self, class_ind: Union[int, slice, tuple, list, np.ndarray]):

        member_ind, class_ind_ = self.query_index(class_ind)

        for k, v in self.__dict__.items():
            # Everything but the class index of unique values is sliced as normal
            if k != "_class_index_unique":
                if isinstance(v, np.ma.masked_array):
                    self.__dict__[k] = np.delete(v, np.s_[member_ind], axis=0)
                    self.__dict__[k].mask = np.delete(v.mask, np.s_[member_ind], axis=0)
                elif isinstance(v, np.ndarray):
                    self.__dict__[k] = np.delete(v, np.s_[member_ind], axis=0)
                elif isinstance(v, Time):
                    self.__dict__[k] = Time(
                        np.delete(v.mjd, np.s_[member_ind], axis=0),
                        scale=v.scale,
                        format="mjd"
                    )
                elif isinstance(v, pd.Index):
                    self.__dict__[k] = v.delete(class_ind_)
                elif isinstance(v, (list, Indexable)):
                    del v[member_ind]
                elif isinstance(v, UNSLICEABLE_DATA_STRUCTURES):
                    self.__dict__[k] = v
                elif v is None:
                    pass
                else:
                    err = (
                        f"{type(v)} are not supported."
                    )
                    raise NotImplementedError(err)

            else:
                self.__dict__[k] = np.delete(v, np.s_[class_ind_], axis=0)

        return

    def __next__(self):
        try:
            self._class_index_unique[self.idx]
        except IndexError:
            raise StopIteration
        else:
            next = self[self.idx]
            self.idx += 1
            return next

    def __iter__(self):
        self.idx = 0
        return self

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