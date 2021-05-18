# distutils: language = c++
# cython: profile=True
# cython: linetrace=True

import numpy as np
cimport numpy as np

from libcpp.map cimport map as cpp_map
from libcpp.vector cimport vector as cpp_vector
from libcpp.utility cimport pair as cpp_pair

from cython.operator cimport dereference as deref
from cython.operator cimport preincrement, postincrement

ctypedef cpp_pair[int, int] coordinate_pair
ctypedef cpp_vector[int] hits_vec
ctypedef cpp_map[coordinate_pair, hits_vec] coordinate_map
ctypedef cpp_pair[coordinate_pair, hits_vec] coordinate_map_item


cdef class HotspotMap:
    cdef size_t hotspot_threshold
    cdef coordinate_map coord_map
    cdef cpp_vector[coordinate_pair] hot_spots

    def __cinit__(self, size_t hotspot_threshold):
        self.hotspot_threshold = hotspot_threshold
        self.coord_map = coordinate_map()
        self.hot_spots = cpp_vector[coordinate_pair]()

    def add(self, int x, int y, int idx):
        cdef coordinate_pair xy = coordinate_pair(x, y)
        cdef hits_vec hits = self.coord_map[xy]

        hits.push_back(idx)
        self.coord_map[xy] = hits
        if hits.size() == self.hotspot_threshold:
            self.hot_spots.push_back(xy)

    def map(self):
        m = {}
        it = self.coord_map.begin()
        while it != self.coord_map.end():
            m[deref(it).first] = deref(it).second
            postincrement(it)
        return m

    def hotspots(self):
        cdef list hotspots = []
        cdef hits_vec hits
        cdef cpp_map[coordinate_pair, hits_vec].iterator it
        cdef cpp_vector[int].iterator hits_vec_iterator

        for coord in self.hot_spots:
            it = self.coord_map.find(coord)
            # Create a list from the vector
            hotspot = []
            hits_vec = deref(it).second
            hits_vec_iterator = hits_vec.begin()
            while hits_vec_iterator != hits_vec.end():
                hotspot.append(deref(hits_vec_iterator))
                preincrement(hits_vec_iterator)
            hotspots.append(hotspot)
        return hotspots


cpdef hotspot_search_cpp(
     np.ndarray points,
     double eps,
     int min_samples):
    # Create two hotspot maps, offset by half the bin width to deal with edges.
    cdef HotspotMap map1 = HotspotMap(min_samples)
    cdef HotspotMap map2 = HotspotMap(min_samples)

    cdef np.ndarray points_rounded = np.floor((points / eps).T).astype('int64')
    cdef np.ndarray points_rounded_offset = np.floor(((points + eps/2) / eps).T).astype('int64')

    cdef int x, y
    for idx in range(len(points_rounded)):
        x = points_rounded[idx][0]
        y = points_rounded[idx][1]
        map1.add(x, y, idx)
        x = points_rounded_offset[idx][0]
        y = points_rounded_offset[idx][1]
        map2.add(x, y, idx)
        idx += 1

    return map1.hotspots(), map2.hotspots()


def hotspot_search_simple(points, eps, min_samples):
    # Create two histograms, offset by half the bin width to deal with edges.
    hist1 = {}
    hist2 = {}

    # Reshape the 2 X N points array into a 5 X N array.
    #  Columns are:
    #    index
    #    X rounded to nearest eps
    #    Y rounded to nearest eps
    #    X+eps/2, rounded to nearest eps
    #    Y+eps/2, rounded to nearest eps
    indices = np.arange(0, len(points))
    points_rounded = ((points / eps).T).astype('int64')
    points_rounded_offset = ((points + eps/2) / eps).T.astype('int64')

    points = np.stack(
        (indices,
         points_rounded[0],
         points_rounded[1],
         points_rounded_offset[0],
         points_rounded_offset[1]), 1)

    # Loop over the points, keeping track of how many share the same rounded x-y
    # values. If at least 5 share an x-y value, call that a "hotspot", and mark
    # it for later.
    #
    # TODO: optimize this loop - it's the vast majority of runtime (>90%)
    # according to line_profiler
    hotspots1 = set()
    hotspots2 = set()
    for (idx, x1, y1, x2, y2) in points:
        key1 = (x1, y1)
        if key1 in hist1:
            hist1[key1].append(idx)
            if len(hist1[key1]) == min_samples:
                hotspots1.add(key1)
        else:
            hist1[key1] = [idx]

        key2 = (x2, y2)
        if key2 in hist2:
            hist2[key2].append(idx)
            if len(hist2[key2]) == min_samples:
                hotspots2.add(key2)
        else:
            hist2[key2] = [idx]

    # Loop over the hotspots, and pull out the underlying indexes. Call those
    # hotspots "clusters."
    # To deal with edge effects, pull out of the other hotspot set.
    #
    # hotspots1:        hotspots2:
    #
    #  +---+---+---+
    #  |0,0|1,0|2,0|   +---+---+---+
    #  +---+---+---+   |0,0|1,0|2,0|
    #  |0,1|1,1|2,1|   +---+---+---+
    #  +---+---+---+   |0,1|1,1|2,1|
    #  |0,2|1,2|2,2|   +---+---+---+
    #  +---+---+---+   |0,2|1,2|2,2|
    #                  +---+---+---+
    #
    # If a hotspots2 value (x, y) is lit up, we can search nearby values by
    # checking hotspots1 in (x-1, y-1), (x-1, y), (x, y-1), and (x, y).
    #
    # If hotspots1 is lit up, we can check (x, y), (x+1, y), (x, y+1), and (x+1,
    # y+1).
    clusters = []
    visited = set()
    for (x, y) in hotspots1:
        cluster = []
        for coord in [(x, y), (x+1, y), (x, y+1), (x+1, y+1)]:
            if coord in hist2:
                cluster.extend(hist2[coord])
                visited.add(coord)
        if len(cluster) >= min_samples:
            clusters.append(np.array(cluster))

    for (x, y) in hotspots2:
        if (x, y) in visited:
            # Skip this point because it was covered from the previous loop.
            continue

        cluster = []
        for coord in [(x, y), (x-1, y), (x, y-1), (x-1, y-1)]:
            if coord in hist1:
                cluster.extend(hist1[coord])
        if len(cluster) >= min_samples:
            clusters.append(np.array(cluster))
    return clusters
