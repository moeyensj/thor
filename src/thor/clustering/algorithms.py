from typing import Optional, Protocol, Tuple

from ..orbit import TestOrbitEphemeris
from ..range_and_transform import TransformedDetections
from .data import ClusterMembers, Clusters


class ClusteringAlgorithm(Protocol):
    """
    Protocol for clustering algorithms that operate on transformed detections
    and return clusters.

    Implementations receive the full set of transformed detections (in the
    test orbit's co-moving gnomonic frame) and optionally the test orbit
    ephemeris. They are responsible for the complete clustering process,
    whether that involves a velocity-grid sweep with 2D clustering or
    direct 3D line-finding.
    """

    def find_clusters(
        self,
        transformed_detections: TransformedDetections,
        test_orbit_ephemeris: Optional[TestOrbitEphemeris] = None,
    ) -> Tuple[Clusters, ClusterMembers]:
        """
        Find clusters in transformed detections.

        Parameters
        ----------
        transformed_detections : TransformedDetections
            Observations transformed into the test orbit's co-moving
            gnomonic frame.
        test_orbit_ephemeris : TestOrbitEphemeris, optional
            Test orbit ephemeris with covariances, used by some algorithms
            to derive clustering parameters automatically.

        Returns
        -------
        clusters : Clusters
            Table of discovered clusters.
        cluster_members : ClusterMembers
            Table mapping each cluster to its member observations.
        """
        ...
