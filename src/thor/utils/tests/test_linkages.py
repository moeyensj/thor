import uuid

import numpy as np
import pyarrow as pa
import quivr as qv
from adam_core.coordinates import Origin, SphericalCoordinates
from adam_core.time import Timestamp

from ...observations.observations import Observations
from ...observations.photometry import Photometry
from ..linkages import sort_by_id_and_time


class Linkages(qv.Table):
    linkage_id = qv.LargeStringColumn(default=lambda: uuid.uuid4().hex)


class LinkageMembers(qv.Table):
    linkage_id = qv.LargeStringColumn(nullable=True)
    obs_id = qv.LargeStringColumn(nullable=True)


def test_sort_by_id_and_time():
    # Create a table of linkages and linkage members and test that sorting them by linkage ID
    # and observation time works as expected
    linkages = Linkages.from_kwargs(
        linkage_id=[
            "linkage_03",
            "linkage_04",
            "linkage_01",
            "linkage_05",
            "linkage_02",
        ],
    )

    linkage_members = LinkageMembers.from_kwargs(
        linkage_id=[
            "linkage_03",
            "linkage_03",
            "linkage_03",
            "linkage_04",
            "linkage_04",
            "linkage_04",
            "linkage_01",
            "linkage_01",
            "linkage_01",
            "linkage_05",
            "linkage_05",
            "linkage_05",
            "linkage_02",
            "linkage_02",
            "linkage_02",
        ],
        obs_id=[
            "obs_03",
            "obs_02",
            "obs_04",
            "obs_05",
            "obs_03",
            "obs_04",
            "obs_01",
            "obs_03",
            "obs_02",
            "obs_04",
            "obs_05",
            "obs_03",
            "obs_02",
            "obs_03",
            "obs_01",
        ],
    )

    observations = Observations.from_kwargs(
        id=[f"obs_{i:02d}" for i in range(1, 6)],
        exposure_id=[f"exposure_{i:01d}" for i in range(1, 6)],
        coordinates=SphericalCoordinates.from_kwargs(
            rho=np.random.random(5),
            lon=np.random.random(5),
            lat=np.random.random(5),
            vrho=np.random.random(5),
            vlon=np.random.random(5),
            vlat=np.random.random(5),
            time=Timestamp.from_mjd(np.arange(59000, 59005)),
            origin=Origin.from_kwargs(code=pa.repeat("500", 5)),
            frame="eclipitic",
        ),
        photometry=Photometry.from_kwargs(
            mag=np.random.random(5),
            filter=pa.repeat("V", 5),
        ),
        state_id=["a", "b", "c", "d", "e"],
    )

    sorted_linkages, sorted_linkage_members = sort_by_id_and_time(
        linkages, linkage_members, observations, "linkage_id"
    )

    assert sorted_linkages.linkage_id.to_pylist() == [
        "linkage_01",
        "linkage_02",
        "linkage_03",
        "linkage_04",
        "linkage_05",
    ]
    assert sorted_linkage_members.linkage_id.to_pylist() == [
        "linkage_01",
        "linkage_01",
        "linkage_01",
        "linkage_02",
        "linkage_02",
        "linkage_02",
        "linkage_03",
        "linkage_03",
        "linkage_03",
        "linkage_04",
        "linkage_04",
        "linkage_04",
        "linkage_05",
        "linkage_05",
        "linkage_05",
    ]
    assert sorted_linkage_members.obs_id.to_pylist() == [
        "obs_01",
        "obs_02",
        "obs_03",
        "obs_01",
        "obs_02",
        "obs_03",
        "obs_02",
        "obs_03",
        "obs_04",
        "obs_03",
        "obs_04",
        "obs_05",
        "obs_03",
        "obs_04",
        "obs_05",
    ]
