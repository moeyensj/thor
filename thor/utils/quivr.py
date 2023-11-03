from typing import List, Literal, Optional

import numpy as np
import pyarrow as pa
import quivr as qv

__all__ = [
    "drop_duplicates",
]


def drop_duplicates(
    table: qv.AnyTable,
    subset: Optional[List[str]] = None,
    keep: Literal["first", "last"] = "first",
) -> qv.AnyTable:
    """
    Drop duplicate rows from a `~quivr.Table`. This function is similar to
    `~pandas.DataFrame.drop_duplicates` but it supports nested columns (representing
    nested tables).

    Parameters
    ----------
    table : `~quivr.Table`
        Table to drop duplicate rows from.
    subset : list of str, optional
        Subset of columns to consider when dropping duplicates. If not specified then
        all columns are used.
    keep : {'first', 'last'}, default 'first'
        If there are duplicate rows then keep the first or last row.

    Returns
    -------
    table : `~quivr.Table`
        Table with duplicate rows removed.
    """
    # Flatten the table so nested columns are dot-delimited at the top level
    flattened_table = table.flattened_table()

    # If subset is not specified then use all the columns
    if subset is None:
        subset = [c for c in flattened_table.column_names]

    # Add an index column to the flattened table
    flattened_table = flattened_table.add_column(
        0, "index", pa.array(np.arange(len(flattened_table)))
    )

    if keep == "first":
        agg_func = keep
    elif keep == "last":
        agg_func = keep
    else:
        raise ValueError("keep must be either 'first' or 'last'")
    indices = (
        flattened_table.group_by(subset, use_threads=False)
        .aggregate([("index", agg_func)])
        .column(f"index_{agg_func}")
    )

    # Take the indices from the flattened table and use them to index into the original table
    return table.take(indices)
