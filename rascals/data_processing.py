import numpy as np
import pandas as pd

from .config import Config

__all__ = ["findObsInCell",
           "createQuery",
           "queryNight",
           "queryCell",
           "getObservations"]

def findObsInCell(obsIds, coords, coord_center, fieldArea=10, fieldShape="square"):
    """
    Find the observation IDs in a circular / spherical region 
    about a central point.
    
    Parameters
    ----------
    obsIds : `~np.ndarray` (N, 1)
        Array of observation IDs corresponding to the coords.
    coords : `~np.ndarray` (N, D)
        Array of coordinates of N rows for each observation
        and D dimensions. 
    coord_center : `~np.ndarray` (1, D)
        Array containing coordinates in d dimensions about which
        to search. 
    fieldArea : float, optional
        Field area in square degrees. 
        [Default = 10]
    fieldShape : str, optional
        Field's geometric shape: one of 'square' or 'circle'.
        [Default = 'square']
    
    Returns
    -------
    `~np.ndarray`
        Array of observation IDs that fall within the search radius.
    """
    if fieldShape == "square":
        half_side = np.sqrt(fieldArea) / 2
        return obsIds[np.all(np.abs(coords - coord_center) <= half_side, axis=1)]
    elif fieldShape == "circle":
        radius = np.sqrt(fieldArea / np.pi)
        distances = np.sqrt(np.sum((coords - coord_center)**2, axis=1))
        return obsIds[np.where(distances <= radius)[0]]
    else:
        raise ValueError("fieldType should be one of 'square' or 'circle'")
    return
   
def createQuery(queryType,
                observationColumns=Config.observationColumns,
                truthColumns=Config.truthColumns):
    if queryType == "observation":
        columns = list(observationColumns.values())
    elif queryType == "truth":
        columns = list(observationColumns.values()) + list(truthColumns.values())
    else:
        raise ValueError("queryType should be one of 'observation' or 'truth'")
    query = '"' + '", "'.join(columns) + '"'
    return query

def queryNight(con,
               night,
               queryType="observation",
               observationColumns=Config.observationColumns,
               truthColumns=Config.truthColumns):
    
    query = createQuery(queryType,
                        observationColumns=observationColumns,
                        truthColumns=truthColumns)
    night_df = pd.read_sql("""SELECT {} FROM ephemeris
                              WHERE night = {}""".format(query, night), con)
    columnMapping = {**observationColumns, **truthColumns}
    inverseMapping = {value : key for key, value in columnMapping.items()}
    night_df.rename(columns=inverseMapping, inplace=True)
    return night_df

def queryCell(con,
              cell,
              queryType="observation",
              chunksize=50000,
              observationColumns=Config.observationColumns,
              truthColumns=Config.truthColumns):

    query = createQuery(queryType,
                        observationColumns=observationColumns,
                        truthColumns=truthColumns)
    
    chunks = []
    for chunk in pd.read_sql("""SELECT {0} FROM ephemeris
                                 WHERE (({1} >= {2} AND {1} <= {3})
                                 AND ({4} >= {5} AND {4} <= {6})
                                 AND ({7} >= {8} AND {7} <= {9}))""".format(query,
                                                                            observationColumns["exp_mjd"],
                                                                            *cell.mjdRange,
                                                                            observationColumns["RA_deg"],
                                                                            *cell.xRange,
                                                                            observationColumns["Dec_deg"],
                                                                            *cell.yRange), 
                             con,
                             chunksize=chunksize):
        chunks.append(chunk)
    
    cell_df = pd.concat(chunks)
    columnMapping = {**observationColumns, **truthColumns}
    inverseMapping = {value : key for key, value in columnMapping.items()}
    cell_df.rename(columns=inverseMapping, inplace=True)                                                                    

    keep = rascals.findObsInCell(cell_df["obs_id"].values,
                                 cell_df[["RA_deg", "Dec_deg"]].as_matrix(),
                                 cell.center,
                                 cell.radius)
    cell_df = cell_df[cell_df["obs_id"].isin(keep)]
    return cell_df

def getObservations(raRange,
                    decRange,
                    mjdRange,
                    con,
                    queryType="truth",
                    observationColumns=Config.observationColumns,
                    truthColumns=Config.truthColumns): 
        
        query = createQuery(queryType,
                            observationColumns=observationColumns,
                            truthColumns=truthColumns)
        
        observations = pd.read_sql("""
            SELECT {0} FROM ephemeris
            WHERE (({1} >= {2} AND {1} <= {3})
            AND ({4} >= {5} AND {4} <= {6})
            AND ({7} >= {8} AND {7} <= {9}))
        """.format(query,
                   observationColumns["exp_mjd"],
                   *mjdRange,
                   observationColumns["RA_deg"],
                   *raRange,
                   observationColumns["Dec_deg"],
                   *decRange), con)

        columnMapping = {**observationColumns, **truthColumns}
        inverseMapping = {value : key for key, value in columnMapping.items()}
        observations.rename(columns=inverseMapping, inplace=True)
        print("Found {} observations.".format(len(observations)))
        return observations
