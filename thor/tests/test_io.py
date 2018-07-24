import os
import pandas as pd
import sqlite3 as sql
from pandas.util.testing import assert_frame_equal

from ..io import buildObjectDatabase

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def test_io():
    # Read test database tables 
    con_test = sql.connect(os.path.join(DATA_DIR, "OBJECTS_TEST.db"))
    mpcOrbitCat_test = pd.read_sql("""SELECT * FROM mpcOrbitCat""", con_test)
    oorbOrbitCat_test = pd.read_sql("""SELECT * FROM oorbOrbitCat""", con_test)
    ephemeris_test = pd.read_sql("""SELECT * FROM ephemeris""", con_test)

    # Run current version of code
    con_current = buildObjectDatabase(os.path.join(DATA_DIR, "OBJECTS_TEST_current.db"), 
                                      mpcorbFile=os.path.join(DATA_DIR, "MPCORB_TEST.DAT"),
                                      orbFile=os.path.join(DATA_DIR, "MPCORB_59580_TEST.orb"),
                                      ephFile=os.path.join(DATA_DIR, "MPCORB_OBS_TEST.eph"))
    mpcOrbitCat_current = pd.read_sql("""SELECT * FROM mpcOrbitCat""", con_current)
    oorbOrbitCat_current = pd.read_sql("""SELECT * FROM oorbOrbitCat""", con_current)
    ephemeris_current = pd.read_sql("""SELECT * FROM ephemeris""", con_current)
    
    # Check that the dataframes are the same
    assert_frame_equal(mpcOrbitCat_test, mpcOrbitCat_current)
    assert_frame_equal(oorbOrbitCat_test, oorbOrbitCat_current)
    assert_frame_equal(ephemeris_test, ephemeris_current)
    
    # Delete current
    con_current.close()
    con_test.close()
    os.remove(os.path.join(DATA_DIR, "OBJECTS_TEST_current.db"))