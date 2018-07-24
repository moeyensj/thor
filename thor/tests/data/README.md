To generate this test data set:

With oorb setup and MPCORB.DAT downloaded:

```
head -n 45 MPCORB.DAT > MPCORB_TEST.DAT

oorb --task=mpcorb --mpcorb=MPCORB_TEST.DAT --orb-out=MPCORB_TEST.orb

oorb --task=propagation --orb-in=MPCORB_TEST.orb --epoch-mjd-utc=59580.173 --orb-out=data/MPCORB_59580_TEST.orb

oorb --task=ephemeris --code=I11 --orb-in=MPCORB_59580_TEST.orb --timespan=30.0 --step=1.0 > MPCORB_OBS_TEST.eph
```

Then to build database:

```
from thor import buildObjectDatabase

con = buildObjectDatabase("OBJECTS_TEST.db", 
                          mpcorbFile="MPCORB_TEST.DAT",
                          orbFile="MPCORB_59580_TEST.orb",
                          ephFile="MPCORB_OBS_TEST.eph")
                          
                    
```  

To build DataFrame files:


```
from thor import readMPCORBFile
from thor import readORBFile
from thor import readEPHFile

mpcOrbitCat = readMPCORBFile("MPCORB_TEST.DAT")
mpcOrbitCat.to_csv("mpcOrbitCat_TEST.df", index=False, sep=" ")

oorbOrbitCat = readORBFile("MPCORB_59580_TEST.orb")
oorbOrbitCat.to_csv("oorbOrbitCat_TEST.df", index=False, sep=" ")

ephemeris = readEPHFile("MPCORB_OBS_TEST.eph")
ephemeris.to_csv("ephemeris_TEST.df", index=False, sep=" ")
```