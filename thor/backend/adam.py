from adam import PropagationParams
from adam import OpmParams
from adam import BatchPropagation
from adam import BatchPropagations
from adam import ConfigManager
from adam import Batch
from adam import Batches
from adam import BatchRunManager
from adam import PropagationParams
from adam import OpmParams
from adam import ConfigManager
from adam import Projects
from adam import RestRequests
from adam import AuthenticatingRestProxy
from adam.astro_utils import icrf_to_jpl_ecliptic
from adam.astro_utils import jpl_ecliptic_to_icrf
from adam.stk.io import ephemeris_file_data_to_dataframe
from .backend import Backend
from ..coordinates import transformCoordinates

import datetime
import time
import pandas as pd
import numpy as np

JPL_OBLIQUITY = np.deg2rad(84381.448 / 3600.0)

ADAM_CONFIG = {
    
}

class ADAM(Backend):
    
    def __init__(self, **kwargs):
        
        # Make sure only the correct kwargs
        # are passed to the constructor
        allowed_kwargs = ADAM_CONFIG.keys()
        for k in kwargs:
            if k not in allowed_kwargs:
                raise ValueError()
        
        # If an allowed kwarg is missing, add the 
        # default 
        for k in allowed_kwargs:
            if k not in kwargs:
                kwargs[k] = ADAM_CONFIG[k]
        
        super(ADAM, self).__init__(name="ADAM", **kwargs)

        return   


    def _propagateOrbits(self, orbits, t1):

        propagated = []

        propagated = pd.DataFrame(
        propagated,
        columns=[
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "epoch_mjd_tdb",
            "orbit_id"
        ]
        )

        for i in range(orbits.num_orbits):
                    
            orbit_id = orbits.ids[i]

            state_vec_string = "{}".format(",".join(orbits.cartesian[i].astype("str")))
            state_vec = state_vec_string.split(",")

            config_manager = ConfigManager()
            config_manager.set_default_env('dev')
            config = config_manager.get_config()

            start_vec=[]

            start_vec.append(float(state_vec[0])*149597870.700)
            start_vec.append(float(state_vec[1])*149597870.700)
            start_vec.append(float(state_vec[2])*149597870.700)
            start_vec.append(float(state_vec[3])*1731.45683681)
            start_vec.append(float(state_vec[4])*1731.45683681)
            start_vec.append(float(state_vec[5])*1731.45683681)

            start_vec_fin = jpl_ecliptic_to_icrf(start_vec[0], start_vec[1], start_vec[2], start_vec[3], start_vec[4], start_vec[5])

            propagation_params = PropagationParams({
            'start_time': str(t1.tdb.isot[0]), 
            'end_time': str(t1.tdb.isot[len(t1)-1]), 
            'project_uuid': config['workspace'], 
            })

            opm_params = OpmParams({
            'epoch': str(t1.tdb.isot[0]),
            'state_vector': start_vec_fin,
            })

            batch = Batch(propagation_params, opm_params)
            auth_rest = AuthenticatingRestProxy(RestRequests())
            batches_module = Batches(auth_rest)
            BatchRunManager(batches_module, [batch]).run()

            if (batch.get_calc_state() == 'FAILED'):
                print("Run has failed")
            else:
                while (batch.get_calc_state() != 'COMPLETED'):
                    assert (batch.get_calc_state() != 'FAILED')
            assert (batch.get_calc_state() == 'COMPLETED')

            stk_ephemeris = batch._results.get_final_ephemeris()
            stk_input = stk_ephemeris.split("\n")
            eph_output = ephemeris_file_data_to_dataframe(stk_input)

            eph_1 = eph_output[eph_output['Epoch'].isin(t1.tdb.isot)]
            eph_1['orbit_id'] = i
            eph_1.columns=['epoch_mjd_tdb', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'orbit_id']

        propagated = propagated.append(eph_1, ignore_index = True)
        propagated["x_1"] = propagated["x"] / 149597870.700
        propagated["y_1"] = propagated["y"] / 149597870.700
        propagated["z_1"] = propagated["z"] / 149597870.700
        propagated["vx_1"] = propagated["vx"] / 1731.45683681
        propagated["vy_1"] = propagated["vy"] / 1731.45683681
        propagated["vz_1"] = propagated["vz"] / 1731.45683681

        propagated = pd.DataFrame(
        propagated,
        columns=[
            "orbit_id",
            "epoch_mjd_tdb",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
        ]
        )

        return propagated



    def _generateEphemeris(self, orbits, observers):



        return

    def _orbitDetermination(self, observations):



        return