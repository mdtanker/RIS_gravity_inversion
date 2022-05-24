import numpy as np
from grav_column_der import grav_column_der
from numba import njit
from numba_progress import ProgressBar

def jacobian(
    df_grav, 
    df, 
    spacing
):
    MATDATPAR = (np.zeros([len(df_grav),len(df)]))
    for i,j in enumerate(df_grav.Gobs):
        MATDATPAR[i,0:len(df)]=grav_column_der(df_grav.y.iloc[i], # coords of gravity observation points
                                            df_grav.x.iloc[i],
                                            df_grav.z.iloc[i],  
                                            df.northing, df.easting,     
                                            df.top, 
                                            df.bottom,
                                            spacing,     
                                            df.density/1000)
    return MATDATPAR

# with ProgressBar(total=tot_num_jobs) as progress:

# @njit(nogil=False)
def jacobian2(
    # num_iterations,
    df_grav,
    df, 
    spacing, 
    progress_proxy
):
    MATDATPAR = (np.zeros([len(df_grav),len(df)]))
    # for i in range(len(df_grav)):
    for i,j in enumerate(df_grav.Gobs):
        MATDATPAR[i,0:len(df)]=grav_column_der(df_grav.y.iloc[i], # coords of gravity observation points
                                            df_grav.x.iloc[i],
                                            df_grav.z.iloc[i],  
                                            df.northing, df.easting,     
                                            df.top, 
                                            df.bottom,
                                            spacing,     
                                            df.density/1000)
        progress_proxy.update(1)
    return MATDATPAR
    