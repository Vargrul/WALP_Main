import numpy as np
import walp_misc_functions
import walp_light_solver_lstsq

def render_single_channel(rho, Is, Ia, sm, c, N=None, L=None, dot=None):    # Only calculate is dot was not given
    if dot is None and (N is not None and L is not None):
        # Prepare a vector of the Light Vector for the dot product calculations
        L_ar = np.ones(N.shape) * L

        # Calculate the dot product for each normal in regards to the light vector
        dot = walp_misc_functions.mp_dot(N, L_ar, maximum=0)

        # Reshape the result to make it match the other parameters for vector wise calculations
        dot = np.reshape(dot, (dot.shape[0],1))
    elif dot is None:
        raise ValueError("Input N and L is needed when no dot array is given!")

    # Calculate pixel value
    p = walp_light_solver_lstsq.a(rho, sm, dot=dot)*Is + walp_light_solver_lstsq.b(rho, c)*Ia

    return p

def sim_data(rho, Is, Ia, sm, c, shape, N=None, L=None, dot=None):
    if dot is None and (N is not None and L is not None):
        # Prepare a vector of the Light Vector for the dot product calculations
        L_ar = np.ones(N.shape) * L

        # Calculate the dot product for each normal in regards to the light vector
        dot = walp_misc_functions.mp_dot(N, L_ar, maximum=0)

        # Reshape the result to make it match the other parameters for vector wise calculations
        dot = np.reshape(dot, (dot.shape[0],1))
    elif dot is None:
        raise ValueError("Input N and L is needed when no dot array is given!")


    r = render_single_channel(rho[:,0].reshape((rho.shape[0],1)), Is[0], Ia[0], sm, c, dot=dot)*255
    g = render_single_channel(rho[:,1].reshape((rho.shape[0],1)), Is[1], Ia[1], sm, c, dot=dot)*255
    b = render_single_channel(rho[:,2].reshape((rho.shape[0],1)), Is[2], Ia[2], sm, c, dot=dot)*255

    # This is to ensure the data does not overflow a uint8 - This can happen due to over exposure in the "ground truth" image
    r = np.minimum(255, np.reshape(r, shape[0:2]))
    g = np.minimum(255, np.reshape(g, shape[0:2]))
    b = np.minimum(255, np.reshape(b, shape[0:2]))

    p = np.dstack((r,g,b)).astype(np.uint8)

    # Return the simulated pixel value
    return p