import numpy as np
import matplotlib.pyplot as plt


def generate_circle(n_in=100,n_out=50,do_plot=False):
    """
    This function generates a circle in 2D with n_in points and n_out points of noise

    Parameters
    ----------
    n_in : int, optional
        The number of points on the circle. The default is 100.
    n_out : int, optional
        The number of points of noise. The default is 50.
    do_plot : bool, optional
        If True, the function plot the circle. The default is False.
    """
    # Generate points on unit circle
    t = np.linspace(0,2*np.pi,n_in)
    X = np.array([np.cos(t),np.sin(t),np.zeros(n_in)]).T

    # Generate noise on the square
    X_circle = np.concatenate([X,-1+2*np.random.rand(n_out,3)],axis=0)

    if do_plot:
        plt.scatter(X_circle[:,0],X_circle[:,1])
        plt.axis('equal')
        plt.show()
    return X_circle


def generate_sphere(n_in,n_out,do_plot=False):
    """
    This function generates a sphere in 3D with n_in points and n_out points of noise

    Parameters
    ----------
    n_in : int
        The number of points on the sphere
    n_out : int
        The number of points of noise
    do_plot : bool, optional
        If True, the function plot the sphere. The default is False.
    """
    # Generate data points on the 3D sphere
    X = np.random.normal(0,1,(n_in,3))
    for i in range(n_in):
        X[i]/=np.linalg.norm(X,axis=1)[i]

    # Generate noise on the 3D cube
    X_sphere = np.concatenate([X,-1+2*np.random.rand(n_out,3)],axis=0)

    if do_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_sphere[:,0],X_sphere[:,1],X_sphere[:,2])
        plt.show()
    return X_sphere


if __name__ == '__main__':

    ############## YOU CAN MODIFY THE FOLLOWING LINES ##############
    N_in = 100 # Number of points on the circle/sphere
    N_out = 50 # Number of points of noise
    ################################################################

    generate_sphere(N_in,N_out,True)
    
