import gudhi as gd
import numpy as np
import matplotlib.pyplot as plt
from Filtration.generate_toy_dataset import generate_circle,generate_sphere


def standard_filtration(X,max_dimension):
    """
    Get the persistence diagram of a standard filtration of a point cloud

    Parameters
    ----------
    X : np.array
        The point cloud
    max_dimension : int
        The maximum dimension of the simplices to consider
    
    """
    standard = gd.RipsComplex(points=X,max_edge_length=np.inf)
    st = standard.create_simplex_tree(max_dimension=max_dimension)
    st.compute_persistence()
    p_standard = st.persistence()
    gd.plot_persistence_diagram(p_standard)
    plt.show()


if __name__ == '__main__':
    
    ############## YOU CAN MODIFY THE FOLLOWING LINES ##############
    N_in = 100 # Number of points on the circle/sphere
    N_out = 50 # Number of points of noise
    ################################################################


    # Generate a circle
    X_circle = generate_circle(N_in,N_out)
    standard_filtration(X_circle,2)
    
    # Generate a sphere
    X_sphere = generate_sphere(N_in,N_out)
    standard_filtration(X_sphere,3)