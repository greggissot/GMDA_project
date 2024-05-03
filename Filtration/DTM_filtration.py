import gudhi as gd
import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from Filtration.generate_toy_dataset import generate_circle,generate_sphere




def radius(t,fx,p):
    """
    Compute the radius of the ball 
    Parameters
    ----------
    t : float
        The index
    fx : float
        The values of the function at point x
    p : float
        The p-norm used to compute the distance
    Returns
    -------
    float
        The distance between the measure and the function
    """
    # Compute the distance
    if t<fx:
        return -np.inf
    elif p==np.inf:
        return t
    else:
        return (t**p - fx**p)**(1/p)
    

def compute_DTM_vector(X,k):
    """
    Compute the DTM matrix
    Parameters
    ----------
    X : ndarray
        The data
    k : int
        The number of nearest neighbors must be inferior to X.shape[0]
    Returns
    -------
    ndarray
        The DTM vector
    """

    kd_tree = KDTree(X,leaf_size=30,metric='euclidean')
    dist,ind = kd_tree.query(X,k,return_distance=True)
    DTM = np.sqrt(np.sum(dist**2,axis=1)/k)
    return DTM


def get_filtration_value_of_edge(p,fx,fy,distance_xy,max_iter=100):
    """
    Compute the filtration value of an edge
    Parameters
    ----------
    p : float
        The p-norm used to compute the distance
    fx : float
        The values of the function at point x
    fy : float
        The values of the function at point y
    distance_xy : float
        The distance between the two points
    Returns
    -------
    float
        The filtration value of the edge
    """
    # Compute the filtration value of the edge
    if p == np.inf:
        return max([fx,fy,distance_xy/2])
    elif p==1:
        return 0.5*(fx+fy+distance_xy)
    elif p==2:
        return np.sqrt((fx+fy)**2+distance_xy**2)*np.sqrt((fx-fy)**2+distance_xy**2)/(2*distance_xy+1e-10)
    else:
        if distance_xy <= abs(fx**p-fy**p)**(1/p):
            return max([fx,fy])
        else:
            tmin = max([fx,fy])
            tmax = (distance_xy**p + tmin**p)**(1/p)
            for _ in range(max_iter):
                t = (tmin+tmax)/2
                if (t**p - fx**p)**(1/p) + (t**p - fy**p)**(1/p) > distance_xy:
                    tmax = t
                else:
                    tmin = t

            return t
        

def DTM_filtration(X,k,p,dimension_max):
    """
    Compute the DTM filtration
    Parameters
    ----------
    X : ndarray
        The data
    k : int
        The number of nearest neighbors must be inferior to X.shape[0]
    p : float
        The p-norm used to compute the distance
    dimension_max : int
        The maximum dimension of the simplicial complex
    Returns
    -------
    tuple
        The simplicial complex and the filtration
    """
    # Compute the DTM vector
    DTM = compute_DTM_vector(X,k)
    
    # Compute the filtration
    complex = gd.RipsComplex(points=X,max_edge_length=np.inf)
    st = complex.create_simplex_tree()
    for i in range(X.shape[0]):
        st.assign_filtration([i],DTM[i])
    for i in range(X.shape[0]):
        for j in range(i+1,X.shape[0]):
            st.assign_filtration([i,j],get_filtration_value_of_edge(p,DTM[i],DTM[j],np.linalg.norm(X[i,:]-X[j,:],ord=p)))

    st.expansion(dimension_max)

    return st


if __name__ == '__main__':

    ############## YOU CAN MODIFY THE FOLLOWING LINES ##############
    k_neighbors = 50 # Number of nearest neighbors
    p_norm = 1 # p-norm
    N_in = 100 # Number of points on the circle/sphere
    N_out = 50 # Number of points of noise
    ################################################################


    # Generate a circle
    X_circle = generate_circle(N_in,N_out)
    
    # Compute the DTM filtration
    st = DTM_filtration(X_circle,k_neighbors,p_norm,2)
    
    # Plot the persistence diagram
    gd.plot_persistence_diagram(st.persistence())
    plt.show()
    
    # Generate a sphere
    X_sphere = generate_sphere(N_in,N_out)
    
    # Compute the DTM filtration
    st = DTM_filtration(X_sphere,k_neighbors,p_norm,3)
    
    # Plot the persistence diagram
    gd.plot_persistence_diagram(st.persistence())
    plt.show()  



