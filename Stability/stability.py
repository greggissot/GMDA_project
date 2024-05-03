import numpy as np
import gudhi as gd
from Filtration.generate_toy_dataset import generate_circle
from scipy.spatial.distance import directed_hausdorff
import ot
from tqdm import tqdm
from scipy.stats import describe
import matplotlib.pyplot as plt
import os 

from Filtration.DTM_filtration import DTM_filtration


def compute_wasserstein_distance(X1: np.ndarray, X2: np.ndarray) -> float:
    """Compute Wasserstein distance between two point clouds

    Args:
        X1 (np.ndarray): First point cloud 
        X2 (np.ndarray): Second point cloud

    Returns:
        float: Wasserstein distance between the two point clouds
    """    
    C = ot.dist(X1, X2)
    T = ot.sinkhorn([], [], C, 1)
    return np.sum(T * C)


def compute_hausdorff_distance(X1: np.ndarray, X2: np.ndarray) -> float:
    """Compute Hausdorff distance between two point clouds

    Args:
        X1 (np.ndarray): First point cloud
        X2 (np.ndarray): Second point cloud

    Returns:
        float: Hausdorff distance
    """    
    dist1_to_2 = directed_hausdorff(X1, X2)[0]
    dist2_to_1 = directed_hausdorff(X2, X1)[0]
    return max(dist1_to_2, dist2_to_1)


def compute_bottleneck_distance(pd1: np.ndarray, pd2: np.ndarray) -> float:
    """Compute bottleneck distance between two persistence diagrams

    Args:
        pd1 (np.ndarray): First persistence diagram
        pd2 (np.ndarray): Second persistence diagram

    Returns:
        float: Bottleneck distance
    """    
    return gd.bottleneck_distance(pd1, pd2)


def simulate_stability(n_points: int, k_neighbors: int, p_norm: int, dimension_max: int, epsilon: float, dim_used: int, save_fig: bool = False) -> float:
    """Simulate the stability of persistence diagrams

    Args:
        n_points (int): Number of points in the circle
        k_neighbors (int): Number of neighbors for the DTM filtration
        p_norm (int): p-norm for the DTM filtration
        dimension_max (int): Maximum dimension for the DTM filtration
        epsilon (float): Perturbation factor
        dim_used (int): Dimension of the persistence diagram to use to compute the Bottleneck distance
        save_fig (bool, optional): Save figure with the point clouds and persistence diagrams. Defaults to False.

    Returns:
        float: Ratio between the bottleneck distance and the combination of Hausdorff and Wasserstein distances
    """    
    # Generate a circle
    X_circle = generate_circle(n_points, 0)

    # Compute persistence diagrams for the original point cloud
    st =  DTM_filtration(X_circle, k_neighbors, p_norm, dimension_max)
    st.compute_persistence()
    dim_pd = st.persistence_intervals_in_dimension(dim_used)

    # Generate perturbed point cloud
    l = []
    while len(l) < len(X_circle):
        pt = np.random.rand(3) * 2 - 1  
        if np.linalg.norm(pt) <= 1:
            l.append(pt) 
    noise = np.array(l) * epsilon
    X_circle_perturbed = X_circle + noise

    # Compute persistence diagrams for the perturbed point cloud
    st_per =  DTM_filtration(X_circle_perturbed, k_neighbors, p_norm, dimension_max)
    st_per.compute_persistence()
    dim_pd_perturbed = st_per.persistence_intervals_in_dimension(dim_used)

    # Compute Wasserstein and Haussdorff distance between point clouds
    wasserstein_distance = compute_wasserstein_distance(X_circle, X_circle_perturbed)
    hausdorff_distance = compute_hausdorff_distance(X_circle, X_circle_perturbed)

    # Compute bottleneck distance between persistence diagrams
    bottleneck_distance = compute_bottleneck_distance(dim_pd, dim_pd_perturbed)

    # Compute ratio Bottleneck distance over combination of Hausdorff and Wasserstein Distances
    ratio = bottleneck_distance / (m**(-0.5) * wasserstein_distance + 2**(1/p_norm) * hausdorff_distance)

    if save_fig:
        # Save figure with four subplot, two points clouds and two persistence diagram
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs[0, 0].scatter(X_circle[:, 0], X_circle[:, 1])
        axs[0, 0].set_title('Original Point Cloud')
        axs[0, 1].scatter(X_circle_perturbed[:, 0], X_circle_perturbed[:, 1])
        axs[0, 1].set_title('Perturbed Point Cloud')
        gd.plot_persistence_diagram(st.persistence(), axes=axs[1, 0])
        axs[1, 0].set_title('Persistence Diagram Original')
        gd.plot_persistence_diagram(st_per.persistence(), axes=axs[1, 1])
        axs[1, 1].set_title('Persistence Diagram Perturbed')
        fig.savefig(f'results_{epsilon}_{dim_used}/pd_point_clouds.png')
        plt.close(fig)

    return ratio

if __name__ == '__main__':
    ############## YOU CAN MODIFY THE FOLLOWING LINES ##############
    num_iterations = 500 # Number of iterations
    k_neighbors = 50 # Number of nearest neighbors
    p_norm = 1 # p-norm
    dim_used = 0 # Dimension of the persistence diagram to use to compute the Bottleneck distance
    n_points = 100 # Number of points in the circle
    epsilon = 0.5 # Perturbation factor
    ################################################################
    m = k_neighbors/n_points
    dimension_max = 2 
    save_fig = False
    os.makedirs(f'results_{epsilon}_{dim_used}', exist_ok=True)


    # Launch the test N times and get store the ratios 
    ratios = []
    for _ in tqdm(range(num_iterations)):
        ratio = simulate_stability(n_points, k_neighbors, p_norm, dimension_max, epsilon, dim_used, save_fig)
        ratios.append(ratio)

    # Compute and save statistics for the ratios
    statistics = describe(ratios)

    with open(f'results_{epsilon}_{dim_used}/ratio_statistics.txt', 'w') as f:
        f.write("Statistics for the ratio:\n")
        f.write(f"Number of observations: {statistics.nobs}\n")
        f.write(f"Minimum: {statistics.minmax[0]}\n")
        f.write(f"Maximum: {statistics.minmax[1]}\n")
        f.write(f"Mean: {statistics.mean}\n")
        f.write(f"Variance: {statistics.variance}\n")
        f.write(f"Skewness: {statistics.skewness}\n")
        f.write(f"Kurtosis: {statistics.kurtosis}\n")

    plt.boxplot(ratios)
    plt.xlabel("Ratio")
    plt.ylabel("Frequency")
    plt.title("Boxplot of Ratios")
    plt.savefig(f'results_{epsilon}_{dim_used}/boxplot_ratios.png')
    plt.close()
