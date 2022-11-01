# Author: Kristjan Orr Dadason 
# Date: 29/10/2022
# Project: PCA
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.



import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from tools import load_cancer


def standardize(X: np.ndarray) -> np.ndarray:
    '''
    Standardize an array of shape [N x 1]

    Input arguments:
    * X (np.ndarray): An array of shape [N x 1]

    Returns:
    (np.ndarray): A standardized version of X, also
    of shape [N x 1]
    '''
    return (X-np.mean(X))/np.std(X)

def scatter_standardized_dims(
    X: np.ndarray,
    i: int,
    j: int,
):
    '''
    Plots a scatter plot of N points where the n-th point
    has the coordinate (X_ni, X_nj)

    Input arguments:
    * X (np.ndarray): A [N x f] array
    * i (int): The first index
    * j (int): The second index
    '''
    X_hat = standardize(X)
    plt.scatter(X_hat[:,i], X_hat[:,j])
    plt.xlabel(f'Dimension {i}')
    plt.ylabel(f'Dimension {j}')
    plt.grid()



def _scatter_cancer():
    X, y = load_cancer()
    X_hat = standardize(X)
    # 5 by 6 subplots
    # scatter plot of dimension 0 vs all other dimensions
    print(X_hat[:,0])
    fig, ax = plt.subplots(5, 6)
    for i in range(5):
        for j in range(6):
            ax[i, j].scatter(X_hat[:, 0], X_hat[:, i*6+j])
            #ax[i, j].set_xlabel('Dimension 0')
            ax[i, j].set_xlabel(f'vs dim {i*6+j}')
            # take away numbers on x and y axis
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])



def _plot_pca_components():
    pca = PCA(n_components=30)
    X, y = load_cancer()
    X_hat = standardize(X)
    pca.fit_transform(X_hat)
    components = pca.components_
    """ for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(components[i])
        plt.title(f'PCA {i+1}')
        #plt.xticks([])
        #plt.yticks([])
        plt.grid() """
    for i in range(30):
        plt.subplot(5, 6, i+1)
        plt.plot(components[i])
        plt.title(f'PCA {i+1}')
        plt.xticks([])
        plt.yticks([])
        plt.grid()
        
    plt.show()


def _plot_eigen_values():
    pca = PCA(n_components=30)
    X, y = load_cancer()
    X_hat = standardize(X)
    pca.fit_transform(X_hat)
    eigenvalues = pca.explained_variance_
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue')
    plt.plot(eigenvalues)
    plt.grid()
    plt.show()


def _plot_log_eigen_values():
    pca = PCA(n_components=30)
    X, y = load_cancer()
    X_hat = standardize(X)
    pca.fit_transform(X_hat)
    eigenvalues = pca.explained_variance_
    plt.xlabel('Eigenvalue index')
    plt.ylabel('$\log_{10}$ Eigenvalue')
    #plt.ylabel('Eigenvalue')
    # plot log of eigenvalues
    plt.plot(np.log10(eigenvalues))
    #plt.gca().set_yscale('log')
    plt.grid()
    plt.show()


def _plot_cum_variance():
    pca = PCA(n_components=30)
    X, y = load_cancer()
    X_hat = standardize(X)
    pca.fit_transform(X_hat)
    eigenvalues = pca.explained_variance_
    cumulative_eigenvalues = np.cumsum(eigenvalues)
    percentage_variance = cumulative_eigenvalues/np.sum(eigenvalues)
    plt.plot(percentage_variance * 100)
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Percentage variance')
    plt.grid()
    plt.show()
    
    
def indep_plot_first_2():
    pca = PCA(n_components=30)
    X, y = load_cancer()
    X_hat = standardize(X)
    pca.fit_transform(X_hat)
    X_hat_pca = pca.transform(X_hat)
    # malignant = 0, benign = 1
    plt.scatter(X_hat_pca[y==0, 0], X_hat_pca[y==0, 1], color='red', marker='x')
    plt.scatter(X_hat_pca[y==1, 0], X_hat_pca[y==1, 1], color='blue', s=5)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend(['Malignant', 'Benign'])
    plt.grid()
    plt.show()
    
def indep_plot_first_3():
    pca = PCA(n_components=30)
    X, y = load_cancer()
    X_hat = standardize(X)
    pca.fit_transform(X_hat)
    X_hat_pca = pca.transform(X_hat)
    # malignant = 0, benign = 1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_hat_pca[y==0, 0], X_hat_pca[y==0, 1], X_hat_pca[y==0, 2], color='red',marker='x')
    ax.scatter(X_hat_pca[y==1, 0], X_hat_pca[y==1, 1], X_hat_pca[y==1, 2], color='blue',s=5)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    plt.legend(['Malignant', 'Benign'])
    plt.grid()
    plt.show()
    


if __name__ == '__main__':
    #print(standardize(np.array([[0, 0], [0, 0], [1, 1], [1, 1]])))
    """ X = np.array([
    [1, 2, 3, 4],
    [0, 0, 0, 0],
    [4, 5, 5, 4],
    [2, 2, 2, 2],
    [8, 6, 4, 2]])
    scatter_standardized_dims(X, 0, 2)
    plt.show() """
    #_scatter_cancer()
    #plt.show()
    #_plot_pca_components()
    #_plot_eigen_values()
    #_plot_log_eigen_values()
    #_plot_cum_variance()
    indep_plot_first_2()
    indep_plot_first_3()