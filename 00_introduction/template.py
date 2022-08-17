import numpy as np
import matplotlib.pyplot as plt


def normal(x: np.ndarray, sigma: float, mu: float) -> np.ndarray:
    # Part 1.1
    return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2)/(2 * sigma**2))

#print(normal(0, 1, 0))
#print(normal(3, 1, 5))
#print(normal(np.array([-1,0,1]), 1, 0))

def plot_normal(sigma: float, mu:float, x_start: float, x_end: float):
    # Part 1.2
    x_range = np.linspace(x_start, x_end, 500)
    plt.plot(x_range, normal(x_range, sigma, mu))

#plot_normal(0.5, 0, -2, 2)

def _plot_three_normals():
    # Part 1.2
    # plot three normal distributions with different parameters on the same figure
    # use the function plot_normal
    fig = plt.figure()
    sigma = [0.5, 1.5, 0.25]
    mu = [0, -0.5, 1.5]
    x_start = -5
    x_end = 5
    for i in range(len(sigma)):
        fig = plot_normal(sigma[i], mu[i], x_start, x_end)
    return fig


def normal_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
    # Part 2.1
    p = np.zeros(len(x))
    for i in zip(sigmas, mus, weights):
        p += normal(x, i[0], i[1]) * i[2]
    return p


def _compare_components_and_mixture():
    #mynd = _plot_three_normals()
    NN = normal_mixture(np.linspace(-5, 5, 500), [0.5, 1.5, 0.25],[0, -0.5, 1.5], [1/3, 1/3, 1/3])
    plt.plot(np.linspace(-5, 5, 500), NN)
    return 0

def sample_gaussian_mixture(sigmas: list, mus: list, weights: list, n_samples: int = 500):
    # Part 3.1
    throw = np.random.multinomial(n_samples, weights)
    samples = []
    # sample from each gaussian
    for i in range(len(sigmas)):
        for j in range(throw[i]):
            samples.append(np.random.normal(mus[i], sigmas[i]))
    # return the samples as a numpy array
    return np.array(samples)
    
    
    
    
def _plot_mixture_and_samples(n_samples=100):
    # Part 3.2  
    #n_samples = 100
    x = np.linspace(-10, 10, n_samples)
    NN = sample_gaussian_mixture([0.3, 0.5, 1.0], [0, -1, 1.5], [0.2, 0.3, 0.5], n_samples)
    #plt.plot(x, NN)
    plt.hist(NN, 100, density=True)
    ss = normal_mixture(x, [0.3, 0.5, 1.0], [0, -1, 1.5], [0.2, 0.3, 0.5])
    plt.plot(x, ss)

if __name__ == '__main__':
    # select your function to test here and do `python3 template.py`
    #_compare_components_and_mixture()
    #plt.savefig('1_2_1.png')
    # save figure as 2_2_1.png to the plots folder
    #plt.savefig('C:/T809DATA_2022_KOD/00_introduction/plots/2_2_1.png')
    #plt.show()
    # save the figure to a file
    #print(normal_mixture(np.linspace(-5, 5, 5), [0.5, 0.25, 1], [0, 1, 1.5], [1/3, 1/3, 1/3]))
    #print(normal_mixture(np.linspace(-2, 2, 4), [0.5], [0], [1]))
    np.random.seed(0)
    #sample_gaussian_mixture([0.3, 0.5, 1.0], [0, -1, 1.5], [0.2, 0.3, 0.5], 10)
    plt.subplot(141)
    _plot_mixture_and_samples(10)
    plt.subplot(142)
    _plot_mixture_and_samples(100)
    plt.subplot(143)
    _plot_mixture_and_samples(500)
    plt.subplot(144)
    _plot_mixture_and_samples(1000)
    #plt.savefig('C:/T809DATA_2022_KOD/00_introduction/plots/3_2_1.png')
    plt.show()
    