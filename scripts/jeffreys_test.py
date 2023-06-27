# Simple test script for evaluating the Jeffreys divergence between two pairs of Gaussian energy distributions
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.integrate as spi

def gaussian_2d_pdf(x, mu=np.zeros(2), sigma=np.eye(2)):
    sigma_inv = np.linalg.inv(sigma)
    return np.exp(-0.5 * (x - mu).T @ sigma_inv @ (x - mu)) / (np.sqrt((2 * np.pi) ** 2) * np.sqrt(np.linalg.det(sigma)))

# unit variance gaussian
def gaussian_unit_var_pdf(s, a, mu=np.zeros(2), offset=0):
    if mu[0] == -2:
        offset = 100
    # negative pdf because low energy is better
    return -1 * (np.exp(-0.5 * ((s - mu[0]) ** 2 + (a - mu[1]) ** 2)) / (np.sqrt((2 * np.pi) ** 2)) + offset)

# numerically estimate Z integral
def estimate_Z(s, mu):
    a, b = mu[1] - 3, mu[1] + 3 # captures 99.7% of the mass
    f = lambda a: np.exp(-1 * gaussian_unit_var_pdf(s, a, mu))
    y, err = spi.quad(f, a, b)
    #assert err < 1e-6
    return y

# mu1 and mu2 parameterize the two gaussian energy functions we want to compare
def get_jeffreys(mu1, mu2):
    # numerically estimate integral
    def jeffreys(s):
        Z1, Z2 = estimate_Z(s, mu1), estimate_Z(s, mu2)
        a1, b1 = mu1[1] - 3, mu1[1] + 3
        a2, b2 = mu2[1] - 3, mu2[1] + 3
        f1 = lambda a: np.exp(-1 * gaussian_unit_var_pdf(s,a,mu1)) * (gaussian_unit_var_pdf(s,a,mu2) - gaussian_unit_var_pdf(s,a,mu1))
        f2 = lambda a: np.exp(-1 * gaussian_unit_var_pdf(s,a,mu2)) * (gaussian_unit_var_pdf(s,a,mu1) - gaussian_unit_var_pdf(s,a,mu2))
        y1, err1 = spi.quad(f1, a1, b1)
        y2, err2 = spi.quad(f2, a2, b2)
        #assert err1 < 1e-6 and err2 < 1e-6
        return y1/Z1 + y2/Z2
    return jeffreys

# test code
mu1 = [-2,0]
mu2 = [2,0]
# plot gaussians
x, y = np.meshgrid(np.linspace(-4,4,100), np.linspace(-4,4,100))
z1 = gaussian_unit_var_pdf(x,y,mu1)
z2 = gaussian_unit_var_pdf(x,y,mu2)
fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(x,y,z1,cmap='Blues')
# ax.plot_surface(x,y,z2,cmap='Greens')
# plt.show()

ax = plt.imshow(z2, cmap='Greens_r')
ticklabels = np.array([-4.,-3.2,-2.4,-1.6,-0.8,0.,0.8,1.6,2.4,3.2,4.])
plt.xticks(np.linspace(0,100,11), labels=ticklabels)
plt.yticks(np.linspace(0,100,11), labels=-1*ticklabels)
plt.colorbar()
plt.savefig('gaussian.png')

J = get_jeffreys(mu1, mu2)
x = np.linspace(-4,4,100)
y = [J(s) for s in x]
plt.clf()
plt.plot(x,y)
plt.savefig('jeffreys.png')

