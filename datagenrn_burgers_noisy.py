n=int(input())

x_train=np.random.uniform(-1,1,n)
t_train=np.random.uniform(0.01,1,n)
u_train=np.zeros(n)
noise=0.01
#solution
# u = -2v*(phi_x / phi)
import numpy as np
from scipy.integrate import quad

nu = 0.01/np.pi

# initial condition
def phi0(eta):
    return np.exp(-(np.cos(np.pi*eta)-1)/(2*nu*np.pi))

def phi(x, t):

    if t == 0:
        return phi0(x)

    prefactor = 1.0/np.sqrt(4*np.pi*nu*t)

    def integrand(eta):
        return (
            np.exp(-(x-eta)**2/(4*nu*t))
            * phi0(eta)
        )

    I, _ = quad(integrand, -10, 10)

    return prefactor*I

def phi_x(x, t):

    prefactor = 1.0/np.sqrt(4*np.pi*nu*t)

    def integrand(eta):

        G = np.exp(-(x-eta)**2/(4*nu*t))

        return (
            -(x-eta)/(2*nu*t)
            * G
            * phi0(eta)
        )

    I, _ = quad(integrand, -10, 10)

    return prefactor*I
for i in range(0, n):

   u_train[i]=-2*nu*phi_x(x_train[i], t_train[i])/phi(x_train[i], t_train[i])

u_train= u_train + noise*np.std(u_train)*np.random.randn(n)

x0= np.linspace(-1,1,100)[:,None]
t0 = np.zeros((100,1))
u0 = -np.sin(np.pi*x0)

x_train=x_train[:,None]
t_train=t_train[:,None]
u_train=u_train[:,None]

x_train = np.vstack([x_train, x0])
t_train = np.vstack([t_train, t0])
u_train = np.vstack([u_train, u0])
