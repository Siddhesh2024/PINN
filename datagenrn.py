t = np.linspace(0, 5, 200)[:, None]

g = 9.81
v0 = 20
theta = np.pi/4

x = v0*np.cos(theta)*t
y = v0*np.sin(theta)*t - 0.5*g*t**2

u = np.hstack((x, y))


noise = 0.5

u_noisy = u + noise*np.random.randn(*u.shape)


N_f = 1000
t_f = np.random.uniform(t.min(), t.max(), (N_f,1))
