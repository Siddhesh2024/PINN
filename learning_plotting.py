import matplotlib.pyplot as plt


layers=[2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
model = PhysicsInformedNN(x_train, t_train, u_train, x_f_train, t_f_train, layers)
model.train(200000)




plt.figure(figsize=(8,4))

plt.imshow(
    U,
    extent=[-1,1,0,1],
    origin='lower',
    aspect='auto',
    cmap='jet'
)

plt.xlabel('x')
plt.ylabel('t')
plt.colorbar(label='u(x,t)')
plt.title('PINN Solution')
plt.show()



times = [0.0, 0.25, 0.5, 0.75, 1.0]

plt.figure(figsize=(8,5))

for tt in times:
    idx = np.argmin(np.abs(t - tt))
    plt.plot(x, U[idx], label=f't={t[idx]:.2f}')

plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.legend()
plt.grid(True)
plt.show()
