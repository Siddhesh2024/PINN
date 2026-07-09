layers = [1,50,50,50,50, 50, 50, 50,2]

model = PINN(t, u_noisy, layers)

model.train(5000)


u_pred = model.predict(t)

plt.figure(figsize=(8,5))
plt.plot(u[:,0], u[:,1], label="Exact")
plt.scatter(u_pred[:,0], u_pred[:,1],, '--', label="PINN")
plt.scatter(u_noisy[:,0], u_noisy[:,1], s=10, alpha=0.4,
            label="Noisy data")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
