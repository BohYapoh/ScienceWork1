def Visualize(bloch, t_ev, bloch_sv_final):
    center = np.array([0, 0, 0])
    length = 1.2
    n = 100
    xs = np.linspace(0, 1, n)
    ys = np.sin(xs * 6 * np.pi)
    zs = np.cos(xs * 6 * np.pi)
    plt.style.use('_mpl-gallery')
    plt.figure(2)
    fig2, ax2d = plt.subplots()
    plt.plot(t_ev, bloch[:, 0], label='X')
    plt.plot(t_ev, bloch[:, 1], label='Y')
    plt.plot(t_ev, bloch[:, 2], label='Z')
    print(bloch[499, 0], bloch[499, 1], bloch[499, 2])
    plt.xlabel("time")
    plt.ylabel("parameter")
    plt.figure(1)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.quiver(*center, length, 0, 0, color='r', linewidth=1.5)  # X
    ax.quiver(*center, 0, length, 0, color='g', linewidth=1.5)  # Y
    ax.quiver(*center, 0, 0, length, color='b', linewidth=1.5)  # Z
    ax.set(xticklabels=[],
           yticklabels=[],
           zticklabels=[])
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.text(length, 0, 0, 'X', color='r')
    ax.text(0, length, 0, 'Y', color='g')
    ax.text(0, 0, length, 'Z', color='b')
    ax.set_box_aspect([1, 1, 1])
    u, v = np.mgrid[0:2 * np.pi:25j, 0:np.pi:25j]
    x_sphere = np.cos(u) * np.sin(v)
    y_sphere = np.sin(u) * np.sin(v)
    z_sphere = np.cos(v)
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color=(0, 0.6, 1, 0.4))
    point = ax.scatter([bloch[0, 0]], [bloch[0, 1]], [bloch[0, 2]], color='blue', s=50)
    quiv = ax.quiver(*center, [bloch[0, 0]], [bloch[0, 1]], [bloch[0, 2]], color='blue', linewidth=1.5)
    quiv1 = ax.quiver(*center, [bloch_sv_final[0]], [bloch_sv_final[1]], [bloch_sv_final[2]], color='brown', linewidth=1.5)
    for i in range(len(t_ev)):
        if i % 20 == 0:
            # point.remove()
            point = ax.scatter([bloch[i, 0]], [bloch[i, 1]], [bloch[i, 2]], color='red', s=50)
            quiv.remove()
            quiv = ax.quiver(*center, [bloch[i, 0]], [bloch[i, 1]], [bloch[i, 2]], color='red', linewidth=1.5)
            print(i)
            # point._offsets3d = ([bloch[i,0]], [bloch[i,1]], [bloch[i,2]])
            # ax.scatter([bloch[i,0]], [bloch[i,1]], [bloch[i,2]], color='red', s=50)
            quiv._offsets3d = ([bloch[i, 0]], [bloch[i, 1]], [bloch[i, 2]])
            plt.pause(0.01)
    quiv.remove()
    quiv = ax.quiver(*center, [bloch[i, 0]], [bloch[i, 1]], [bloch[i, 2]], color='red', linewidth=1.5)
    quiv._offsets3d = ([bloch[i, 0]], [bloch[i, 1]], [bloch[i, 2]])