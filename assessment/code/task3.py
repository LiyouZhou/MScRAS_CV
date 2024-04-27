import numpy as np


def load_var(name):
    """
    Load a variable from a csv file.
    """
    val = []
    with open(name + ".csv", "r") as fd:
        val = [float(x) for x in fd.read().strip().split(",")]
    return np.array(val)


def kalmanPredict(x, P, F, Q):
    """
    Kalman filter predict step.
    """
    xp = F @ x
    # predict state
    Pp = F @ P @ F.transpose() + Q
    # predict state covariance
    return xp, Pp


def kalmanUpdate(x, P, H, R, z):
    """
    Kalman filter update step.
    """
    # print(H.shape, P.shape, R.shape)
    S = H @ P @ H.transpose() + R
    # innovation covariance
    K = P @ H.transpose() @ np.linalg.inv(S)
    # Kalman gain
    zp = H @ x
    # predicted observation

    gate = (z - zp).transpose() @ np.linalg.inv(S) @ (z - zp)

    # if gate > 9.21:
    if gate > 10000000:
        plt.plot(z[0], z[1], "ro", label="excluded observation")
        print("Observation outside validation gate")
        xe = x
        Pe = P
    else:
        xe = x + K @ (z - zp)
        # estimated state
        Pe = P - K @ S @ K.transpose()
        # estimated covariance

    return xe, Pe


def kalmanTracking(z, dt=0.5):
    """
    Kalman filter for tracking a moving object.
    """
    # number of samples
    N = len(z[0])

    # fmt: off
    F = np.array([
        [1,dt,0,0],
        [0,1,0,0],
        [0,0,1,dt],
        [0,0,0,1]]) # CV motion model
    Q = np.array([
        [0.16,0,0,0],
        [0,0.36,0,0],
        [0,0,0.16,0],
        [0,0,0,0.36]]) # motion noise
    H = np.array([
        [1,0,0,0],
        [0,0,1,0]]) # Cartesian observation model
    R = np.array([
        [0.25, 0],
        [0, 0.25]]) # observation noise
    # fmt: on

    x = np.array([u[0], 0, v[0], 0]).transpose()
    # initial state
    P = Q
    # initial state covariance
    s = np.zeros([4, N])
    for i in range(N):
        xp, Pp = kalmanPredict(x, P, F, Q)
        x, P = kalmanUpdate(xp, Pp, H, R, z[:, i])
        s[:, i] = x
        # save current state

    px = s[0, :]  # NOTE: s(2, :) and s(4, :), not considered here,
    py = s[2, :]

    return px, py


if __name__ == "__main__":
    u = load_var("na")
    v = load_var("nb")
    x = load_var("x")
    y = load_var("y")

    px, py = kalmanTracking(np.array([u, v]))

    e = (px - x) ** 2 + (py - y) ** 2
    rms = np.sqrt(np.sum(e) / len(e))
    print("RMS Error: ", rms)
