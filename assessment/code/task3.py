import numpy as np
import fire
from scipy.optimize import minimize


def load_var(name):
    """
    Load a variable from a csv file.
    """
    val = []
    with open(name + ".csv", "r") as fd:
        val = [float(x) for x in fd.read().strip().split(",")]
    return np.array(val)


u = load_var("na")
v = load_var("nb")
x = load_var("x")
y = load_var("y")


def kalmanPredict(x, P, F, Q):
    """
    Kalman filter predict step.
    """
    # predict state
    xp = F @ x
    # predict state covariance
    Pp = F @ P @ F.transpose() + Q
    return xp, Pp


def kalmanUpdate(x, P, H, R, z):
    """
    Kalman filter update step.
    """
    # print(H.shape, P.shape, R.shape)
    # innovation covariance
    S = H @ P @ H.transpose() + R
    # Kalman gain
    K = P @ H.transpose() @ np.linalg.inv(S)
    # predicted observation
    zp = H @ x

    gate = (z - zp).transpose() @ np.linalg.inv(S) @ (z - zp)

    # if gate > 9.21:
    if gate > 10000000:
        print("Observation outside validation gate")
        xe = x
        Pe = P
    else:
        xe = x + K @ (z - zp)
        # estimated state
        Pe = P - K @ S @ K.transpose()
        # estimated covariance

    return xe, Pe


def kalmanTracking(
    z, measurement_noise=[0.25, 0.25], motion_noise=[0.16, 0.36, 0.16, 0.36]
):
    """
    Kalman filter for tracking a moving object.
    """
    # number of samples
    N = len(z[0])
    dt = 0.5

    # fmt: off
    # CV motion model
    F = np.array([
        [1,dt,0,0],
        [0,1,0,0],
        [0,0,1,dt],
        [0,0,0,1]])
    # motion noise
    Q = np.array([
        [motion_noise[0],0,0,0],
        [0,motion_noise[1],0,0],
        [0,0,motion_noise[2],0],
        [0,0,0,motion_noise[3]]]) # motion noise
    H = np.array([
        [1,0,0,0],
        [0,0,1,0]]) # Cartesian observation model
    R = np.array([
        [measurement_noise[0], 0],
        [0, measurement_noise[1]]]) # observation noise
    # fmt: on

    u = z[0]
    v = z[1]
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

    px = s[0, :]  # NOTE: s(1, :) and s(3, :), not considered here,
    py = s[2, :]

    vx = s[1, :]
    vy = s[3, :]

    return px, py, vx, vy


def objective_function(params):
    px, py, _, _ = kalmanTracking(
        np.array([u, v]),
        measurement_noise=params[:2],
        motion_noise=params[2:],
    )
    e = (px - x) ** 2 + (py - y) ** 2
    return np.sqrt(np.mean(e))


def main(
    measurement_noise=[0.25, 0.25],
    motion_noise=[0.16, 0.36, 0.16, 0.36],
    optimize=False,
):

    px, py, _, _ = kalmanTracking(np.array([u, v]), measurement_noise, motion_noise)

    e = (px - x) ** 2 + (py - y) ** 2
    rms = np.sqrt(np.sum(e) / len(e))
    print("RMS Error: ", rms)

    if optimize:
        params = measurement_noise + motion_noise
        res = minimize(
            objective_function,
            params,
            bounds=[(0, 1)] * len(params),
        )

        for val in res.x:
            print(f"{val:.2f}")

        motion_noise = res.x[2:]
        measurement_noise = res.x[:2]

        Q = np.array(
            [
                [motion_noise[0], 0, 0, 0],
                [0, motion_noise[1], 0, 0],
                [0, 0, motion_noise[2], 0],
                [0, 0, 0, motion_noise[3]],
            ]
        )  # motion noise

        R = np.array(
            [[measurement_noise[0], 0], [0, measurement_noise[1]]]
        )  # observation noise

        print("Optimised Q: ")
        print(Q)
        print("Optimised R: ")
        print(R)

        px, py, _, _ = kalmanTracking(
            np.array([u, v]),
            measurement_noise=res.x[:2],
            motion_noise=res.x[2:],
        )
        e = (px - x) ** 2 + (py - y) ** 2
        rms = np.sqrt(np.mean(e))
        print("Optimised RMS Error: ", rms)


if __name__ == "__main__":
    fire.Fire(main)
