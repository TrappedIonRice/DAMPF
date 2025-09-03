import numpy as np

def solve_rk23(fun, t0, y0, t_end, h0=1e-3, rtol=1e-6, atol=1e-9,
              safety=0.9, fac_min=0.2, fac_max=5.0, max_steps=100000):
    """
    Adaptive RK23 (Bogacki-Shampine 3(2)) solver.
    Parameters:
        fun: callable f(t, y) -> dy/dt (y can be numpy array)
        t0: initial time
        y0: initial state (scalar or array-like)
        t_end: final time (must be > t0 for forward integration)
        h0: initial step size
        rtol, atol: relative/absolute tolerances (for error test)
        safety: safety factor for step-size update
        fac_min, fac_max: min/max factor for step-size change
        max_steps: maximum number of accepted steps to avoid infinite loops
    Returns:
        ts: numpy array of times (including t0 and t_end)
        ys: numpy array of solution states with shape (len(ts), len(y0_flat))
             if y is scalar, returns shape (len(ts),)
    Notes:
        This implementation uses the standard Bogacki–Shampine coefficients:
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 k1)
        k3 = f(t + 3/4 h, y + 3/4 h k2)
        y_high (3rd order) = y + h*(2/9 k1 + 1/3 k2 + 4/9 k3)
        k4 = f(t + h, y_high)
        y_low  (2nd order) = y + h*(7/24 k1 + 1/4 k2 + 1/3 k3 + 1/8 k4)
        Error estimate = y_high - y_low
    """
    # normalize inputs
    t = float(t0)
    t_end = float(t_end)
    forward = (t_end > t0)
    if not forward:
        raise ValueError("This solver currently supports forward integration (t_end > t0).")
    y = np.array(y0, dtype=float)
    y_shape = y.shape
    y_flat = y.reshape(-1)  # work with 1D array
    n = y_flat.size

    h = float(h0)
    tiny = 1e-16

    ts = [t]
    ys = [y_flat.copy()]

    # FSAL: we can store k1 from previous step. Initialize as None so we compute it at first step.
    k1 = None

    steps = 0
    while t < t_end and steps < max_steps:
        # Ensure step doesn't overshoot
        if t + h > t_end:
            h = t_end - t
        if h <= 0:
            raise RuntimeError("Non-positive step size encountered.")

        # compute k1 (use previous k4 if FSAL available)
        if k1 is None:
            k1 = np.asarray(fun(t, y_flat)).reshape(-1)
        else:
            # k1 already stored from previous step's k4 (FSAL)
            pass

        # Stage 2
        y2 = y_flat + (h * 0.5) * k1
        k2 = np.asarray(fun(t + 0.5 * h, y2)).reshape(-1)

        # Stage 3
        y3 = y_flat + (h * 0.75) * k2
        k3 = np.asarray(fun(t + 0.75 * h, y3)).reshape(-1)

        # High-order (3rd) solution candidate
        y_high = y_flat + h * ( (2.0/9.0) * k1 + (1.0/3.0) * k2 + (4.0/9.0) * k3 )

        # Stage 4 (for low-order solution and FSAL)
        k4 = np.asarray(fun(t + h, y_high)).reshape(-1)

        # Low-order (2nd) solution
        y_low = y_flat + h * ( (7.0/24.0) * k1 + (1.0/4.0) * k2 + (1.0/3.0) * k3 + (1.0/8.0) * k4 )

        # Error estimate (vector)
        err_vec = y_high - y_low

        # compute normalized error using infinity norm (max over components)
        scale = atol + rtol * np.maximum(np.abs(y_flat), np.abs(y_high))
        # avoid division by zero
        scale[scale == 0.0] = tiny
        err_ratio = np.max(np.abs(err_vec) / scale)

        # decide accept/reject
        if err_ratio <= 1.0:
            # accept step
            t = t + h
            y_flat = y_high.copy()
            ts.append(t)
            ys.append(y_flat.copy())

            # FSAL: k4 becomes next step's k1
            k1 = k4.copy()

            steps += 1
        else:
            # reject: do not advance t or y. We'll recompute with smaller h.
            # k1 remains as before (we will recompute k1 on successful accept or keep FSAL)
            pass

        # update step size
        # high-order is 3 => use p = 3
        if err_ratio == 0:
            factor = fac_max
        else:
            factor = safety * (err_ratio ** (-1.0 / 3.0))
            # clamp factor
            factor = max(fac_min, min(fac_max, factor))
        h = h * factor

        # avoid extremely small step sizes
        if abs(h) < tiny:
            raise RuntimeError("Step size got too small.")

    if steps >= max_steps:
        raise RuntimeError("Maximum number of steps exceeded.")

    ts = np.array(ts)
    ys = np.array(ys)
    # reshape back if scalar or original shape
    if n == 1:
        ys = ys.reshape(-1)
    else:
        # each row is flattened; reshape each to original shape if requested by user
        # here we return shape (len(ts), *y_shape)
        ys = ys.reshape((len(ts),) + y_shape)

    return ts, ys

# ----------------------------
# Example: solve dy/dt = -y, y(0)=1 on t in [0,5]
if __name__ == "__main__":
    import math

    def f(t, y):
        # y can be 1D array
        return -y

    t0 = 0.0
    y0 = 1.0
    t_end = 5.0

    ts, ys = solve_rk23(f, t0, y0, t_end, h0=0.1, rtol=1e-6, atol=1e-9)

    # print summary
    print("num steps:", len(ts)-1)
    # compare with exact solution
    y_exact = np.exp(-ts)
    max_err = np.max(np.abs(ys - y_exact))
    print("max error vs exact:", max_err)

    # optional: plot step locations (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        plt.plot(ts, np.exp(-ts), '-', label='exact')
        plt.plot(ts, ys, 'o-', label='rk23')
        plt.legend()
        plt.xlabel('t')
        plt.ylabel('y(t)')
        plt.title('RK23 adaptive steps')
        plt.show()
    except Exception:
        pass
