import logging

import casadi as ca

logger = logging.getLogger("rtctools")

try:
    from casadi import interp1d
except ImportError:
    logger.warning('interp1d not available in this version of CasADi.  Linear interpolation will not work.')
    interp1d = None


def is_affine(e, v):
    try:
        Af = ca.Function('f', [v], [ca.jacobian(e, v)]).expand()
    except RuntimeError as e:
        if "'eval_sx' not defined for" in str(e):
            Af = ca.Function('f', [v], [ca.jacobian(e, v)])
        else:
            raise
    return (Af.sparsity_jac(0, 0).nnz() == 0)


def nullvertcat(*L):
    """
    Like vertcat, but creates an MX with consistent dimensions even if L is empty.
    """
    if len(L) == 0:
        return ca.DM(0, 1)
    else:
        return ca.vertcat(*L)


def reduce_matvec(e, v):
    """
    Reduces the MX graph e of linear operations on p into a matrix-vector product.

    This reduces the number of nodes required to represent the linear operations.
    """
    Af = ca.Function('Af', [ca.MX()], [ca.jacobian(e, v)])
    A = Af(ca.DM())
    return ca.reshape(ca.mtimes(A, v), e.shape)


def substitute_in_external(expr, symbols, values):
    if len(symbols) == 0 or all(isinstance(x, ca.DM) for x in expr):
        return expr
    else:
        f = ca.Function('f', symbols, expr)
        return f.call(values, True, False)


def interpolate(ts, xs, t, equidistant, mode=0):
    if interp1d is not None:
        if mode == 0:
            mode_str = 'linear'
        elif mode == 1:
            mode_str = 'floor'
        else:
            mode_str = 'ceil'
        return interp1d(ts, xs, t, mode_str, equidistant)
    else:
        # This interpolation routine may fail when there are nan values in the data xs.
        ts = ca.MX(ts)
        xs = ca.MX(xs)
        t = ca.MX(t)

        n_intervals = ts.numel() - 1
        n_evaluate = t.numel()

        left_ts = ca.repmat(ca.transpose(ts[:-1]), n_evaluate, 1)
        right_ts = ca.repmat(ca.transpose(ts[1:]), n_evaluate, 1)
        left_xs = xs[:-1]
        right_xs = xs[1:]
        t_mat = ca.repmat(t, 1, n_intervals)
        if mode == 0:
            is_in_interval = (left_ts <= t_mat) * (t_mat < right_ts)
            weight_left = (right_ts - t_mat) / (right_ts - left_ts)
            weight_right = (t_mat - left_ts) / (right_ts - left_ts)
            weight_left = weight_left * is_in_interval
            weight_right = weight_right * is_in_interval
            result = ca.mtimes(weight_left, left_xs)
            result += ca.mtimes(weight_right, right_xs)
            result += (t < ts[0]) * xs[0]
            result += (ts[-1] <= t) * xs[-1]
        elif mode == 1:
            is_in_interval = (left_ts <= t_mat) * (t_mat < right_ts)
            result = ca.mtimes(is_in_interval, left_xs)
            result += (t < ts[0]) * xs[0]
            result += (ts[-1] <= t) * xs[-1]
        else:
            is_in_interval = (left_ts < t_mat) * (t_mat <= right_ts)
            result = ca.mtimes(is_in_interval, right_xs)
            result += (t <= ts[0]) * xs[0]
            result += (ts[-1] < t) * xs[-1]
        return result
