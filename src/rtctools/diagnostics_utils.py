import copy
import logging
import textwrap

import casadi as ca

import numpy as np


logger = logging.getLogger("rtctools")


def convert_constraints(constraints, lbg, ubg, b, n_dec):
    # lbg = np.array(ca.veccat(*lbg))[:, 0]
    # ubg = np.array(ca.veccat(*ubg))[:, 0]
    b = np.array(b)[:, 0]
    lbg = np.array(lbg)
    ubg = np.array(ubg)
    constraints_converted = copy.deepcopy(constraints)
    for i, _ in enumerate(constraints_converted):
        cur_constr = constraints_converted[i]
        lower, upper, b_i = round(lbg[i], n_dec), round(ubg[i], n_dec), round(b[i], n_dec)

        if len(cur_constr) > 0:
            if cur_constr[0] == "-":
                cur_constr[1] = "-" + cur_constr[1]
            cur_constr.pop(0)

        c_str = " ".join(cur_constr)

        if np.isfinite(lower) and np.isfinite(upper) and lower == upper:
            constraints_converted[i] = "{} = {}".format(c_str, lower - b_i)
        elif np.isfinite(lower) and np.isfinite(upper):
            constraints_converted[i] = "{} <= {} <= {}".format(lower - b_i, c_str, upper - b_i)
        elif np.isfinite(lower):
            constraints_converted[i] = "{} >= {}".format(c_str, lower - b_i)
        elif np.isfinite(upper):
            constraints_converted[i] = "{} <= {}".format(c_str, upper - b_i)
        else:
            raise Exception(lower, b, constraints_converted[i])
    return constraints_converted


def get_varnames(casadi_equations):
    indices = casadi_equations["indices"][0]
    expand_f_g = casadi_equations["func"]

    var_names = []
    for k, v in indices.items():
        if isinstance(v, int):
            var_names.append("{}__{}".format(k, v))
        elif isinstance(v, np.ndarray):
            for i in np.arange(0, len(v)):
                var_names.append("{}__{}".format(k, i))
        else:
            for i in range(0, v.stop - v.start, 1 if v.step is None else v.step):
                var_names.append("{}__{}".format(k, i))

    n_derivatives = expand_f_g.nnz_in() - len(var_names)
    for i in range(n_derivatives):
        var_names.append("DERIVATIVE__{}".format(i))

    # CPLEX does not like [] in variable names
    for i, v in enumerate(var_names):
        v = v.replace("[", "_I")
        v = v.replace("]", "I_")
        var_names[i] = v

    return var_names


def get_systems_of_equations(casadi_equations):
    expand_f_g = casadi_equations["func"]
    X = ca.SX.sym("X", expand_f_g.nnz_in())
    f, g = expand_f_g(X)
    eq_systems = []
    for o in [f, g]:
        Af = ca.Function("Af", [X], [ca.jacobian(o, X)])
        bf = ca.Function("bf", [X], [o])

        A = Af(0)
        A = ca.sparsify(A)

        b = bf(0)
        b = ca.sparsify(b)
        eq_systems.append((A, b))

    return {key: value for key, value in zip(["objective", "constraints"], eq_systems)}


def casadi_to_lp(casadi_equations, lp_name=None):
    """Convert the model as formulated with casadi types to a human-readable
    format.
    """
    n_dec = 4  # number of decimals
    try:
        lbx, ubx, lbg, ubg, x0 = casadi_equations["other"]
        eq_systems = get_systems_of_equations(casadi_equations)
        var_names = get_varnames(casadi_equations)

        # OBJECTIVE
        try:
            A, b = eq_systems["objective"]
            objective = []
            ind = np.array(A)[0, :]

            for v, c in zip(var_names, ind):
                if c != 0:
                    objective.extend(["+" if c > 0 else "-", str(abs(c)), v])

            if objective[0] == "-":
                objective[1] = "-" + objective[1]

            objective.pop(0)
            objective_str = " ".join(objective)
            objective_str = "  " + objective_str
        except IndexError:
            logger.warning("Cannot convert non-linear objective! Objective string is set to 1")
            objective_str = "1"

        # CONSTRAINTS
        A, b = eq_systems["constraints"]

        A_csc = A.tocsc()
        A_coo = A_csc.tocoo()

        constraints = [[] for i in range(A.shape[0])]

        for i, j, c in zip(A_coo.row, A_coo.col, A_coo.data):
            constraints[i].extend(["+" if c > 0 else "-", str(abs(round(c, n_dec))), var_names[j]])

        converted_constraints = convert_constraints(constraints, lbg, ubg, b, n_dec)
        constraints_str = "  " + "\n  ".join(converted_constraints)

        # Bounds
        bounds = []
        for v, lower, upper in zip(var_names, lbx, ubx):
            bounds.append("{} <= {} <= {}".format(lower, v, upper))
        bounds_str = "  " + "\n  ".join(bounds)
        if lp_name:
            with open("myproblem_{}.lp".format(lp_name), "w") as o:
                o.write("Minimize\n")
                for x in textwrap.wrap(
                    objective_str, width=255
                ):  # lp-format has max length of 255 chars
                    o.write(x + "\n")
                o.write("Subject To\n")
                o.write(constraints_str + "\n")
                o.write("Bounds\n")
                o.write(bounds_str + "\n")
                o.write("End")
            with open("constraints.lp", "w") as o:
                o.write(constraints_str + "\n")

        return constraints

    except Exception as e:
        message = (
            "Error occured while generating lp file! {}".format(e)
            + "\n Does the problem contain non-linear constraints?"
        )
        logger.error(message)
        raise Exception(message)
