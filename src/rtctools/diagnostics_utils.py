import copy
import logging
import textwrap

import casadi as ca

import numpy as np


logger = logging.getLogger("rtctools")


def casadi_to_lp(casadi_equations, lp_name=None):
    """Convert the model as formulated with casadi types to a human-readable
    format.
    """
    n_dec = 4  # number of decimals
    try:
        d = casadi_equations
        indices = d["indices"][0]
        expand_f_g = d["func"]
        lbx, ubx, lbg, ubg, x0 = d["other"]
        X = ca.SX.sym("X", expand_f_g.nnz_in())
        f, g = expand_f_g(X)

        in_var = X
        out = []
        for o in [f, g]:
            Af = ca.Function("Af", [in_var], [ca.jacobian(o, in_var)])
            bf = ca.Function("bf", [in_var], [o])

            A = Af(0)
            A = ca.sparsify(A)

            b = bf(0)
            b = ca.sparsify(b)
            out.append((A, b))

        var_names = []
        for k, v in indices.items():
            if isinstance(v, int):
                var_names.append("{}__{}".format(k, v))
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

        # OBJECTIVE
        try:
            A, b = out[0]
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
        except Exception:
            logger.warning("Cannot convert non-linear objective! Objective string is set to 1")
            objective_str = "1"

        # CONSTRAINTS
        A, b = out[1]
        ca.veccat(*lbg)
        lbg = np.array(ca.veccat(*lbg))[:, 0]
        ubg = np.array(ca.veccat(*ubg))[:, 0]

        A_csc = A.tocsc()
        A_coo = A_csc.tocoo()
        b = np.array(b)[:, 0]

        constraints = [[] for i in range(A.shape[0])]

        for i, j, c in zip(A_coo.row, A_coo.col, A_coo.data):
            constraints[i].extend(["+" if c > 0 else "-", str(abs(round(c, n_dec))), var_names[j]])

        constraints_original = copy.deepcopy(constraints)
        for i in range(len(constraints)):
            cur_constr = constraints[i]
            lower, upper, b_i = round(lbg[i], n_dec), round(ubg[i], n_dec), round(b[i], n_dec)

            if len(cur_constr) > 0:
                if cur_constr[0] == "-":
                    cur_constr[1] = "-" + cur_constr[1]
                cur_constr.pop(0)

            c_str = " ".join(cur_constr)

            if np.isfinite(lower) and np.isfinite(upper) and lower == upper:
                constraints[i] = "{} = {}".format(c_str, lower - b_i)
            elif np.isfinite(lower) and np.isfinite(upper):
                constraints[i] = "{} <= {} <= {}".format(lower - b_i, c_str, upper - b_i)
            elif np.isfinite(lower):
                constraints[i] = "{} >= {}".format(c_str, lower - b_i)
            elif np.isfinite(upper):
                constraints[i] = "{} <= {}".format(c_str, upper - b_i)
            else:
                raise Exception(lower, b, constraints[i])

        constraints_str = "  " + "\n  ".join(constraints)

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

        return constraints, constraints_original, list(var_names)

    except Exception as e:
        message = (
            "Error occured while generating lp file! {}".format(e)
            + "\n Does the problem contain non-linear constraints?"
        )
        logger.error(message)
        raise Exception(message)
