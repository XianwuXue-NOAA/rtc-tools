from rtctools.optimization.collocated_integrated_optimization_problem \
    import CollocatedIntegratedOptimizationProblem
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.optimization.goal_programming_mixin \
    import GoalProgrammingMixin, StateGoal
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.util import run_optimization_problem


class StateRangeGoal(StateGoal):
    """
    A state goal applies for each state variable and each time step.
    This StateRangeGoal class is used in all volume and flow goals
    in this optimization problem.
    """
    def __init__(self, target_min, target_max, variable, priority, weight,
                 function_range, function_nominal=1):
        self.target_min = target_min
        self.target_max = target_max
        self.state = variable
        self.priority = priority
        self.weight = weight
        self.function_range = function_range
        self.function_nominal = function_nominal

    order = 1


class BlueRiver(GoalProgrammingMixin, CSVMixin, ModelicaMixin,
                CollocatedIntegratedOptimizationProblem):
    """
    An example in exploring pareto optimal solutions in goal programming
    in RTC-Tools
    """

# non-equidistant data is used in this example (based on a monthly basis)
    csv_equidistant = False

    def path_constraints(self, ensemble_member):
        # Call super() class to not overwrite default behaviour
        constraints = super().path_constraints(ensemble_member)
        # Constrain the volume of storage to remain within these limits
        constraints.append((self.state('TroutLake.V'), 12334818.3754750,
                            740089102.5285000))
        return constraints

    def path_goals(self):
        # Call super() class to not overwrite default behaviour
        g = super().path_goals()

        # The objectives (goals) for the reservoir system:
        # Reservoir volume goal for dam safety (maximum volue) and
        # minimum storage
        g.append(StateRangeGoal(505066826.8963750, 715419465.7775500,
                                'TroutLake_V', 1, 1.0, [0, 740089102.5285000]))
        # Reservoir volume goal for recreational boating
        g.append(StateRangeGoal(616740918.8, 629075737.1,
                                'TroutLake_V', 5, 3.5, [0, 740089102.5285000]))
        # Flow range goal for River City: ecology (minimum flow) and flood
        # protection (maximum flow)
        g.append(StateRangeGoal(7.044311236, 23.48103745,
                                'RiverCity_Q', 3, 1.0, [0, 100]))
        # Reservoir release goal for recreation rafting
        g.append(StateRangeGoal(self.get_timeseries('Rafting_Qmin'),
                                self.get_timeseries('Rafting_Qmax'),
                                'TroutLake_Q_out', 4, 1.0, [0, 100]))
        return g

    def solver_options(self):
        options = super().solver_options()
        solver = options['solver']
        options[solver]['print_level'] = 1
        return options


# Run
if __name__ == "__main__":
    run_optimization_problem(BlueRiver)
