Controlling a Cascade of Weirs:  Local PID Control vs. Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../images/lavant-2976738_1280.jpg

.. :href: https://pixabay.com/en/lavant-water-water-power-energy-2976738/
.. pixabay content is released under a CC0 Public Domain licence - no attribution needed

.. note::

    This is a more advanced example that implements multi-objective optimization
    in RTC-Tools. It also capitalizes on the homotopy techniques available in
    RTC-Tools. If you are a first-time user of RTC-Tools, see :doc:`basic`.

One of the advantages of using RTC-Tools for control is that it is capable of
making decisions that are optimal for the whole network and for all future
timesteps within the model time horizon. This is in contrast to local control
algorithms such as PID controllers, where the control decision must be made on
past states and local information alone. Furthermore, unlike a PID-style
controller, RTC-Tools does not have gain parameters that need to be tuned.

This example models a cascading channel system, and compares a local control
scheme using PID controllers with the RTC-Tools approach that uses Goal
Programming.


The Model
---------

For this example, water is flowing through a multilevel channel system. The
model has three channel sections. There is an inflow forcing at the upstream
boundary and a water level at the downstream boundary. The decision variables
are the flow rates (and by extension, the weir settings) between the channels.

In OpenModelica Connection Editor, the model looks like this:

.. image:: ../../images/channel_wave_damping_omedit.png

In text mode, the Modelica model looks as follows (with annotation statements
removed):

.. literalinclude:: ../../_build/mo/channel_wave_damping.mo
  :language: modelica
  :lineno-match:

.. note::

    In order to simulate and show how the PID controllers activate only once
    the incoming wave has propagated downstream, we will discretize the model
    in time with a resolution of 5 minutes. With our spatial resolution of 4
    level nodes per 20 km reach, this produces a CFL number of approximately 0.4.

    For optimization-based control, such a fine temporal resolution is not needed,
    as the system is able to look ahead and plan corrective measures ahead of time.
    In this case, CFL numbers of up to 1 or even higher are typically used.

    Nevertheless, in order to present a consistent comparison, a 5 minute
    time step is also used for the optimization example. It is easy to explore
    the effect of the time step size on the optimization results by changing the
    value of the ``step_size`` class variable.

To run the model with the local control scheme, we make a second model that
constrains the flow rates to the weir settings as determined by local PID
controller elements:

.. literalinclude:: ../../../examples/channel_wave_damping/model/ExampleLocalControl.mo
  :language: modelica
  :lineno-match:

The local control model makes use of a PID controller class:

.. literalinclude:: ../../../examples/channel_wave_damping/model/PIDController.mo
  :language: modelica
  :lineno-match:

.. important::

    Modellers should take care to set proper values for the initial
    derivatives, in order to avoid spurious waves at the start of the
    optimization run. In this example we assume a steady state initial
    condition, as indicated and enforced by the ``SteadyStateInitializationMixin``
    in the Python code.

The Optimization Problem
------------------------

Goals
'''''

In this model, we define a TargetLevelGoal to find a requested target level:

.. literalinclude:: ../../../examples/channel_wave_damping/src/example_optimization.py
  :language: python
  :pyobject: TargetLevelGoal
  :lineno-match:

We will later apply this goal to the upstream and middle channels.

You can read more about the components of goals in the documentation:
:doc:`../../optimization/multi_objective`.

Optimization Problem
''''''''''''''''''''

We construct the class by declaring it and inheriting the desired parent
classes.

.. literalinclude:: ../../../examples/channel_wave_damping/src/example_optimization.py
  :language: python
  :pyobject: ExampleOptimization
  :lineno-match:
  :end-before: Goal Programming Approach

The ``StepSizeParameterMixin`` defines the step size parameter and sets the optimization
time steps, while the ``SteadyStateInitializationMixin`` constrains the initial
conditions to be steady-state.

Next, we instantiate the goals. There are two water level goals, applied at the
upper and middle channels. The goals are very simple—they just target a
specific water level.

.. literalinclude:: ../../../examples/channel_wave_damping/src/example_optimization.py
  :language: python
  :pyobject: ExampleOptimization.path_goals
  :lineno-match:

We want to apply these goals to every timestep, so we use the ``path_goals()``
method. This is a method that returns a list of the path goals we defined above.
Note that with path goals, each timestep is implemented as an independent goal—
if we cannot satisfy our min/max on time step A, it will not affect our desire
to satisfy the goal at time step B.

For comparison, we also define an optimization problem that uses a local control
scheme. This example does not use any goals, as the flow rate is regulated by
the PID Controller.

.. literalinclude:: ../../../examples/channel_wave_damping/src/example_local_control.py
  :language: python
  :pyobject: ExampleLocalControl
  :lineno-match:


Run the Optimization Problem
''''''''''''''''''''''''''''

To make our script run, at the bottom of our file we just have to call the
``run_optimization_problem()`` method we imported on the optimization problem
classes we just created. We do this for both the local control model and the
goal programming model.

.. literalinclude:: ../../../examples/channel_wave_damping/src/example_optimization.py
  :language: python
  :lineno-match:
  :start-after: # Run

The Whole Script
''''''''''''''''

All together, all the scripts are as as follows:

``step_size_parameter_mixin.py``:

.. literalinclude:: ../../../examples/channel_wave_damping/src/step_size_parameter_mixin.py
  :language: python
  :lineno-match:

``steady_state_initialization_mixin.py``:

.. literalinclude:: ../../../examples/channel_wave_damping/src/steady_state_initialization_mixin.py
  :language: python
  :lineno-match:

``example_local_control.py``:

.. literalinclude:: ../../../examples/channel_wave_damping/src/example_local_control.py
  :language: python
  :lineno-match:

``example_optimization.py``:

.. literalinclude:: ../../../examples/channel_wave_damping/src/example_optimization.py
  :language: python
  :lineno-match:


Extracting Results
------------------

The results from the run are found in ``output/timeseries_export.csv`` and
``output/timeseries_export_local_control.csv``. Here are the results when
plotted using the python library matplotlib:

.. plot:: examples/pyplots/channel_wave_damping.py

In this example, the PID controller is tuned poorly and ends up amplifying
the incoming wave as it propagates downstream. The optimizing controller,
in contrast, does not amplify the wave and maintains the target water level
throughout the wave event.