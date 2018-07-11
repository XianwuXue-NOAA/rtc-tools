Cascading Channels: Modeling Channel Hydraulics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../images/pont-du-gard-1742029_1280.jpg

.. :href: https://pixabay.com/en/pont-du-gard-aqueduct-roman-unesco-1742029/
.. pixabay content is released under a CC0 Public Domain licence - no attribution needed

.. note::

    This is a more advanced example that implements multi-objective optimization
    in RTC-Tools. It also capitalizes on the homotopy techniques available in
    RTC-Tools. If you are a first-time user of RTC-Tools, see :doc:`basic`.

Goal programming is a way to satisfy (sometimes conflicting) goals by ranking
the goals by priority. In this example, we specify two goals. The higher
priority goal will be to maintain the water levels in the channels within a
desired band. The lower priority goal will be to extract water to meet a
forecasted drinking water demand.

The Model
---------

For this example, water is flowing through a multilevel channel system. The
model has three channel sections. There is an extraction pump at the downstream
end of the middle channel. The algorithm will first attempt to maintain water
levels in the channels within the desired water level band. Using the remaining
flexibility in the model, the algorithm will attempt to meet the diurnal
demand pattern as best as it can with the extraction pump.

In OpenModelica Connection Editor, the model looks like this:

.. image:: ../../images/cascading_channels_omedit.png

In text mode, the Modelica model looks as follows (with annotation statements
removed):

.. literalinclude:: ../../_build/mo/cascading_channels.mo
  :language: modelica
  :lineno-match:

The Optimization Problem
------------------------

The python script consists of the following blocks:

* Import of packages
* Declaration of Goals
* Declaration of the optimization problem class

  * Constructor
  * Implementation of ``pre()`` method
  * Implementation of ``path_goals()`` method

* A run statement


Goals
'''''

In this model, we define two generic StateGoal subclasses:

.. literalinclude:: ../../../examples/cascading_channels/src/example.py
  :language: python
  :pyobject: RangeGoal
  :lineno-match:

.. literalinclude:: ../../../examples/cascading_channels/src/example.py
  :language: python
  :pyobject: TargetGoal
  :lineno-match:

These goals are actually really similar. The only difference is that the
``TargetGoal`` uses the same timeseries for its ``target_max`` and
``target_min`` attributes. This goal will try to minimize the difference between
the target and the goal's state. This is in contrast to the ``RangeGoal``, which
has a separate min and max that define an acceptable range of values.

You can read more about the components of goals in the documentation:
:doc:`../../optimization/multi_objective`.

.. _cascading-channels-optimization-problem:

Optimization Problem
''''''''''''''''''''

We construct the class by declaring it and inheriting the desired parent
classes. An interesting feature of this model is that we use the
``SmoothDynamicsMixin``. This mixin takes advantage of any remaining flexibility
in the model and minimizes the derivatives of the control variables. As a
result, the flow through the control structures is as smooth as possible. To see
how you might do this with a normal goal, see the ``MinimizeChangeInQpumpGoal``
in :ref:`goal-programming-declaring-goals`.

.. literalinclude:: ../../../examples/cascading_channels/src/example.py
  :language: python
  :pyobject: Example
  :lineno-match:
  :end-before: channels

In our new class, we implement the ``pre()`` method. This method is a good place
to do some preprocessing of the data to make sure it is all there when the model
runs.

.. literalinclude:: ../../../examples/cascading_channels/src/example.py
  :language: python
  :pyobject: Example.pre
  :lineno-match:

Next, we instantiate the goals. The highest priority goal in this example will
be to keep the water levels within a desired range. We apply this goal
iteratively over all the water level states, and give them a priority of 1. The
second goal is to track a target extraction flow rate with the extraction pump.
We give this goal a priority of 2.

.. literalinclude:: ../../../examples/cascading_channels/src/example.py
  :language: python
  :pyobject: Example.path_goals
  :lineno-match:

We want to apply these goals to every timestep, so we use the ``path_goals()``
method. This is a method that returns a list of the path goals we defined above.
Note that with path goals, each timestep is implemented as an independent goal—
if we cannot satisfy our min/max on time step A, it will not affect our desire
to satisfy the goal at time step B. Goals that inherit ``StateGoal`` are always
path goals.

Run the Optimization Problem
''''''''''''''''''''''''''''

To make our script run, at the bottom of our file we just have to call
the ``run_optimization_problem()`` method we imported on the optimization
problem class we just created.

.. literalinclude:: ../../../examples/cascading_channels/src/example.py
  :language: python
  :lineno-match:
  :start-after: # Run

The Whole Script
''''''''''''''''

All together, the whole example script is as follows:

.. literalinclude:: ../../../examples/cascading_channels/src/example.py
  :language: python
  :lineno-match:

Extracting Results
------------------

The results from the run are found in ``output/timeseries_export.csv``. Any
CSV-reading software can import it, but this is how results can be plotted using
the python library matplotlib:

.. plot:: examples/pyplots/cascading_channels_results.py