.. _custom equations:

Custom equations
~~~~~~~~~~~~~~~~

This example illustrates how to implement custom equations in python.
This could be used to implement equations that are not supported by Modelica.

Consider the following model:

.. math::
    \frac{dx}{dt} &= y, \\
    y &= -\frac{2}{3600}x.

The equation for ``x`` is given in the Modelica file,
while the equation for ``y`` is implemented in the python code.

The Modelica file is given below.

.. literalinclude:: ../../../examples/simulation_with_custom_equations/model/simple_model.mo
  :language: modelica

Since ``y`` is not defined in Modelica, it is defined here an input.

This example illustrates two ways to implement the equation for ``y``.
The first method is the adviced method and implements the equation by
defining custom residuals using casadi symbols.
The results will be the same as when implementing all equations in the Modelica file.

The second method is slightly more flexible as it does not require casadi symbols,
but is more prone to numerical instabilities.
The results can also differ from when implementing all equations
in the Modelica file.

Method 1 (adviced): using custom residuals
------------------------------------------

The first method is based on adding custom residuals to the problem.
This is illustrated in the optimization problem below.

.. literalinclude:: ../../../examples/simulation_with_custom_equations/src/simple_model.py
  :language: python
  :pyobject: SimpleModel

The equation for ``y`` is implemented by defining custom residuals.
A residual is an expression that should equal ``0``.
In this case, the equation :math:`y=-(2/3600) x` can be rewritten as
:math:`y+(2/3600)x=0` and hence the residual is :math:`y+(2/3600)`.
The custom states are those that are defined as input states in Modelica,
but are implemented through equations in python.
In this case, the custom states just consist of ``y``.

Method 2 (less stable): using custom explicit variable updates
--------------------------------------------------------------

The alternative approach is based on adding explicit variable updates to the python code.
This is illustratetd in the optimization problem below.

.. literalinclude:: ../../../examples/simulation_with_custom_equations/src/simple_model.py
  :language: python
  :pyobject: SimpleModelLessStable

The equation for ``y`` is defined here by the method ``get_y``.
Furthermore, the method ``initialize`` is overwritten and consists of the following parts:

* Set ``y`` using a dummy variable.
  It is important that ``y`` is given some numeric value,
  since otherwise the default ``super().initialize()`` method will crash.
* Call the default initialize method ``super().initialize()``.
  This will read the initial value of ``x`` from an input file.

Finally, the update method ``update`` is overwritten and consists of the following two parts:

* Update ``y`` by calling ``self.set_var('y', self.get_y())``.
  Since the default update method uses an implicit time-stepping scheme,
  it expects the value of ``y`` to be that of the new time.
  However, we don't have the value of ``x`` at the new time yet,
  so we instead approximate the new value of ``y`` using the current value
  of ``x``.
* Call the default update method ``super().update(dt)`` to update ``x``.

**WARNING**: by default, rtc-tools uses an implicit time-stepping scheme
and it expects all inputs to correspond to the new time.
However, to update the input variables,
we can only use the values at the current time.
This might result in numerical instabilites and inaccuracies.

**NOTE**: if we would have implemented both equations in the Modelica file,
the numerical scheme would be given by the following two equations.

.. math::
    \frac{x(t_n) - x(t_{n-1})}{\Delta t} &= y(t_n), \\
    y(t_n) &= -\frac{2}{3600}x(t_n).

Instead, with this implementation,
the numerical scheme is given by

.. math::
    \frac{x(t_n) - x(t_{n-1})}{\Delta t} &= y(t_n), \\
    y(t_n) &= -\frac{2}{3600}x(t_{n-1}),

where :math:`t_i` denote the points in time.
Note that the right-hand-side in the second equation differs
in these implementations.
