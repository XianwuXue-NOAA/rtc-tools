Customized variable updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example will illustrate how to implement an equation outside of a Modelica file
using customized variable updates.
This could be used to implement equations that are not supported by Modelica.

Consider the following model:

.. math::
    \frac{dx}{dt} &= y, \\
    y &= -\frac{2}{3600}x.

The equation for ``x`` is given in the Modelica file,
while the equation for ``y`` is implemented in the python code.

The Modelica file is given below.

.. literalinclude:: ../../../examples/simulation_with_customized_update/model/simple_model.mo
  :language: modelica

Since ``y`` is not defined in Modelica, it is defined here an input.

The optimization problem is given below.

.. literalinclude:: ../../../examples/simulation_with_customized_update/src/simple_model.py
  :language: python
  :pyobject: SimpleModel


The variable ``y`` is defined here by the method ``get_y``.
Furthermore, the method ``initialize`` is overwritten and consists of the following parts:

* Set ``y`` using a dummy variable.
  It is important that ``y`` is given some numeric value,
  since otherwise the default ``super().initialize()`` method will crash.
* Call the default initialize method ``super().initialize()``.
  This will read the initial value of``x`` from an input file.
* Set ``y`` using the ``get_y`` method.

Finally, the update method ``update`` is overwritten and consists of the following two parts:

* Call the default update method ``super().update(dt)`` to update ``x``.
  Since the default update method uses an implicit time-stepping scheme,
  it expects the value of ``y`` to be that of the new time.
  However, we cannot compute the new ``y`` yet,
  since it depends on the new ``x``.
  We therefore still pass the value of ``y`` at the previous time instead.
* Update ``y`` by calling ``self.set_var('y', self.get_y())``.

**WARNING: by default, rtc-tools uses an implicit time-stepping scheme
and it expects all inputs to correspond to the new time.
However, when using variables with a customized update,
these variables are not yet available at the new time and we pass
the value at the previous time instead.
This might result in numerical instabilites and inaccuracies.**

**NOTE**: if we would have implemented both equations in the Modelica file,
the numerical scheme would be given by the following two equations.

.. math::
    \frac{x(t_n) - x(t_{n-1})}{\Delta t} &= y(t_n), \\
    y(t_n) &= -\frac{2}{3600}x(t_n).

Instead, with this implementation,
the numerical scheme is given by

.. math::
    \frac{x(t_n) - x(t_{n-1})}{\Delta t} &= y(t_{n-1}), \\
    y(t_n) &= -\frac{2}{3600}x(t_n),

where :math:`t_i` denote the points in time.
Note that the right-hand-side in the first equation differs
in these implementations.
