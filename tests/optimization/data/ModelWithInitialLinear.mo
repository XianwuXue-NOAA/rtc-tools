model ModelWithInitialLinear
	Real x;
	Real w(start=0.0, fixed=true);
	Real alias;
	Real negative_alias;

	parameter Real k = 1.0;

	parameter Real u_max;
	input Real u(fixed=false, min = -2, max = u_max);

	output Real y;

	Real x_delayed;
	Real x_delayed_extra;

	input Real constant_input(fixed=true);
	output Real constant_output;

initial equation
	x = 1.1;

equation
	der(x) = k * x + u;
	der(w) = x;

	x_delayed = delay(x, 0.1);
	x_delayed_extra = delay(x, 0.125);

	alias = x;

	negative_alias = -x;

	y + x = 3.0;

	constant_output = constant_input;

end ModelWithInitialLinear;