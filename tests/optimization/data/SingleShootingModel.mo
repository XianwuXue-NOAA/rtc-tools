model SingleShootingModel
	Real x(start=1.1);
	Real w(start=0.0);
	Real a;

	parameter Real k = 1.0;

	input Real u(fixed=false);

	input Real constant_input(fixed=true);

equation
	der(x) = k * x + u;
	der(w) = x;
	a = x + w + constant_input;

end SingleShootingModel;