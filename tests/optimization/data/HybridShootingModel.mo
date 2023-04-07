model HybridShootingModel
	Real x(start=1.1);
	Real w(start=0.0);
	Real a;

	parameter Real k = 1.0;

	input Real u(fixed=false);

equation
	der(x) = k * x + u;
	der(w) = x;
	a = x + w;

end HybridShootingModel;