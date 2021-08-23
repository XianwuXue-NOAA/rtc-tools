model ModelDelay
	Real x;
	Real w;
	input Real u(fixed=false);

	output Real x_delayed;
	output Real w_delayed;
equation
	der(x) = x + u;
	der(w) = x;

	x_delayed = delay(x, 0.1);
	w_delayed = delay(w, 0.1);
end ModelDelay;
