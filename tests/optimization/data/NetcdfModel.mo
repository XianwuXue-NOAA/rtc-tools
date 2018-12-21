model NetcdfModel
	Real loc_a__x(start=1.1);
	Real loc_a__w(start=0.0);
	Real alias;

	parameter Real k = 1.0;

	input Real loc_b__u(fixed=false);

	output Real loc_c__y;

	output Real loc_a__z;

	input Real loc_a__x_delayed(fixed=false);

	output Real loc_c__switched;

	input Real loc_a__constant_input(fixed=true);
	output Real loc_a__constant_output;

equation
	der(loc_a__x) = k * loc_a__x + loc_b__u;
	der(loc_a__w) = loc_a__x;

	alias = loc_a__x;

	loc_c__y + loc_a__x = 3.0;

	loc_a__z = alias^2 + sin(time);

	loc_a__x_delayed = delay(loc_a__x, 0.1);

	if loc_a__x > 0.5 then
		loc_c__switched = 1.0;
	else
		loc_c__switched = 2.0;
	end if;

	loc_a__constant_output = loc_a__constant_input;

end NetcdfModel;