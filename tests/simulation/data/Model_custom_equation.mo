model Model_custom_equation
    Real x(start=2.0);
    input Real y;
equation
    der(x) = y;
    // y implemented in python (y = -2 * x)
end Model_custom_equation;
