model SimpleModel
    input Real y;  // Depends on x through a formula that is implemented in python.
    output Real x;
equation
    der(x) = y;
end SimpleModel;