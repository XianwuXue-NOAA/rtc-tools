model PIDController
  input Real state;
  parameter Real target_value;
  parameter Real P = 1.0;
  parameter Real I = 0.0;
  parameter Real D = 0.0;
  parameter Real feed_forward = 0.0;
  output Real control_action;
  Real _error;
  Real _error_integral(nominal = 1e6);
equation
  _error = target_value - state;
  der(_error_integral) = _error;
  control_action = feed_forward + P * _error + I * _error_integral + D * der(_error);
initial equation
  _error_integral = 0.0;
end PIDController;