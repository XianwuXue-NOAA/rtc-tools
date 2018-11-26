model ExampleHybrid
  extends Example;
  PIDController middle_pid(
    state = dam_middle.HQUp.H,
    target_value = 15.0,
    P = -200.0,
    I = -0.01,
    D = 0.0,
    feed_forward = 100.0,
    control_action = dam_middle.Q
  );
  output Modelica.SIunits.VolumeFlowRate Q_dam_middle = dam_middle.Q;
  input Modelica.SIunits.VolumeFlowRate Q_dam_upstream(fixed = false, min = 0.0, max = 1000.0, nominal = 100.0) = dam_upstream.Q;
initial equation
  //upstream.H[4] = 20.0;
  //middle.H[middle.n_level_nodes] = 15;
  upstream.H[upstream.n_level_nodes] = 20;
end ExampleHybrid;
