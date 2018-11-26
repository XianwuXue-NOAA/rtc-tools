model Example
  // Structures
  Deltares.ChannelFlow.Hydraulic.BoundaryConditions.Discharge Discharge annotation(Placement(visible = true, transformation(origin = {-90, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 90)));
  Deltares.ChannelFlow.Hydraulic.BoundaryConditions.Level Level annotation(Placement(visible = true, transformation(origin = {90, 0}, extent = {{-10, 10}, {10, -10}}, rotation = 90)));
  Deltares.ChannelFlow.Hydraulic.Branches.HomotopicTrapezoidal upstream(
    theta = theta,
    semi_implicit_step_size = step_size,
    H_b_up = 15,
    H_b_down = 15,
    bottom_width_up = 50,
    bottom_width_down = 50,
    length = 20000,
    uniform_nominal_depth = 5,
    friction_coefficient = 35,
    n_level_nodes = 21,
    Q(each nominal = 100.0)
  ) annotation(Placement(visible = true, transformation(origin = {-60, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.Hydraulic.Branches.HomotopicTrapezoidal middle(
    theta = theta,
    semi_implicit_step_size = step_size,
    H_b_up = 10,
    H_b_down = 10,
    bottom_width_up = 50,
    bottom_width_down = 50,
    length = 20000,
    uniform_nominal_depth = 5,
    friction_coefficient = 35,
    n_level_nodes = 21,
    Q(each nominal = 100.0)
  ) annotation(Placement(visible = true, transformation(origin = {0, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.Hydraulic.Branches.HomotopicTrapezoidal downstream(
    theta = theta,
    semi_implicit_step_size = step_size,
    H_b_up = 5,
    H_b_down = 5,
    bottom_width_up = 50,
    bottom_width_down = 50,
    length = 20000,
    uniform_nominal_depth = 5,
    friction_coefficient = 35,
    n_level_nodes = 21,
    Q(each nominal = 100.0)
  ) annotation(Placement(visible = true, transformation(origin = {58, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.Hydraulic.Structures.DischargeControlledStructure dam_middle annotation(Placement(visible = true, transformation(origin = {30, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.Hydraulic.Structures.DischargeControlledStructure dam_upstream annotation(Placement(visible = true, transformation(origin = {-30, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));

  // Inputs
  input Modelica.SIunits.Position Level_H(fixed = true) = Level.H;
  input Modelica.SIunits.VolumeFlowRate Inflow_Q(fixed = true) = Discharge.Q;

  // Outputs
  output Modelica.SIunits.Position H_middle = middle.H[middle.n_level_nodes];
  output Modelica.SIunits.Position H_upstream = upstream.H[upstream.n_level_nodes];
  output Modelica.SIunits.VolumeFlowRate Q_in = Inflow_Q;

  // Parameters
  parameter Modelica.SIunits.Duration step_size;
  parameter Real theta;
equation
  connect(dam_middle.HQDown, downstream.HQUp) annotation(Line(points = {{38, 0}, {50, 0}}, color = {0, 0, 255}));
  connect(dam_middle.HQUp, middle.HQDown) annotation(Line(points = {{8, 0}, {22, 0}, {22, 0}, {22, 0}}, color = {0, 0, 255}));
  connect(dam_upstream.HQDown, middle.HQUp) annotation(Line(points = {{-22, 0}, {-8, 0}}, color = {0, 0, 255}));
  connect(dam_upstream.HQUp, upstream.HQDown) annotation(Line(points = {{-52, 0}, {-38, 0}}, color = {0, 0, 255}));
  connect(Discharge.HQ, upstream.HQUp) annotation(Line(points = {{-82, 0}, {-68, 0}}, color = {0, 0, 255}));
  connect(Level.HQ, downstream.HQDown) annotation(Line(points = {{66, 0}, {82, 0}, {82, 0}, {82, 0}}, color = {0, 0, 255}));
initial equation
  //der(downstream.H[1:upstream.n_level_nodes - 1]) = 0;//fill(0.0, upstream.n_level_nodes - 1);
  //der(middle.H[1:middle.n_level_nodes - 1]) = 0; //fill(0.0, middle.n_level_nodes - 1);
  //der(upstream.H[1:downstream.n_level_nodes - 1]) = 0;// fill(0.0, downstream.n_level_nodes - 1);
  //downstream.Q[2:downstream.n_level_nodes + 1] = Inflow_Q;
  //middle.Q[2:middle.n_level_nodes + 1] = Inflow_Q;
  //upstream.Q[2:upstream.n_level_nodes + 1] = Inflow_Q;
  // Start on set point
  //downstream.H[downstream.n_level_nodes + 0] = 10;
end Example;
