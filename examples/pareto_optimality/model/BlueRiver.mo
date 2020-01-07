model BlueRiver
  import SI = Modelica.SIunits;

  // The inflow boundary condition node for the Trout Lake reservoir:
  Deltares.ChannelFlow.SimpleRouting.BoundaryConditions.Inflow TroutIn(Q(nominal = 0.8)) annotation(Placement(visible = true, transformation(origin = {-78, 95}, extent = {{-5, -5}, {5, 5}}, rotation = -90)));
  // The reservoir node for Trout Lake. The nominal value is important for the optimization accuracy and performance. Ideally, the nominal value, the bounds (maximum and minimum specified here) and the goals (specified in Python) have the same order of magnitue. 
  Deltares.ChannelFlow.SimpleRouting.Reservoir.Reservoir TroutLake(V(min = 0, max = 740089102.5285000, nominal = 616740918.7737500), Q_turbine(nominal = 10), Q_spill(nominal = 1), QIn.Q(nominal = 0.8), QOut.Q(nominal = 46.05282095), n_QForcing = 0) annotation(Placement(visible = true, transformation(origin = {-46, 81}, extent = {{-5, -5}, {5, 5}}, rotation = 0)));
  // the inflow point for lateral flow at Alder
  Deltares.ChannelFlow.SimpleRouting.BoundaryConditions.Inflow InflowAlder(Q(nominal = 0.2)) annotation(Placement(visible = true, transformation(origin = {16, 82}, extent = {{-10, -10}, {4, 4}}, rotation = 180)));  
  // The node that represents Alder
  Deltares.ChannelFlow.SimpleRouting.Nodes.Node Alder(nin = 2, nout = 1, n_QForcing = 0, QIn.Q(each nominal = 0.3), QOut.Q(each nominal = 10)) annotation(Placement(visible = true, transformation(origin = {-23, 63}, extent = {{-5, -5}, {5, 5}}, rotation = -90)));
  // The downstream node, River City
  Deltares.ChannelFlow.SimpleRouting.BoundaryConditions.Terminal RiverCity(Q(nominal = 10)) annotation(Placement(visible = true, transformation(origin = {-7, 31}, extent = {{-5, -5}, {5, 5}}, rotation = -90)));

  // from inflow gauge to the Trout Lake Reservoir
  Deltares.ChannelFlow.SimpleRouting.Branches.Steady Inflow_TroutLake annotation(Placement(visible = true, transformation(origin = {-57, 89}, extent = {{-10, -10}, {4, 4}}, rotation = -10)));  
  // the branch between Trout Lake Reservoir and Alder
  Deltares.ChannelFlow.SimpleRouting.Branches.Steady TroutLake_Alder annotation(Placement(visible = true, transformation(origin = {-30.5, 75.5}, extent = {{-3.5, -3.5}, {4, 4}}, rotation = -45)));
  // the lateral branch
  Deltares.ChannelFlow.SimpleRouting.Branches.Steady Lateral_Alder annotation(Placement(visible = true, transformation(origin = {-5, 72}, extent = {{-10, -10}, {4, 4}}, rotation = 180)));
  // the branch between Alder and RiverCity
  Deltares.ChannelFlow.SimpleRouting.Branches.Steady Alder_RiverCity annotation(Placement(visible = true, transformation(origin = {-15, 47}, extent = {{-4, -4}, {4, 4}}, rotation = -70)));

  // Inflow
  input SI.VolumeFlowRate TroutLake_Inflow(fixed = true);
  input SI.VolumeFlowRate Alder_Inflow(fixed = true);
  input SI.VolumeFlowRate TroutLake_Q_turbine(min = 0, max = 115.1323355, nominal = 46.05282095, fixed = false);

  // Reservoir storage volume
  output SI.Volume TroutLake_V;
  // Total reservoir outflow
  output SI.VolumeFlowRate TroutLake_Q_out(min = 0, max = 115.1323355, nominal = 46.05282095);
  // Reservoir turbine flow
  output SI.VolumeFlowRate TroutLake_Q_spill;
  output SI.VolumeFlowRate RiverCity_Q(min = 0, max = 1000, fixed = false);

equation
  connect(Alder.QOut[1], Alder_RiverCity.QIn) annotation(Line(points = {{-23, 59},{-16, 50}}));
  connect(Alder_RiverCity.QOut, RiverCity.QIn) annotation(Line(points = {{-14, 44}, {-7, 35}}));
  connect(TroutLake_Alder.QOut, Alder.QIn[1]) annotation(Line(points = {{-29, 74}, {-23, 67}}));
  connect(TroutLake.QOut, TroutLake_Alder.QIn) annotation(Line(points = {{-42, 81}, {-32, 77}}));
  connect(Lateral_Alder.QOut, Alder.QIn[2]) annotation(Line(points = {{-8, 75}, {-23, 67}}));
  connect(Inflow_TroutLake.QOut, TroutLake.QIn) annotation(Line(points = {{-55, 86}, {-50, 81}}));
  connect(TroutIn.QOut, Inflow_TroutLake.QIn) annotation(Line(points = {{-78, 91}, {-66, 88}}));
  connect(InflowAlder.QOut, Lateral_Alder.QIn) annotation(Line(points = {{13, 85}, {4, 75}}));

// Assign external time series (input and output) or values to the model objects
  TroutIn.Q = TroutLake_Inflow;
  InflowAlder.Q=Alder_Inflow;
  TroutLake.Q_turbine = TroutLake_Q_turbine;
  TroutLake.Q_spill = 0.0;
  TroutLake_Q_spill = TroutLake.Q_spill;
  TroutLake_Q_out = TroutLake.QOut.Q;
  TroutLake_V = TroutLake.V;
  RiverCity_Q = RiverCity.Q;
 
end BlueRiver;
