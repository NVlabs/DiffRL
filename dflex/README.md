# A Differentiable Multiphysics Engine for PyTorch

dFlex is a physics engine for Python. It is written entirely in PyTorch and supports reverse mode differentiation w.r.t. to any simulation inputs.

It includes a USD-based visualization library (`dflex.render`), which can generate time-sampled USD files, or update an existing stage on-the-fly.

## Prerequisites

* Python 3.6
* PyTorch 1.4.0 or higher
* Pixar USD lib (for visualization)

Pre-built USD Python libraries can be downloaded from https://developer.nvidia.com/usd, once they are downloaded you should follow the instructions to add them to your PYTHONPATH environment variable.

## Using the built-in backend

By default dFlex uses the built-in PyTorch cpp-extensions mechanism to compile auto-generated simulation kernels. 

- Windows users should ensure they have Visual Studio 2019 installed

## Setup and Running

To use the engine you can import first the simulation module:

```python
    import dflex.sim
```

To build physical models there is a helper class available in `dflex.sim.ModelBuilder`. This can be used to create models programmatically from Python. For example, to create a chain of particles:

```python
    builder = dflex.sim.ModelBuilder()

    # anchor point (zero mass)
    builder.add_particle((0, 1.0, 0.0), (0.0, 0.0, 0.0), 0.0)

    # build chain
    for i in range(1,10):
        builder.add_particle((i, 1.0, 0.0), (0.0, 0.0, 0.0), 1.0)
        builder.add_spring(i-1, i, 1.e+3, 0.0, 0)

    # add ground plane
    builder.add_shape_plane((0.0, 1.0, 0.0, 0.0), 0)
```

Once you have built your model you must convert it to a finalized PyTorch simulation data structure using `finalize()`:

```python
    model = builder.finalize('cpu')
```


The model object represents static (non-time varying) data such as constraints, collision shapes, etc. The model is stored in PyTorch tensors, allowing differentiation with respect to both model and state.

## Time Stepping

To advance the simulation forward in time (forward dynamics), we use an `integrator` object. dFlex currently offers semi-implicit and fully implicit (planned), via. the `dflex.sim.ExplicitIntegrator`, and `dflex.sim.ImplicitIntegrator` classes as follows:

```python
    sim_dt = 1.0/60.0
    sim_steps = 100

    integrator = dflex.sim.ExplicitIntegrator()

    for i in range(0, sim_steps):
        state = integrator.forward(model, state, sim_dt)
```

## Rendering

To visualize the scene dFlex supports a USD-based update via. the `dflex.render.UsdRenderer` class. To create a renderer you must first create the USD stage, and the physical model.

```python
    import dflex.render

    stage = Usd.Stage.CreateNew("test.usda")

    renderer = dflex.render.UsdRenderer(model, stage)
    renderer.draw_points = True
    renderer.draw_springs = True
    renderer.draw_shapes = True
```

Each frame the renderer should be updated with the current model state and the current elapsed simulation time:

```python
    renderer.update(state, sim_time)
```

## Contact

Miles Macklin (mmacklin@nvidia.com)