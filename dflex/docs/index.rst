Welcome to dFlex's documentation!
==================================

dFlex is a differentiable multiphysics engine for PyTorch. It is written entirely in Python and supports reverse mode differentiation w.r.t. to any simulation inputs.

It includes a USD-based visualization module (:class:`dflex.render`), which can generate time-sampled USD files, or update an existing stage on-the-fly.

Prerequisites
-------------

* Python 3.6
* PyTorch 1.4.0 or higher
* Pixar USD lib (for visualization)

Pre-built USD Python libraries can be downloaded from https://developer.nvidia.com/usd, once they are downloaded you should follow the instructions to add them to your PYTHONPATH environment variable.


.. toctree::
   :maxdepth: 3
   :caption: Contents:

   modules/model
   modules/sim
   modules/render

Quick Start
-----------------

First ensure that the package is installed in your local Python environment (use the -e option if you will be doing development):

.. code-block::
	
    pip install -e dflex

Then, to use the engine you can import the simulation module as follows:

.. code-block::
	
    import dflex

To build physical models there is a helper class available in :class:`dflex.model.ModelBuilder`. This can be used to create models programmatically from Python. For example, to create a chain of particles:

.. code-block::

    builder = dflex.model.ModelBuilder()

    # anchor point (zero mass)
    builder.add_particle((0, 1.0, 0.0), (0.0, 0.0, 0.0), 0.0)

    # build chain
    for i in range(1,10):
        builder.add_particle((i, 1.0, 0.0), (0.0, 0.0, 0.0), 1.0)
        builder.add_spring(i-1, i, 1.e+3, 0.0, 0)

    # add ground plane
    builder.add_shape_plane((0.0, 1.0, 0.0, 0.0), 0)


Once you have built your model you must convert it to a finalized PyTorch simulation data structure using :func:`dflex.model.ModelBuilder.finalize()`:

.. code-block::

    model = builder.finalize('cpu')



The model object represents static (non-time varying) data such as constraints, collision shapes, etc. The model is stored in PyTorch tensors, allowing differentiation with respect to both model and state.

Time Stepping
-------------

To advance the simulation forward in time (forward dynamics), we use an `integrator` object. dFlex currently offers semi-implicit and fully implicit (planned), via. the :class:`dflex.sim.SemiImplicitIntegrator` class as follows:

.. code-block::

    sim_dt = 1.0/60.0
    sim_steps = 100

    integrator = dflex.sim.SemiImplicitIntegrator()

    for i in range(0, sim_steps):
        state = integrator.forward(model, state, sim_dt)

Rendering
---------

To visualize the scene dFlex supports a USD-based update via. the :class:`dflex.render.UsdRenderer` class. To create a renderer you must first create the USD stage, and the physical model.

.. code-block::

    import dflex.render

    stage = Usd.Stage.CreateNew("test.usda")

    renderer = dflex.render.UsdRenderer(model, stage)
    renderer.draw_points = True
    renderer.draw_springs = True
    renderer.draw_shapes = True


Each frame the renderer should be updated with the current model state and the current elapsed simulation time:

.. code-block::

    renderer.update(state, sim_time)



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
