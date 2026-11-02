# Status and known limitations

NLCAS Notebook is a research prototype. This public summary covers issues
relevant to running the application; internal investigation logs and
conference working data are not distributed.

- **Reload and partial substitution:** An unexpected evaluation was observed
  during partial substitution after reloading a saved Maxima notebook. The
  cause is not established and a general fix has not been verified. Check
  intermediate symbolic results. The observed workflow used an explicit
  numeric evaluation with all variables bound instead of that substitution.
- **Sampled plots:** Inequality regions and contours are approximated on a
  finite grid. Narrow features can be missed. View bounds clip the picture;
  point labels do not certify optimality. Axes currently scale independently.
- **Adviser proposals:** Review references, axes, constraints and scalar
  coordinates before approving a specification. A proposal can fail to parse
  and may require refinement.
- **Optional runtimes:** CAS engines, numeric Python dependencies, model
  servers and weights must be installed/configured separately. Large model
  profiles in the repository are configuration examples, not bundled models.

See [plotting specification](docs/developer_manuals/36_cartesian_plot_layers.md)
and [setup guide](docs/setup_guide.md).
