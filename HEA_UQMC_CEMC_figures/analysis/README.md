# Activity metrics and selection

Run `postprocess_activity.py --help` for arguments. This portable adapter uses the original Study 2/3 scaling, metrics, temperature selection, and smoothed-histogram selection functions in `selection_core.py`. It accepts header-based `.f32` sidecars or raw float32 `.bin` shards with adjacent JSON metadata. Each slab is logged before averaging over runs. The saved experimental values and map layouts come from `analysis_data/<dataset>/`.

The output is a new analysis directory. Existing publication inputs are not overwritten. To plot a new analysis, copy the corresponding generated tables into a separate copy of the relevant figure folder. Do not replace the historical data unless that is intended. Study 1 uses its own scoring code under the Pt workflow.
