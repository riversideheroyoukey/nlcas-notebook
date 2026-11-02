"""The shared declarative plotting vocabulary given to the spec adviser."""
CARTESIAN_PROMPT = """
For plots combining curves, inequalities, contours or annotated points, use
intent.kind="cartesian_plot", method.name="direct_eval", and
presentation.kind="cartesian". This is a deterministic plotting facility;
do NOT ask the coding model to implement it. The presentation.options schema is:
{"x":"axis variable", "y":"second axis variable or omit for 1D lines",
 "layers":[...], "title":"", "xlabel":"", "ylabel":"",
 "formats":["png","svg","pdf"], "dpi":200, "legend":true}.
Each axis MUST have a sweep input {name,role:"sweep",spec:{min,max,points}},
with finite min<max and integer points>=2 (typically 200). Non-axis parameters
are fixed inputs. Axis variables may be absent from the selected expression.
Use outputs:[{name:"plot",shape:"field",over:[axis names]}].
Layers (use only registered expr_ids from Available expressions):
- {kind:"line",expr_id:"E...",label:"..."}: explicit scalar y=f(x); only x varies.
- {kind:"contour",expr_id:"E...",levels:[0],label:"..."}: scalar level sets.
  A registered equality means lhs-rhs, so level 0 draws its boundary.
- {kind:"region",constraints:["E...",...],label:"...",alpha:0.25}:
  intersection of registered <,<=,>,>= inequalities; may be nonlinear.
  Different region layers compare different scenarios; never merge their constraints.
- {kind:"point",x:{expr_id:"E..."},y:{expr_id:"E..."},label:"..."}:
  coordinates reference computed scalar results, or explicit user-provided numbers.
  Points accept marker ("o","s","*","x","+","^","D",".") and markersize (0..40).
A contour level may also be {expr_id:"E..."} for a computed scalar result.
All layers accept optional color, linestyle ("-","--",":","-."), and
fixed_values:{parameter:number} for comparing parameter settings of the same formula.
Never invent an optimum or a parameter value. Missing coordinate results or
constraints must be registered/computed first; explain the missing information
in the intent goal rather than fabricating reference IDs. Plotting performs no
optimization. Shaded regions are samples on the requested bounded grid, not
an exact geometric proof; bounds and resolution must be explicit in the spec.
"""
