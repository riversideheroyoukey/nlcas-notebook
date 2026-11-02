# Practice-loop sample scenario: change of variables in a definite integral (S-o)

A detailed write-up of practice-loop scenario-menu item **S-o (analysis / volume
integrals)**. A range-bearing definite integral is given as a display-only,
unevaluated object (a region predicate or a LaTeX $\int\int$); a change of variables
builds and shows a new range-bearing integral (integrand·$|J|$ + new bounds), which is
then evaluated to a number. Three cases: the polar unit disk $\to\pi/2$, the spherical
unit-ball volume $\to 4\pi/3$, and a general linear Jacobian (user-supplied new
bounds) $\to 25/2$. Both the transform and the evaluation are rule-based (no LLM).

**Backend:** standard (Maxima) / **Domain:** electromagnetism (electromagnetism).

## Steps and expected results

| # | Input | Expected result |
|---|---|---|
| 1 | `Definite-integrate x^2+y^2 over the region x^2+y^2<=1` | ∬ over the unit disk (display only) |
| 2 | `Change variables in E1 from x,y to r,theta via x=r*cos(theta), y=r*sin(theta)` | ∫∫ $r^3$ + polar bounds |
| 3 | `Evaluate this definite integral` | $\pi/2$ |
| 4 | `Definite-integrate 1 over the region x^2+y^2+z^2<=1` | ∭ over the unit ball |
| 5 | `Change variables in E4 from x,y,z to r,theta,phi via …` | spherical Jacobian |
| 6 | `Evaluate this definite integral` | $4\pi/3$ |
| 7 | `$\int_{0}^{1}\int_{0}^{1}(x+y)\,dx\,dy$` | LaTeX ∫∫ (display only) |
| 8 | `Change variables in E7 from x,y to u:0..1, v:0..1 via x=2*u+v, y=u+3*v` | $\det J = 5$ |
| 9 | `Evaluate this definite integral` | $25/2$ |

The bounds are auto-derived for standard coordinate changes (polar×disk,
spherical×ball) and user-supplied for the general linear Jacobian. Change-of-variables,
the definite-integral object and the evaluation are all general operations.
→ [scenario_coordinate_jacobian_jp.md](scenario_coordinate_jacobian_jp.md)

**Worked example:** [`domains/electromagnetism/examples/`](../../domains/electromagnetism/examples/)
(`scenario_coordinate_jacobian_{jp,en}.{json,tex}`).

## generator command block (the single source the generator reads)

```scenario domain=electromagnetism
@note: Register a range-bearing definite integral of $x^2+y^2$ over the unit disk $x^2+y^2\le1$. At this point it is not yet handed to the CAS — it is kept as a display-only, unevaluated object.
Definite-integrate x^2+y^2 over the region x^2+y^2<=1
@note: Change this integral from Cartesian coordinates $(x,y)$ to polar coordinates $(r,\theta)$ ($x=r\cos\theta,\ y=r\sin\theta$). The Jacobian determinant $|J|=r$ is multiplied into the integrand, and the bounds are automatically rewritten in terms of the new variables.
Change variables in E1 from x,y to r,theta via x=r*cos(theta), y=r*sin(theta)
@note: Evaluate the transformed integral to obtain a number.
Evaluate this definite integral
@note: Now register a range-bearing definite integral of $1$ (the volume itself) over the unit ball $x^2+y^2+z^2\le1$, again as a display-only object.
Definite-integrate 1 over the region x^2+y^2+z^2<=1
@note: Change from Cartesian to spherical coordinates $(r,\theta,\phi)$ ($x=r\sin\theta\cos\phi,\ y=r\sin\theta\sin\phi,\ z=r\cos\theta$). The Jacobian determinant $|J|=r^2\sin\theta$ appears in the volume element.
Change variables in E4 from x,y,z to r,theta,phi via x=r*sin(theta)*cos(phi), y=r*sin(theta)*sin(phi), z=r*cos(theta)
@note: Evaluate the transformed integral to obtain the volume of the unit ball.
Evaluate this definite integral
@note: As an example where the coordinate-change formula is not a standard one, enter $\int_0^1\int_0^1(x+y)\,dx\,dy$ directly in LaTeX and register it as a display-only object.
$\int_{0}^{1} \int_{0}^{1} (x+y) \, dx \, dy$
@note: Change variables via the linear map $x=2u+v,\ y=u+3v$ from $(x,y)$ to $(u,v)$ (Jacobian determinant $\det J=5$). Since this is not a standard coordinate change, the new integration bounds $u:0..1,\ v:0..1$ must be given explicitly by the user here.
Change variables in E7 from x,y to u:0..1, v:0..1 via x=2*u+v, y=u+3*v
@note: Evaluate the transformed integral to obtain a number.
Evaluate this definite integral
```
