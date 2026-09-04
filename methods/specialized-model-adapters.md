# Specialized-model adapters

## Scope

`exonym planetsynth` accepts a declared candidate-local giant-planet characterization and invokes a narrowly defined optional package interface. `exonym pyppluss` evaluates one declared ringed or oblate anomalous-transit hypothesis against candidate-local normalized flux. `exonym catwoman` and `exonym squishyplanet` expose the same candidate-local adapter boundary for terminator-asymmetry hypotheses. All adapters preserve raw package output when available, runtime metadata, input hashes, and a normalized run manifest. Unsupported or unverified interfaces fail closed without synthetic model output. None writes a claim, validates a planet, or changes lifecycle state.

## Equations and units

The planetsynth adapter does not reimplement a cooling or evolution equation. It passes mass in `M_jup`, radius in `R_jup`, age in Gyr, and equilibrium temperature in K to the installed package, then reports its finite radius in `R_jup` and luminosity in `L_sun`. Its declared applicability guards are:

```
0.1 <= M/M_jup <= 20
0.5 <= R/R_jup <= 2.5
0.001 <= age/Gyr <= 20
0 <= T_eq/K <= 3000
```

The pyPplusS adapter passes time in days and dimensionless normalized flux to the package. Given measured flux `f_i` and returned model flux `m_i`, it records:

```
r_i = f_i - m_i
RMS = sqrt(sum(r_i^2) / N)
max_abs_residual = max(|r_i|)
```

Both diagnostics are dimensionless relative flux. They are goodness-of-fit summaries for one declared model, not model-selection evidence or a physical identification.

## Assumptions and failure modes

- planetsynth results depend on the installed model grid, boundary conditions, composition assumptions, irradiation treatment, and correctness of the supplied characterization. The adapter does not extrapolate outside its declared applicability range.
- pyPplusS compares one specified geometry. Stellar spots, instrumental trends, cadence integration, dilution, ordinary transit degeneracies, and alternative anomaly models can produce similar residuals.
- Candidate time samples must be strictly increasing. pyPplusS normalized flux must remain between 0.5 and 1.5, and ring radii must satisfy planet radius < inner radius < outer radius.
- Missing package metadata, unsupported package interface, import failure, runtime failure, non-finite results, or a wrong-length model output produces only an unavailable or failed manifest and no normalized scientific output.
- Catwoman and SquishyPlanet are optional adapters, not bundled Exonym engines. Their availability must be checked from the installed package contract; an unavailable or unverified contract is a candidate-local status record, not evidence for or against terminator asymmetry.

## Literature provenance gap

The package-specific papers commonly associated with giant-planet cooling,
oblateness, and exoring models are not retained and verified in
`literature/README.md`. The adapters therefore do not cite them as an EXONYM
method implementation. They only dispatch a declared optional-package interface
and preserve package/runtime provenance. A package source may be added here only
after its primary source is locally retained with an exact ADS bibcode and DOI.
