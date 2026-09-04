# Scientific method, unit, and claim contract

## Authority and scope

This is the target-neutral documentation index for every module under
`src/exonym/`. It is the source-level companion to the command-level records in
`methods/` and the retained-source registry in `literature/README.md`. It does
not contain candidate identifiers, coordinates, sectors, ephemerides, measured
values, or candidate conclusions.

The index makes an essential distinction required by `AGENTS.md` Rule 6:

1. a **formula-bearing scientific module** evaluates a physical or statistical
   relation and must name a retained primary source, units, applicability, and
   failure boundary;
2. an **official-library/standards adapter** delegates a standard operation to
   Astropy, SciPy, Batman, LDTk, Celerite, or another declared engine and must
   preserve that engine's units and fail closed when its contract is absent; and
3. an **operational module** handles ownership, hashing, serialization,
   lifecycle, execution, rendering, or review. It implements no astrophysical
   equation. Assigning a paper citation to such code would be false provenance.

The retained PDFs named below are the auditable source of every ADS bibcode and
DOI in this document. A named provenance gap is not permission to invent a
citation or to promote an output to science; it is work that must be completed
before the affected relation is extended or interpreted beyond its current
diagnostic boundary.

## Non-negotiable data and unit boundary

| Quantity | Required representation at EXONYM scientific boundaries | Rejection / interpretation rule |
| --- | --- | --- |
| Photometric time | `BTJD_TDB` days for TESS light curves: `BTJD = BJD_TDB - 2457000`; RV observations use `BJD_TDB` days. | UTC, MJD, BJD without declared scale/reference, or mixed time systems are rejected rather than shifted implicitly. |
| Flux | Normalized relative flux, dimensionless. | It is not ppm. Convert a relative deficit to ppm only by multiplying by `PARTS_PER_MILLION`. |
| Photometric uncertainty | Positive, finite normalized relative-flux error per cadence. | Celerite and likelihood paths fail without reported errors; a scatter/MAD surrogate is not created as observational covariance. |
| Transit ephemeris | `period_days`, `epoch_btjd`, `duration_days`, and `depth_ppm`; all values are candidate-owned and hash-bound where consumed. | No synthetic, static, or silently converted ephemeris is accepted as evidence. |
| Stellar parameters | `teff_k` in K, `logg_cgs` in `log10(cm s^-2)`, `[Fe/H]` in dex, `mass_solar` in `M_sun`, and `radius_solar` in `R_sun`. | Missing, nonfinite, incompatible, or unprovenanced parameters block the consumer that requires them. |
| Geometric quantities | Sky positions in ICRS degrees; offsets/separations in arcsec or mas as named; `a_rs`, `rp_rs`, flux ratios, eccentricity, and probabilities are dimensionless. | A unitless value may not be treated as an angular, physical-radius, or probability value without its field contract. |
| SI and CGS quantities | SI values use m, kg, s, J K^-1; CGS values use cm, g, s. | Conversions use `exonym.constants` or Astropy standards, never a locally rounded physical constant. |

Candidate-facing scientific modules consume candidate-owned input below
`candidate/<id>/`; standards, operational, and presentation modules may instead
consume declared scalar values, paths, or no candidate input. Shared source and
documentation remain target-neutral. Hashes establish byte lineage, not physical
correctness.

## Claim invariant

Every current vetting decision, TREX result, automation record, paper-export
manifest, and analysis-status record must keep `claim_eligible: false`. This is
not a threshold convention and a low FPP/NFPP cannot override it. The current
implementation lacks the calibrated, provenance-bound scene-model integration
required to connect observed photometry, calibrated PRF/difference-image source
constraints, complete resolved-neighbor constraints, high-resolution imaging
limits where applicable, stellar characterization, and the scenario model on a
claim-bearing path. Consequently:

- BLS/TLS, survey alerts, injection recovery, and screening are ranking or
  robustness diagnostics, not calibrated discovery reliability;
- localization and dilution are exploratory source-competition diagnostics, not
  a calibrated source assignment;
- transit, SED, asteroseismic, TTV, phase-curve, RV, and specialized-adapter
  outputs are descriptive evidence with their stated model assumptions;
- TRICERATOPS/TREX FPP and NFPP are conditional scenario probabilities, not
  universal planet-validation probabilities; and
- workflow completion, engine availability, a valid schema, a provenance hash,
  checkpoint restore, or an exported manuscript macro cannot create a claim.

The analysis gate therefore remains intentionally blocked. A missing optional
dependency, invalid input, nonfinite calculation, unsupported domain, or failed
calibration is recorded as unavailable/failed/unresolved, never converted into
a rejection, validation, or substitute result.

## Verified primary-source register used by formula-bearing modules

| Key | Retained source and use | NASA ADS bibcode | DOI |
| --- | --- | --- | --- |
| IAU nominal constants | IAU 2015 B3 nominal solar/planetary values used by `constants.py` and `vetting/trex/constants.py`. | `2016AJ....152...41P` | `10.3847/0004-6256/152/2/41` |
| CODATA constants | CODATA 2018 `G`, `k_B`, proton mass, and reference conversions. | `2021RvMP...93b5010T` | `10.1103/RevModPhys.93.025010` |
| Mandel--Agol transit | Quadratic limb-darkened analytic transit profile. | `2002ApJ...580L.171M` | `10.1086/345520` |
| Batman implementation | Exposure-integrated transit forward model adapter. | `2015PASP..127.1161K` | `10.1086/683602` |
| Kipping limb darkening | Quadratic `(u1,u2)` / triangular `(q1,q2)` transform. | `2013MNRAS.435.2152K` | `10.1093/mnras/stt1435` |
| Claret TESS grids | TESS limb/gravity-darkening atmosphere-grid context. | `2017A&A...600A..30C` | `10.1051/0004-6361/201629705` |
| Transit density relation | Circular transit geometry and density constraints. | `2003ApJ...585.1038S`; `2007ApJ...664.1190S` | `10.1086/346105`; `10.1086/519214` |
| Eccentric coordinates | Regular eccentric-orbit parameterization and Jacobian. | `2013PASP..125...83E` | `10.1086/669497` |
| BLS | Box-fitting periodic-transit search. | `2002A&A...391..369K` | `10.1051/0004-6361:20020802` |
| TLS | Limb-darkened transit-search ranking. | `2019A&A...623A..39H` | `10.1051/0004-6361/201834672` |
| Wotan | Time-series detrending backend. | `2019AJ....158..143H` | `10.3847/1538-3881/ab3984` |
| Celerite | Scalable GP covariance implementation. | `2017AJ....154..220F` | `10.3847/1538-3881/aa9332` |
| Asteroseismic scaling | Solar-like `nu_max`, `Delta_nu`, mass, radius, density, and gravity scaling context. | `2011ApJ...743..143H`; `2014ApJS..210....1C` | `10.1088/0004-637X/743/2/143`; `10.1088/0067-0049/210/1/1` |
| Kepler equation / TTV | Kepler solver and first-order near-resonant super-period diagnostic. | `1983CeMec..31...95D`; `2012ApJ...761..122L` | `10.1007/BF01686811`; `10.1088/0004-637X/761/2/122` |
| RV modelling | Fixed-period eccentric Keplerian comparison implementation context. | `2013PASP..125...83E`; `2018PASP..130d4504F` | `10.1086/669497`; `10.1088/1538-3873/aaaaa8` |
| BEER / ellipsoidal | Circular phase-harmonic and leading tidal ellipsoidal diagnostics. | `2011MNRAS.415.3921F`; `1985ApJ...295..143M`; `2017PASP..129g2001S` | `10.1111/j.1365-2966.2011.19011.x`; `10.1086/163359`; `10.1088/1538-3873/aa7112` |
| MIST / extinction | MIST isochrone/bolometric-correction tables and Fitzpatrick extinction context. | `2016ApJS..222....8D`; `2016ApJ...823..102C`; `1999PASP..111...63F` | `10.3847/0067-0049/222/1/8`; `10.3847/0004-637X/823/2/102`; `10.1086/316293` |
| Stellar relations | TESS catalog context, mass/radius relations, and magnitude calibration. | `2018AJ....156..102S`; `2019AJ....158..138S`; `2010A&ARv..18...67T`; `2018A&A...616A...4E` | `10.3847/1538-3881/aad050`; `10.3847/1538-3881/ab3467`; `10.1007/s00159-009-0027-7`; `10.1051/0004-6361/201832756` |
| TRILEGAL / TREX | Galactic population synthesis and conditional false-positive scenarios. | `2005A&A...436..895G`; `2021AJ....161...24G` | `10.1051/0004-6361:20042352`; `10.3847/1538-3881/abd184` |
| Vetting population priors | Planet, binary, multiplicity, and M-dwarf companion priors. | `2013ApJ...766...81F`; `2017ApJS..230...15M`; `2010ApJS..190....1R`; `2019AJ....157..216W` | `10.1088/0004-637X/766/2/81`; `10.3847/1538-4365/aa6fb6`; `10.1088/0067-0049/190/1/1`; `10.3847/1538-3881/ab05dc` |
| Localization context | TESS imaging scale and difference-image/centroid false-positive diagnostics. | `2015JATIS...1a4003R`; `2013PASP..125..889B`; `2018PASP..130f4502T` | `10.1117/1.JATIS.1.1.014003`; `10.1086/671767`; `10.1088/1538-3873/aab694` |
| Evidence comparison | Interpretation context for Bayes factors. | `1995JASA...90..773K` | `10.1080/01621459.1995.10476572` |

## Formula-bearing scientific API contract

The function lists are public or extenally meaningful entry points; private
helpers inherit the same units and limits from their row. Each routine raises
`ValueError`/`RuntimeError` or retuns an explicitly unavailable diagnostic when
its stated domain is not met. No row authorizes a claim.

| Modules and scientific functions | Inputs and retuned units | Domain, assumptions, and hard boundary | Source key(s) |
| --- | --- | --- | --- |
| `constants`: all exported constants; `vetting/trex/constants`: `Msun`, `Rsun`, `Rearth`, `G`, `au` | Standard SI/CGS values exactly named: m, km, cm, kg, g, s, J K^-1, `M_sun`, `R_sun`, or dimensionless conversions. | Standards adapters, not fitted measurements. Astropy is authoritative; no candidate values enter. | IAU nominal constants; CODATA constants |
| `lightcurve`: `phase_hours`, `robust_transit_depth`, `bin_phase_folded_flux`, `calculate_contact_durations`, `kipping_to_quadratic_limb_darkening`, `quadratic_to_kipping_limb_darkening` | BTJD days -> signed phase hours; normalized flux -> depth/error ppm; contact durations hours; radius ratios and limb-darkening coordinates dimensionless. | Linear ephemeris and declared, positive period; circular-contact equations do not replace eccentric modelling. Invalid times, geometry, or coefficients fail. | Transit density relation; Kipping limb darkening |
| `detrending`: `transit_mask_from_ephemeris`, `transit_mask_provenance_from_ephemeris`, `validate_transit_mask_provenance`, `detrend_candidate` | BTJD days, normalized relative flux and positive errors; outputs candidate-local `time_btjd`, normalized detrended flux/error, and sector labels. | Running median is descriptive; Wotan/Celerite are opt-in. Celerite requires reported positive cadence errors. Protected masks require a complete candidate BTJD ephemeris and matching hash. | Wotan; Celerite |
| `search`: `find_transits`, `find_transits_duration_grid`, `find_transits_tls`, `run_bls_on_candidate`, `calculate_ttv_super_period`, `compute_linear_ephemeris_residuals` | Input BTJD days/normalized flux/errors; BLS/TLS output period days, epoch BTJD, duration hours, depth ppm, and dimensionless SNR/SDE; O-C is minutes; super-period days. | BLS/TLS values rank searched models only. TLS requires optional package. Exact first-order resonance can retun infinity. No calibrated false-alarm or completeness conclusion. | BLS; TLS; Kepler equation / TTV |
| `screening`: `fixed_ephemeris_screen`, `run_fixed_ephemeris_screen`; `discovery`: `detrend_by_sector`, `search_duration_grid`, `inject_box_transit`, `mask_box_transit`, `robustness_diagnostics`, `injection_recovery_diagnostics`; `survey`: `run_survey_search`, `run_survey_sensitivity` | BTJD days, normalized flux/error, period/duration days, depth ppm; recovery fractions and ranking statistics dimensionless. | Fixed configuration controls are candidate/survey diagnostics. Injection recovery is finite-trial sensitivity evidence, never population completeness or validation. Missing real photometry fails closed. | BLS; TLS; Wotan (where selected) |
| `activity`: `gls_periodogram`, `sampling_window_periodogram`, `sampling_window_diagnostics`, `segment_harmonic_persistence`, `sinusoid_amplitude_ppm`, `run_stellar_activity` | BTJD days; normalized flux/error; periods days, frequencies day^-1, amplitudes ppm, white-noise FAP dimensionless. | Floating-mean GLS and window/harmonic agreement do not model evolving spots, red noise, or a rotation posterior. Insufficient cadence/coverage fails. | **Gap:** Astropy GLS is delegated standard functionality; the retained registry currently lacks the primary GLS/FAP paper. Preserve diagnostic-only status until it is retained and registered. |
| `asteroseismology`: `frequency_support`, `compute_power_spectrum`, `fit_harvey_granulation_background`, `spacing_correlation`, `estimate_oscillation_envelope`, `seismic_mass_radius`, `seismic_uncertainty_summary`, `seismic_sanity_check`, `run_asteroseismology` | Day-based candidate times; frequency `uHz`; PSD in Astropy `psd` normalization; whitened power dimensionless; `T_eff` K; mass/radius solar; density solar or g cm^-3; `logg` cgs. | Solar-like oscillation scaling is exploratory across solar-like stars through giants with known systematics; not mode identification. Cadence-supported frequency range, finite measurements, and applicability checks are required. The Harvey-style background implementation remains a provenance gap pending a retained primary source. | Asteroseismic scaling |
| `limb_darkening`: `generate_ldtk_quadratic_prior`; `transit_fit`: `stellar_density_a_rs`, `conjunction_distance_a_rs`, `inclination_deg_from_impact_parameter`, `batman_transit_flux`, `fit_transit_light_curve`, `run_mcmc_transit_fit`, `compute_bayesian_model_comparison` | Stellar inputs K, `log10(cm s^-2)`, dex; `u1,u2,q1,q2,rp_rs,a_rs,e` dimensionless; inclination/omega degrees; period days; density solar/CGS as named; output flux normalized, depth ppm. | Quadratic limb darkening and opaque spherical occulter; circular model unless eccentric coordinates are selected. Posterior draws outside physical geometry reject rather than clip. Missing engine, provenance, cadence error, or stellar uncertainty fails. Evidence/logZ is descriptive. | Mandel--Agol transit; Batman implementation; Kipping limb darkening; Claret TESS grids; Transit density relation; Eccentric coordinates |
| `sed`: `run_sed_fit`; `analysis_status`: `_mist_bc_sed_stage` | Candidate MIST table axes `T_eff` K, `logg` cgs, `[Fe/H]` dex, `A_V` mag; observed/model Vega magnitudes mag; profiled apparent bolometric magnitude mag. | Only hash-bound MIST v1.2 BC tables in their native interpolation domain. No blackbody, nearest-node, generic-atmosphere, distance, radius, luminosity, or posterior fallback. Missing/mismatched table or manifest fails. | MIST / extinction |
| `phasecurve`: `_circular_phase_summary`, `resolve_secondary_eclipse_control`, `cluster_sandwich_covariance`, `build_design_matrix`, `fit_phase_curve_components`, `run_phase_curve_search`; `vetting/ellipsoidal`: `gravity_darkening_exponent`, `morris_ellipsoidal_coefficient`, `ellipsoidal_variation_amplitude_ppm`, `ellipsoidal_gate` | BTJD days, normalized flux/errors, phase dimensionless, component amplitudes/errors ppm, covariance ppm^2 after scaling; mass/radius solar, semimajor axis AU, temperature K, inclination degrees. | Circular BEER basis and secondary box are exploratory. Eccentric control requires a matching candidate posterior; no brightness map, calibrated red-noise FAP, or source attribution. Morris leading order is a binary screen, not a mass measurement. | BEER / ellipsoidal |
| `radial_velocity`: `ingest_radial_velocity_observations`, `load_radial_velocity_observations`, `keplerian_velocity_m_per_s`, `fit_radial_velocity` | Observation/reference times BJD_TDB days; velocity/error/jitter/offset K m s^-1; trend m s^-1 day^-1; angles radians intenally; eccentricity dimensionless; BIC/AIC dimensionless. | Fixed candidate period (or declared Gaussian period prior), independent Gaussian residuals plus per-instrument jitter, one eccentric Keplerian versus shared nuisance model. Sparse/malformed input or solver failure writes no fit. Not confirmation or mass inference. | RV modelling; Evidence comparison |
| `catalog`: `calculate_radial_velocity_semi_amplitude`, `calculate_astrometric_wobble_microarcsec`, `calculate_atmospheric_scale_height_km`, `calculate_transmission_signal_ppm` | Inputs explicitly named Earth/solar/AU/pc/K; retuns m s^-1, microarcsec, km, and ppm. | Planning scalings only; no observational likelihood, uncertainty propagation, or candidate evidence. **Gap:** the repository has no locally retained primary source for all four convenience scaling laws; they must not be extended or cited as calibrated characterisation. | Standards only; provenance gap recorded |
| `archive`: `run_archival_vetting`, `load_validated_archival_report`, `load_validated_archival_gaia_sources`; `catalog_federation`: retrieval and astrometric-normalization API; `priors`: ExoFOP record normalizers; `survey_harvest`: `TceFilters`, `NoveltyResult`, `evaluate_live_novelty`, `harvest_tces` | Provider-native fields retain declared units: ICRS degrees, arcsec/mas, mas yr^-1, Julian years, catalog magnitudes, BJD/BTJD per provider contract, period days, duration hours/days, and source ranking metrics. | These preserve/reconcile evidence but do not derive a physical posterior. No-match is not novelty; unavailable provider is not absence of evidence; incompatible time scales fail. | Provider metadata; Gaia photometry propagation uses Evans 2018; no formula is inferred from an archive response |
| `ephemeris_matching`: `lithwick_wu_repulsion_width_fractional`, `record_known_signal_ephemeris`, `match_known_signal_ephemerides` | Period/duration days unless the provider says hours; epoch retains BTJD/BJD scale; masses/radii Earth/solar as named; resonance width and harmonic ratios dimensionless. | Known-signal comparison is review-only; missing provider/time-scale/finite fields and ambiguous alias parity remain unresolved. The Lithwick--Wu scale is a screening width, not a dynamical fit. | Kepler equation / TTV |
| `localization`: `gaussian_prf_kenel`, `calibrated_prf_assets`, `load_calibrated_prf_template`, `calibrated_prf_kenel`, `fit_calibrated_difference_image_prf`, `fit_difference_image_prf`, `fit_depth_map_prf`, `build_difference_image`, `build_depth_map`, `localize_difference_image`, `localize_depth_deficit`, `extract_tpf_difference_image`, `extract_tpf_depth_map`, `run_prf_localization`; `vetting/centroid`: `centroid_offset_z`, `centroid_gate`, `centroid_offset_pvalue` | TPF flux/count units remain source-native; pixel offsets pixels; sky offsets arcsec; coordinates degrees; localization ratio dimensionless; centroid z/p dimensionless. | Current Gaussian/NNLS/prior-source competition is an uncalibrated screen unless a validated calibrated PRF asset is present. Missing TPF/Gaia/ephemeris/asset fails; result cannot assign source or validate. | Localization context |
| `dilution`: `aperture_depth_ppm`, `gaia_g_to_tess_mag`, `gaia_contamination_factor`, `run_dilution_sensitivity` | BTJD days, aperture flux normalized intenally, depth/error ppm, magnitudes mag, contamination factor dimensionless. | Aperture and catalog-band diagnostic only; not a TESS-band scene dilution correction. **Gap:** the adopted Gaia-to-TESS transform needs a retained primary relation before it may be treated as calibrated. | Stellar relations; provenance gap recorded |
| `vetting/oddeven`: `odd_even_z`, `odd_even_gate`; `vetting/centroid`: `centroid_offset_z`, `centroid_gate`, `centroid_offset_pvalue` | Depth/error ppm or fractional flux consistently paired; offsets and sigma arcsec; z and p dimensionless. | Independent-error screens only. Correlated noise, blend geometry, and source assignment are not modeled; invalid errors fail rather than pass. | Localization context for centroid diagnostic; no separate retained primary source for the algebraic odd/even statistic |
| `vetting/trex`: `TargetScene`, `run_trex_vetting`, `stellar_relations`, `J_Ks_to_Tmag`, `companion_flux_ratio`, `dilute_flux`, `separation_at_contrast`, `delta_mag_to_flux_ratio`, `flux_ratio_to_delta_mag`, `tess_surface_brightness_ratio`, `semi_major_axis_cgs`, `a_over_Rs`, `impact_parameter`, `secondary_eclipse_phase`, `sample_rp`, `sample_inc`, `sample_ecc`, `sample_w`, `sample_q`, `lnprior_bound`, `lnprior_background`, `simulate_TP`, `simulate_EB`, `lnL_TP`, `lnL_EB`, `calc_target_evidences`, `compute_fpp_nfpp`, `TrexResult`, `generate_diagnostics`; `vetting/tricera_parse`: `load_fpp_report`, `extract_fpp`, `extract_nfpp`, `fpp_gate`, `run_triceratops_simulation` | Phase-relative time days, normalized flux, sigma normalized flux, period days, depth ppm, exposure days, masses/radii solar or Earth as named, semimajor axis cm, contrast arcsec/delta-mag, parallax mas, FPP/NFPP dimensionless. | Conditional Monte Carlo scenario calculation. Requires hash-bound observed photometry, finite dynamic exposure, complete scene/contrast/TRILEGAL inputs. Numeric anomaly/unavailable scene/runtime is unresolved. `claim_eligible` is always false. | TRILEGAL / TREX; Vetting population priors; Mandel--Agol transit; Batman implementation; Stellar relations |
| `specialized_models`: `run_planetsynth`, `run_pyppluss`, `run_catwoman`, `run_squishyplanet` | PlanetSynth: `M_jup`, `R_jup`, Gyr, K and output `R_jup`, `L_sun`; other adapters: time days, normalized flux/residuals dimensionless. | Declared hypothesis/package adapter only. Unsupported interface, missing package, nonfinite output, or out-of-domain input yields a status manifest and no scientific result. | **Gap:** package-specific primary sources are not yet retained/verified; the adapter preserves package output but does not independently re-derive it. |

## Full source and class inventory

The following table covers every Python module currently shipped below
`src/exonym/`. “Operational” means that there is no astrophysical formula to
cite; its units are filesystem paths, bytes, timestamps, hashes, booleans,
JSON, CLI text, or explicitly labelled values passed through from an artifact.

| Module family | Modules and public classes | Contract |
| --- | --- | --- |
| Package and CLI | `__init__`, `__main__`, `banner`, `wizard` | Operational entry surfaces. They dispatch commands and present text; they do not calculate astrophysical quantities. CLI numeric values retain the target command's named unit and are validated by that command. No claim state can be changed by presentation or argument construction. |
| Workspace, policy, and review | `workspace` (`CandidateWorkspace`), `gatekeeper` (`GateError`), `tracking` (`ChecklistItem`, `DocumentTelemetry`), `review`, `classification`, `tagging`, `analysis_status`, `paper_export` | Operational records, paths, strings, hashes, lifecycle/state enumerations, and checklist counts. `analysis_status`/paper export may describe an artifact’s named units but cannot recompute it or turn it into evidence. The analysis gate and paper exporter preserve `claim_eligible: false`. |
| Isolation, schemas, and resources | `isolation` (`Violation`, `IsolationReport`), `schemas`, `resources` (`ResourceUnavailableError`), `remediation`, `verification_cache` (`CandidateVerificationCache`) | Operational integrity and schema logic. Hashes, byte counts, paths, JSON, and cache fingerprints have no physical units and are not scientific provenance by themselves. Invalid schema/resource/path fails closed. |
| Acquisition and storage | `download` (`DownloadItem`, `DownloadResult`, `DownloadAccessError`, `DownloadError`, `DownloadEngine`), `ingest` (`FetchedProducts`), `storage`, `freeze` (`ReleaseVerificationError`), `checkpoints` | Operational byte acquisition, SHA-256, HTTP status/timeouts in seconds, file byte counts, and archive manifests. No method evaluates photometry or establishes data quality. Freeze/checkpoint restoration is integrity/recovery, not scientific reproduction. |
| Engines and automation | `engines` (`EngineDescriptor`, `EngineStatus`), `autonomous`, `debugger` (`Finding`, `ToolResult`, `DebugReport`), `telemetry` (`LiveTelemetry`) | Operational runtime availability, status, UTC timestamps, telemetry counts/rates, and diagnostic logs. A completed run is execution provenance, not convergence or a claim. |
| Candidate/catalog evidence | `catalog` (`IdentifierError`), `catalog_federation` (`ProviderSpec`, `CatalogRequest`, `TransportResponse`), `archive` (`ArchivalVettingService`), `inputs`, `priors`, `ephemeris_matching`, `survey_harvest` (`TceFilters`, `NoveltyResult`), `ds9` | Provider-native science values are passed with the units in the source contract. DS9 exports validated FK5 coordinates in degrees and is rendering-only. These modules enforce finite values, unit/time-scale declarations, candidate ownership, and hash lineage; a retrieval, no-match, or region file is not a physical inference, novelty proof, or claim. |
| Photometry and inference | `lightcurve`, `detrending` (`OptionalBackendUnavailable`, `DetrendingArtifacts`), `search` (`BLSSearchResult`), `discovery`, `survey` (`SurveyWorkspace`), `screening`, `activity`, `asteroseismology`, `limb_darkening`, `sed`, `transit_fit` (`_GpuBackendUnavailable`, `_AcceleratedTransitFitData`), `ttv`, `phasecurve`, `plotting`, `radial_velocity`, `specialized_models` (`AdapterRun`) | Formula-bearing modules described in the preceding API table. `plotting` is presentation-only and cannot add statistical weight. The two transit-fit helper classes hold backend/data state only; they do not add a separate physical relation. Every artifact must preserve its time/flux/unit contract and stated applicability limit. |
| Vetting and source competition | `statistical_vetting`, `dilution`, `localization`, `vetting` package, `vetting/centroid`, `vetting/oddeven`, `vetting/ellipsoidal`, `vetting/tricera_parse` (`TrexSceneUnavailableError`) | Diagnostic routing and the physical helper functions listed above. They retain named units and fail boundaries but do not calibrate a claim-bearing source scene. |
| TREX internals | `vetting/trex/__init__`, `constants`, `target` (`TargetScene`), `funcs`, `priors`, `likelihoods`, `marginal_likelihoods`, `diagnostics` (`TrexDiagnostic`, `TrexResult`), `_numerics`, `network` (`_LinkParser`), `licensing` | Conditional statistical-vetting implementation. `constants` uses CGS; target scene fields carry degrees, `M_sun`, `R_sun`, K, mag, mas, arcsec/delta-mag, and retained population counts; `_numerics`, network, and licensing are operational support. `_LinkParser` parses HTML links only and has no physical unit. All TREX results remain claim-ineligible. |
| Packaged resources | `_resources/__init__` and files below `_resources/` | Wheel-resource fallback only. No scientific calculation or candidate payload is present. |

## Documentation maintenance requirements

When changing a formula-bearing routine:

1. update the relevant source docstring with its named units, domain, and
   failure behavior;
2. update the row above and the linked `methods/` record if its equation,
   model, prior, or interpretation changes;
3. add the exact primary source’s ADS bibcode and DOI to
   `literature/README.md` only after the source has been locally verified and
   retained; and
4. add a target-neutral test that reaches the production caller, including an
   unsupported/nonfinite/provenance-failure path.

Do not use a passing layout linter, a hash, an engine status, or this document
as evidence that a scientific calculation is calibrated. Focused tests and a
candidate-local evidence chain remain required.