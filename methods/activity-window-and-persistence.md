# Activity sampling-window and harmonic persistence diagnostics

## Scope

`exonym activity` is a candidate-local exploratory diagnostic. It does not
measure a calibrated stellar-rotation posterior, activity false-alarm
probability, or transit false-positive probability. It preserves evidence that
a reviewer needs before considering a periodogram peak as stellar variability.

## Per-segment GLS and analytic probability

For each retained light-curve segment, the command fits a floating-mean
generalized Lomb-Scargle periodogram over the configured period interval. When
positive per-cadence flux errors are available, they are passed to GLS. The
reported analytic white-noise false-alarm probability is the backend's
single-segment, independent-noise extreme-value reference. It is not valid for
time-correlated stellar variability, evolving spots, detrending systematics,
or the total number of analysis choices.

## Sampling window

At every GLS trial frequency `f`, the command evaluates the normalized spectral
window from the retained timestamps `t_i`:

```text
W(f) = | (1 / N) * sum_i exp(-2 pi i f (t_i - mean(t))) |^2
```

The strongest separated window maxima are retained with a frequency resolution
of `1 / T`, where `T` is the segment baseline in days. The diagnostic also
reports whether the selected GLS frequency is within that resolution of a
retained window peak. This proximity is not an alias classifier: it only makes
the cadence/gap structure visible for review.

## Cross-segment persistence and harmonics

Every segment peak frequency, divided by the allowed fundamental, half-, and
double-frequency factors, proposes a reference family. The implementation
chooses the candidate that explains the largest number of segment peaks within
each segment's `1 / T` resolution; ties prefer the lower reference frequency.
Each segment is then reported as compatible or not with that descriptive
fundamental/first-harmonic family. This accommodates common spot-modulation
half/double ambiguities without choosing a physical rotation period
automatically.

The comparison does not model spot evolution, differential rotation, red
noise, duty-cycle selection, or independent seasons. It therefore remains
descriptive evidence rather than a detection test.

## Required interpretation

- Treat the reported modulation amplitude as a fixed-sinusoid description.
- Treat the reported analytic FAP as a white-noise reference only.
- Inspect the window peaks, segment consistency, transit-period harmonics, and
  phase-folded data before any activity interpretation.
- Do not route an activity result to a planet validation or rejection claim
  without independent, candidate-local evidence.

## Retained sources and provenance gap

- McQuillan, Mazeh & Aigrain (2014), *ApJS* 211, 24, ADS
  `2014ApJS..211...24M`, DOI `10.1088/0067-0049/211/2/24`.
- Vanderburg et al. (2016), *ApJS* 222, 14, ADS `2016ApJS..222...14V`, DOI
  `10.3847/0067-0049/222/1/14`.

The production GLS/FAP calculation is delegated to Astropy. Its exact primary
GLS/FAP source is not yet retained in `literature/README.md`; consequently the
reported FAP is explicitly a backend white-noise reference, not a calibrated
activity or validation probability.
