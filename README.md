# RIPHAMMER

Multi-band NMF for cell-type-specific TAD extraction from bulk Hi-C data.

## Installation

```bash
pip install riphammer
```

## Quick start

```bash
# Vectorize mcool files
riphammer vectorize sample1.mcool::resolutions/100000 sample2.mcool::resolutions/100000 -o data/

# Run Multi-band NMF
riphammer fit -i data/ -o results/ --k 8

# Detect TADs from NMF components
riphammer tad -i results/ -o tad/
```

## Python API

```python
import riphammer

# Load data
X, features, distances, samples = riphammer.mcool_to_features(cool_uris)

# Fit Multi-band NMF
model = riphammer.MultibandNMF(n_components=8, max_iter=500)
model.fit(X, distances)

# TAD calling
riphammer.w_to_cool(model.get_W_full()[:, 0], features, "comp1.cool")
insulation, boundaries = riphammer.call_tads("comp1.cool")
```

## License

MIT
