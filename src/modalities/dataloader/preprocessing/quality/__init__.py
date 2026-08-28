"""Quality-based document selection and up/downsampling for pretraining blends.

The package turns per-document quality signals into a training mix:

1. ``registry``      declares where each dataset lives and how it joins to annotations.
2. ``sidecar``       streams the JSONL once and records per-document position, length,
                     estimated token count and native quality metrics.
3. ``propella_join`` attaches external propella annotations to those documents.
4. ``cube``          aggregates the result so any threshold combination can be costed
                     without touching the data again.
5. ``selection``     evaluates a YAML selection against the cube.
6. ``materialize``   writes a filtered ``.idx`` that ``pack_encoded_data`` consumes
                     unchanged, so only selected documents are ever tokenized.
"""
