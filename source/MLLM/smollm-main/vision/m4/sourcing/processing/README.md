# Data Processing Pipelines

Relate to issue [#12](https://github.com/huggingface/m4/issues/12).

We have two v0 data processing pipelines:
- (a) split (for sharding) + parallel/slurm arrays of whatever processing scripts (python or rust for instance)
- (b) apache beam (for creating processing pipelines) + Dataflow (for horizontal scaling)

## App

ngram search is mostly an example.
to launch the app:
```bash
streamlit run app.py --server.port 6006
```
