# Deployment
Local: README quickstart. Docker: `docker build -t mmdq . && docker run --rm mmdq` (tests). ~80 s per seed on CPU (SBERT embedding dominates).
Not production-ready: synthetic data, no persistence of flagged-sample review queue, no human relabeling workflow.
