"""Serving the fine-tuned Spanish<->Kaqchikel model via SageMaker Serverless
Inference (issue #8, ADR 0001). See `deployment/inference.py` for the custom
inference handler and `deployment/package_model.py` / `deployment/deploy.py`
for how a training run's artifact is turned into a registered, deployable
Model Package.
"""
