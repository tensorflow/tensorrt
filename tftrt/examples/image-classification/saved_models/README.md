# Introduction
A collection of sample models to run with TF/TRT and capture potential
performance and numerical regressions.

# Usage
Use `tensorflow/python/compiler/tensorrt/model_tests/run_models` as the driver
of the model tests.

Options:
```bash
$(TENSORFLOW_REPO_PATH)/tensorflow/python/compiler/tensorrt/model_tests/run_models --help
  --batch_size: The batch size used to run the testing model with.
    (default: '128')
    (an integer)
  --diff_tolerance: Log errors whenever mean TensorRT relative difference is larger than the tolerance.
    (default: '0.05')
    (a number)
  --latency_baseline: <CPU|GPU>: The baseline version for latency improvement analysis.
    (default: 'GPU')
  --numerics_baseline: <CPU|GPU>: The baseline version for numerical difference analysis.
    (default: 'CPU')
  --output_dir: Output directory of analysis results.
  --output_format: <CSV|JSON>: Output format of analysis results.
    (default: 'CSV')
  --saved_model_dir: The directory to the testing SavedModel.
    (default: '$(TENSORFLOW_REPO_PATH)/tensorflow/python/compiler/tensorrt/model_tests/sample_model')
  --saved_model_signature_key: The signature key of the testing SavedModel being used.
    (default: 'serving_default')
  --saved_model_tags: The tags of the testing SavedModel being used.;
    repeat this option to specify a list of values
    (default: "['serve']")
  --speedup_tolerance: Log errors whenever mean TensorRT speedup is lower than the tolerance.
    (default: '0.95')
    (a number)
  --[no]use_tf2: Whether to test with TF2 behavior or not (TF1).
    (default: 'true')
```

Example:
```bash
$(TENSORFLOW_REPO_PATH)/tensorflow/python/compiler/tensorrt/model_tests/run_models \
--saved_model_dir=$(TENSORFLOW_TENSORRT_REPO_PATH)/tftrt/tests/resnet50v2 \
--saved_model_signature_key=resnet50v2 --batch_size=1  --use_tf2=false --numerics_baseline=GPU
```

# Models
| Architecture | SavedModel | Signature  | Format       |
|--------------|------------|------------|--------------|
| ResNet50V2   | reset50v2/ | resnet50v2 | SavedModelV1 |
