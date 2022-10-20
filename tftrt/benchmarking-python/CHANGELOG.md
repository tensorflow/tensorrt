# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--

============== Guiding Principles ==============

* Changelogs are for humans, not machines.
* There should be an entry for every single version.
* The same types of changes should be grouped.
* Versions and sections should be linkable.
* The latest version comes first.
* The release date of each version is displayed.
* Mention whether you follow Semantic Versioning.

--------------------------- TEMPLATE -------------------------------------

## [MAJOR.MINOR.PATCH] - YYYY.MM.DD - @UserDoingTheUpdate

Description of the change

-->

# Documentation

##  How To Update The Changelog for a New Release ?

 - **Major Version:** Major refactoring, or changes that make the data
                   non-comparable with any previous release of the benchmark.
                   Changes in shell scripts are to be expected.

 - **Minor Version:** Changes that add a new functionality though not modifying
                   existing metrics or models scripts in a way that would make
                   metrics not comparable between minor releases.

 - **Patch Version:** Changes that are expected to have no change to the
                   operation of the benchmark nor the way metrics are
                   calculated. Basically these changes are transparent for the
                   user.

# Versions

<!-- YOU CAN EDIT FROM HERE -->

## [2.0.3] - 2022.10.20 - @DEKHTIARJonathan

- Profiling for TF-TRT build and convert added
- Argument `tf_profile_export_path` renamed: `tftrt_build_profile_export_path`
- NVIDIA PyCOCOTools Updated to version 0.7.1

## [2.0.2] - 2022.09.30 - @DEKHTIARJonathan

- Bug Fixed in real data GPU Prefetcher that could cause a crash when the dataset
was returning a dictionary of values.
- Raised `abseil` python verbosity to `ERROR` level to declutter `stdout`.

## [2.0.1] - 2022.09.30 - @DEKHTIARJonathan

- Change from `time.time()` to `time.perf_counter()` for better time
measurement.

## [2.0.0] - 2022.08.04 - @DEKHTIARJonathan

- Fix for XLA FP16 actually not being applied due to `"min_graph_nodes": -1`
missing. Therefore AMP skipped when XLA is able to compile the whole graph in
virtually one node.

- HF BERT & BART target changed:
    - `TFBertForPreTraining` -> `TFBertModel`
    - `TFBartForConditionalGeneration` -> `TFBartModel`

## [1.2.0] - 2022.07.31 - @DEKHTIARJonathan

Setting up the benchmarking suite to allow remote upload and storage of the
metrics and experiments.

Adding arguments:
* `--model_name`: Name of the model being executed.
* `--model_source`: Name of the model's source that originally published the model.
* `--experiment_name`: Name of the experiment being conducted.
* `--upload_metrics_endpoint`: Distant endpoint being used to push metrics to.

## [1.1.0] - 2022.07.25 - @DEKHTIARJonathan

Replacing all `print()` calls by `logging.<level>()` calls

## [1.0.1] - 2022.07.25 - @DEKHTIARJonathan

Removing AutoTuning on `get_dequeue_batch_fn` because DALIDataset was not
respecting the limit on the number of batches.

It should not impact the benchmark results, most of the time, the autotuner was
selecting the eager version anyway.

## [1.0.0] - 2022.07.20 - @DEKHTIARJonathan

Initial Versioning Release.
