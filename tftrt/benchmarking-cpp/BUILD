# Description:
#   TensorFlow C++ inference example with TF-TRT model.

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_protos_profiler_service",
)

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

tf_cc_binary(
    name = "tftrt_benchmark_runner",
    srcs = [
        "main.cc",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core/profiler/rpc/client:capture_profile",
        "//tensorflow/core/profiler/rpc/client:profiler_client",
    ] + tf_protos_profiler_service(),
)