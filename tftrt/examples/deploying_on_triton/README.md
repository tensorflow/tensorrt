# Tensorflow with TensorRT (TF-TRT) to Triton

This README showcases how to deploy a simple ResNet model accelerated by using Tensorflow-TensorRT on Triton Inference Server.

## Step 1: Optimize your model with TensorFlow-TensorRT

If you are unfamiliar with Tensorflow-TensorRT, please refer to this [video](https://www.youtube.com/watch?v=w7871kMiAs8&ab_channel=NVIDIADeveloper). The first step in this pipeline is to accelerate your model. If you use TensorFlow as your framework of choice for training, you can either use TensorRT or TensorFlow-TensorRT, depending on your model's operations.

For using Tensorflow-TensorRT, let's first pull the NGC TensorFlow Docker container, which comes installed with both TensorRT and Tensorflow-TensorRT. You may need to create an account and get the API key from [here](https://ngc.nvidia.com/setup/). Sign up and login with your key (follow the instructions [here](https://ngc.nvidia.com/setup/api-key) after signing up).

```
# <xx.xx> is the yy:mm for the publishing tag for NVIDIA's Tensorflow 
# container; eg. 22.04

docker run -it --gpus all -v /path/to/this/folder:/resnet50_eg nvcr.io/nvidia/tensorflow:<xx.xx>-tf2-py3
```

We have already made a sample to use Tensorflow-TensorRT: `tf_trt_resnet50.py`. This sample downloads a ResNet model from Keras and then optimizes it with TensorFlow-TensorRT. For more examples, visit the TF-TRT [Github Repository](https://github.com/tensorflow/tensorrt).

```
python tf_trt_resnet50.py

# you can exit out of this container now
exit
```

## Step 2: Set Up Triton Inference Server

If you are new to the Triton Inference Server and want to learn more, we highly recommend checking out our [Github Repository](https://github.com/triton-inference-server).

To use Triton, we need to make a model repository. The structure of the repository should look something like this:
```
model_repository
|
+-- resnet50
    |
    +-- config.pbtxt
    +-- 1
        |
        +-- model.savedmodel
            |
            +-- saved_model.pb
            +-- variables
                |
                +-- variables.data-00000-of-00001
                +-- variables.index
```

A sample model configuration of the model is included with this demo as `config.pbtxt`. If you are new to Triton, we highly encourage you to check out this [section of our documentation](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md) for more details. Once you have the model repository setup, it is time to launch the Triton server! You can do that with the docker command below.
```
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /full/path/to/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/models --backend-config=tensorflow,version=2
```

## Step 3: Using a Triton Client to Query the Server

Download an example image to test inference.

```
wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
```

Install dependencies.
```
pip install --upgrade tensorflow
pip install pillow
pip install nvidia-pyindex
pip install tritonclient[all]
```

Run client
```
python3 triton_client.py
```

The output of the same should look like below:
```
[b'0.301167:90' b'0.169790:14' b'0.161309:92' b'0.093105:94'
 b'0.058743:136' b'0.050185:11' b'0.033802:91' b'0.011760:88'
 b'0.008309:989' b'0.004927:95' b'0.004905:13' b'0.004095:317'
 b'0.004006:96' b'0.003694:12' b'0.003526:42' b'0.003390:313'
 ...
 b'0.000001:751' b'0.000001:685' b'0.000001:408' b'0.000001:116'
 b'0.000001:627' b'0.000001:933' b'0.000000:661' b'0.000000:148']
```
The output format here is `<confidence_score>:<classification_index>`. To learn how to map these to the label names and more, refer to our [documentation](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_classification.md).