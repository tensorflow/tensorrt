#ifndef TFTRT_SYNTHETIC_H
#define TFTRT_SYNTHETIC_H

#include "base.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/core/framework/tensor.h"

namespace datasets {

class SyntheticDataset : public BaseDataset {
public:
    SyntheticDataset(std::initializer_list<int64_t> shape) {
        tensorflow::Input::Initializer inputTensor(1.0f, tensorflow::TensorShape(shape));
        mData = inputTensor.tensor;
    }
    tensorflow::Tensor getTensor(int offset, int batch_size) override {
        return mData;
    }
private:
    tensorflow::Tensor mData;
};

} // namespace datasets

#endif // TFTRT_SYNTHETIC_H