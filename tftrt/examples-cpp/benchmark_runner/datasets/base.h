#ifndef TFTRT_BASE_H
#define TFTRT_BASE_H

#include "tensorflow/core/framework/tensor.h"

namespace datasets {

class BaseDataset {
public:
    virtual tensorflow::Tensor getTensor(int offset, int batch_size) = 0;
};

} // namespace datasets

#endif // TFTRT_BASE_H