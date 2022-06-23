# Installing dependencies if needed:

pip install pybind11 ujson Cython

python -c "from pycocotools.coco import COCO" > /dev/null 2>&1
DEPENDENCIES_STATUS=$?

if [[ ${DEPENDENCIES_STATUS} != 0 ]]; then
    # Master Branch of 2022/01/11
    PYCOCOTOOLS_BRANCH_OR_COMMIT_ID="142b17a358fdb5a31f9d5153d7a9f3f1cd385178"
    pip install "git+https://github.com/NVIDIA/cocoapi.git@${PYCOCOTOOLS_BRANCH_OR_COMMIT_ID}#egg=pycocotools&subdirectory=PythonAPI"
fi
