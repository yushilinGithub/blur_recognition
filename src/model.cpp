#include "model.h"

XGB::XGB(std::string const& path, uint64_t nlabels): _modelPath(path),_nlabels(nlabels){
    
    std::fstream infile(_modelPath);
    if (!infile.good()) {
        //LOG HERE
        std::cout << "con't find file:" << _modelPath << std::endl;
    }
    if (XGBoosterCreate(NULL, 0, &_booster) == 0 &&  XGBoosterLoadModel(_booster, _modelPath.c_str()) == 0){
        //LOG HERE
    }else{
        //LOG HERE
        _booster = NULL;
    }        
}

int XGB::predict(const float* features,const size_t nrow,const size_t ncol, std::vector<float>& result){
    DMatrixHandle X;
    XGDMatrixCreateFromMat(features, nrow, ncol, NAN, &X);
    const float* out;
    uint64_t l;
    auto ret = XGBoosterPredict(_booster, X, 0, 0, 0, &l, &out);
    if (ret < 0){
        // LOG HERE
        std::cout << "ret<0" << std::endl;
        return -1;
    }
    for (size_t s = 0; s <= sizeof(*out) / sizeof(float); s++) {
        result.push_back(out[s]);
    }
    XGDMatrixFree(X);

    return 0;
}
