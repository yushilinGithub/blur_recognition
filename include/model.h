#ifndef __MODEL_H__

#define __MODEL_H__

#include <string>
#include <xgboost/c_api.h>
#include <vector>
#include <iostream>
#include <fstream>

class XGB
{
public:
    XGB(std::string const& path, uint64_t nlabels);
    int predict(const float* features,const size_t nrow,const size_t ncol, std::vector<float>& result);
    virtual ~XGB(){
        XGBoosterFree(_booster);
    }

    
private:
    std::string const _modelPath;
    BoosterHandle _booster;
    uint64_t const _nlabels;
};

#endif
