

#include "model.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>


int main(int argc, const char* argv[])
{
    auto nsamples = 2;
    std::string modelPath = "D:\\workspace\\camera_blur_recognition\\feature_engeering\\Blur\\Xgboost\\build\\bin\\Release\\model\\model_file_name.json";


    auto xgb = XGBoostPP(modelPath, 2);  // 4列, label有3个, iris例子中分别为三种类型的花，回归任何的话，这里nlabel = 1即可

    //result = array([[9.9658281e-01, 2.4966884e-03, 9.2058454e-04],
    //       [9.9608469e-01, 2.4954407e-03, 1.4198524e-03]], dtype=float32)
    //std::vector<double> features = { -4.03841739638405e-18, 0.011277463, 1.392156863, 21.8832286, 1116.92017, 255, 0.339913888, 0.024874329, 0.001648871, 0.504230469 };


    float features[20] = {-4.03841739638405e-18, 0.011277463, 1.392156863, 21.8832286, 1116.92017, 255, 0.339913888, 0.024874329, 0.001648871, 0.504230469,
        2.79521e-19, 0.00029984, 0.176470588, 6.227292379, 30.49339293, 97, 0.344544962, 0.007255797, 3.27241E-05, 0.095415883};
        

    std::vector<float> y;
    size_t nrow=2, ncol=10;
    auto ret = xgb.predict(features, nrow,ncol,y);
    if (ret != 0){
        std::cout << "predict error" << std::endl;
    }
    
    for (int i=0; i < nrow; i++) {
        std::cout <<"y:" << y[i] << std::endl;
    }
}
