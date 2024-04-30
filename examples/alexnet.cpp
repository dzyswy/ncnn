#include "net.h"
#include "mat.h" 
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <stdlib.h>
#include <float.h>
#include <stdio.h>
#include <vector>

namespace alexnet_param_id {
const int LAYER_data = 0;
const int BLOB_data = 0;
const int LAYER_conv1 = 1;
const int BLOB_conv1 = 1;
const int LAYER_relu1 = 2;
const int BLOB_conv1_relu1 = 2;
const int LAYER_norm1 = 3;
const int BLOB_norm1 = 3;
const int LAYER_pool1 = 4;
const int BLOB_pool1 = 4;
const int LAYER_conv2 = 5;
const int BLOB_conv2 = 5;
const int LAYER_relu2 = 6;
const int BLOB_conv2_relu2 = 6;
const int LAYER_norm2 = 7;
const int BLOB_norm2 = 7;
const int LAYER_pool2 = 8;
const int BLOB_pool2 = 8;
const int LAYER_conv3 = 9;
const int BLOB_conv3 = 9;
const int LAYER_relu3 = 10;
const int BLOB_conv3_relu3 = 10;
const int LAYER_conv4 = 11;
const int BLOB_conv4 = 11;
const int LAYER_relu4 = 12;
const int BLOB_conv4_relu4 = 12;
const int LAYER_conv5 = 13;
const int BLOB_conv5 = 13;
const int LAYER_relu5 = 14;
const int BLOB_conv5_relu5 = 14;
const int LAYER_pool5 = 15;
const int BLOB_pool5 = 15;
const int LAYER_fc6 = 16;
const int BLOB_fc6 = 16;
const int LAYER_relu6 = 17;
const int BLOB_fc6_relu6 = 17;
const int LAYER_drop6 = 18;
const int BLOB_fc6_drop6 = 18;
const int LAYER_fc7 = 19;
const int BLOB_fc7 = 19;
const int LAYER_relu7 = 20;
const int BLOB_fc7_relu7 = 20;
const int LAYER_drop7 = 21;
const int BLOB_fc7_drop7 = 21;
const int LAYER_fc8 = 22;
const int BLOB_fc8 = 22;
const int LAYER_prob = 23;
const int BLOB_prob = 23;
}



int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    ncnn::Net net;
    net.load_param("/work/models/bvlc_alexnet/alexnet.param");
    net.load_model("/work/models/bvlc_alexnet/alexnet.bin");


    unsigned char* rgbdata = m.data;// data pointer to RGB image pixels
    int w = m.cols;// image width
    int h = m.rows;// image height
    int target_width = 227;// target resized width
    int target_height = 227;// target resized height
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgbdata, ncnn::Mat::PIXEL_RGB, w, h, target_width, target_height);

    const float mean_vals[3] = {104.f, 117.f, 123.f};
    in.substract_mean_normalize(mean_vals, 0);

    ncnn::Mat out;
    ncnn::Extractor ex = net.create_extractor();
    // ex.input("data", in);
    // ex.extract("prob", out);
    ex.input(alexnet_param_id::BLOB_data, in);
    ex.extract(alexnet_param_id::BLOB_prob, out);

    ncnn::Mat out_flatterned = out.reshape(out.w * out.h * out.c);
    std::vector<float> scores;
    scores.resize(out_flatterned.w);

    float max_score = 0;
    int max_id = 0;
    for (int j=0; j<out_flatterned.w; j++)
    {
        scores[j] = out_flatterned[j];
        printf("%d: %f\n", j, scores[j]);
        if (scores[j] > max_score) {
            max_score = scores[j];
            max_id = j;
        }
    }
    printf("max_id=%d, max_score=%f\n", max_id, max_score);

    NCNN_LOGE("NCNN_LOGE");
    return 0;
}









