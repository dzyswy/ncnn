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

#include "layer/convolution.h"
#include "layer/deconvolution.h"



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

    cv::imwrite("org_img.png", m);

    cv::Mat img = m.clone(); 

    cv::Mat cubic_img; 
    cv::resize(img, cubic_img, cv::Size(m.cols * 4, m.rows * 4), 0, 0, cv::INTER_CUBIC);
    cv::imwrite("cubic_img.png", cubic_img);


 

    ncnn::Net net; 
    net.load_param("/work/models/sesr/sesr_collapse_m5_x4.param");
    net.load_model("/work/models/sesr/sesr_collapse_m5_x4.bin");



    // cv::Mat a(h, w, CV_32FC3);
    cv::Mat a;
    img.convertTo(a, CV_32FC3, 1 / 256.0);
    ncnn::Mat in_pack3(a.cols, a.rows, 1, (void*)a.data, (size_t)4u * 3, 3);
    ncnn::Mat in;
    ncnn::convert_packing(in_pack3, in, 1);
    

    ncnn::Mat out;
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(false);
    ex.input("input", in);
    ex.extract("output", out);

    ncnn::Mat out_pack3;
    ncnn::convert_packing(out, out_pack3, 3);
    cv::Mat out_img(out.h, out.w, CV_32FC3);
    memcpy((uchar*)out_img.data, out_pack3.data, out.w * out.h * 3 * sizeof(float));

    cv::Mat sr_aimg;
    out_img.convertTo(sr_aimg, CV_8UC3, 255.0);

    cv::imwrite("sesr_img.png", sr_aimg);

    return 0;
}







