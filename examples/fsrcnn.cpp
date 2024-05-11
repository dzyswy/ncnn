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

namespace fsrcnn_x4_param_id {
const int LAYER_input_1 = 0;
const int BLOB_input_1 = 0;
const int LAYER__first_part_first_part_0_Conv = 1;
const int BLOB__first_part_first_part_0_Conv_output_0 = 1;
const int LAYER__first_part_first_part_1_PRelu = 2;
const int BLOB__first_part_first_part_1_PRelu_output_0 = 2;
const int LAYER__mid_part_mid_part_0_Conv = 3;
const int BLOB__mid_part_mid_part_0_Conv_output_0 = 3;
const int LAYER__mid_part_mid_part_1_PRelu = 4;
const int BLOB__mid_part_mid_part_1_PRelu_output_0 = 4;
const int LAYER__mid_part_mid_part_2_Conv = 5;
const int BLOB__mid_part_mid_part_2_Conv_output_0 = 5;
const int LAYER__mid_part_mid_part_3_PRelu = 6;
const int BLOB__mid_part_mid_part_3_PRelu_output_0 = 6;
const int LAYER__mid_part_mid_part_4_Conv = 7;
const int BLOB__mid_part_mid_part_4_Conv_output_0 = 7;
const int LAYER__mid_part_mid_part_5_PRelu = 8;
const int BLOB__mid_part_mid_part_5_PRelu_output_0 = 8;
const int LAYER__mid_part_mid_part_6_Conv = 9;
const int BLOB__mid_part_mid_part_6_Conv_output_0 = 9;
const int LAYER__mid_part_mid_part_7_PRelu = 10;
const int BLOB__mid_part_mid_part_7_PRelu_output_0 = 10;
const int LAYER__mid_part_mid_part_8_Conv = 11;
const int BLOB__mid_part_mid_part_8_Conv_output_0 = 11;
const int LAYER__mid_part_mid_part_9_PRelu = 12;
const int BLOB__mid_part_mid_part_9_PRelu_output_0 = 12;
const int LAYER__mid_part_mid_part_10_Conv = 13;
const int BLOB__mid_part_mid_part_10_Conv_output_0 = 13;
const int LAYER__mid_part_mid_part_11_PRelu = 14;
const int BLOB__mid_part_mid_part_11_PRelu_output_0 = 14;
const int LAYER__last_part_ConvTranspose = 15;
const int BLOB_52 = 15;
} // namespace fsrcnn_x4_param_id

#if 1



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


    cv::Mat yuv_img;
    cv::cvtColor(img, yuv_img, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> channels;
    cv::split(yuv_img, channels);

    ncnn::Net net; 
    net.load_param("/work/models/FSRCNN/fsrcnn_x4.param");
    net.load_model("/work/models/FSRCNN/fsrcnn_x4.bin");
    net.debug_info();
    

    // std::vector<ncnn::Layer*>& layers = net.mutable_layers();
    // for (int i = 0; i < layers.size(); i++)
    // {
    //     ncnn::Layer* layer = layers[i];
    //     printf("%d: %s, type=%s\n", i, layer->name.c_str(), layer->type.c_str());
    //     // for (int j = 0; j < layer->top_shapes.size(); j++)
    //     // {
    //     //     ncnn::Mat& shape = layer->top_shapes[j];
    //     //     printf("    top%d: w=%d, h=%d, d=%d, c=%d\n", j, shape.w, shape.h, shape.d, shape.c);
    //     // }

    //     // for (int j = 0; j < layer->bottom_shapes.size(); j++)
    //     // {
    //     //     ncnn::Mat& shape = layer->bottom_shapes[j];
    //     //     printf("    btm%d: w=%d, h=%d, d=%d, c=%d\n", j, shape.w, shape.h, shape.d, shape.c);
    //     // }
    // }
    


    cv::Mat nimg;
    channels[0].convertTo(nimg, CV_32F, 1 / 256.0);

    ncnn::Mat in(img.cols, img.rows, 1, (void*)nimg.data);
    in = in.clone();

    ncnn::Mat out;
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(false);
    // ex.input("data", in);
    // ex.extract("prob", out);
    ex.input(fsrcnn_x4_param_id::BLOB_input_1, in);
    ex.extract(fsrcnn_x4_param_id::BLOB_52, out);


    // ncnn::Deconvolution* conv1 = (ncnn::Deconvolution*)net.mutable_layers()[15];
    // printf("type=%s, name=%s, num_output=%d, kernel_w=%d, kernel_h=%d\n", conv1->type.c_str(), conv1->name.c_str(), conv1->num_output, conv1->kernel_w, conv1->kernel_h);
    // printf("weight_data: w=%d, h=%d, d=%d, c=%d, total=%d\n", conv1->weight_data.w, conv1->weight_data.h, conv1->weight_data.d, conv1->weight_data.c, conv1->weight_data.total());
    // printf("bias_data: w=%d, h=%d, d=%d, c=%d\n", conv1->bias_data.w, conv1->bias_data.h, conv1->bias_data.d, conv1->bias_data.c);

    // printf("wt0:%f\n", conv1->bias_data[0]);

    // if (conv1->weight_data.empty()) {
    //     printf("Convolution weight_data empty!\n");
    // }
    // printf("weight_data: %p\n", conv1->weight_data.data);


    // std::vector<ncnn::Blob> blobs = net.mutable_blobs();
    // for (int i = 0; i < blobs.size(); i++)
    // {
    //     ncnn::Blob blob = blobs[i];
    //     printf("%d: %s w=%d, h=%d, d=%d, c=%d\n", i, blob.name.c_str(), blob.shape.w, blob.shape.h, blob.shape.d, blob.shape.c);
    // }

    
    cv::Mat fimg;
    fimg.create(out.h, out.w, CV_32F);
    memcpy((uchar*)fimg.data, out.data, out.w * out.h * sizeof(float));

    cv::Mat sr_aimg;
    fimg.convertTo(sr_aimg, CV_8U, 255.0);


    cv::Mat sr_uimg;
    cv::Mat sr_vimg;
    cv::resize(channels[1], sr_uimg, sr_aimg.size());
    cv::resize(channels[2], sr_vimg, sr_aimg.size());

    std::vector<cv::Mat> sr_channels(3);
    sr_channels[0] = sr_aimg.clone();
    sr_channels[1] = sr_uimg.clone();
    sr_channels[2] = sr_vimg.clone();

    cv::Mat fsrcnn_img;
    cv::merge(sr_channels, fsrcnn_img);
    cv::cvtColor(fsrcnn_img, fsrcnn_img, cv::COLOR_YCrCb2BGR);

    cv::imwrite("fsrcnn_img.png", fsrcnn_img);


    return 0;
}

#else
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

    cv::Mat img; 
    cv::resize(m, img, cv::Size(64, 64), 0, 0, cv::INTER_CUBIC);
    cv::imwrite("lr_img.png", img);

    cv::Mat cubic_img; 
    cv::resize(img, cubic_img, cv::Size(256, 256), 0, 0, cv::INTER_CUBIC);
    cv::imwrite("cubic_img.png", cubic_img);

    cv::Mat yuv_img;
    cv::cvtColor(img, yuv_img, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> channels;
    cv::split(yuv_img, channels);

    ncnn::Net net;
    net.load_param("/work/models/FSRCNN/fsrcnn_x4.param");
    net.load_model("/work/models/FSRCNN/fsrcnn_x4.bin");

    cv::Mat nimg;
    channels[0].convertTo(nimg, CV_32F, 1 / 256.0);

    ncnn::Mat in(img.cols, img.rows, 1, (void*)nimg.data);
    in = in.clone();

    ncnn::Mat out;
    ncnn::Extractor ex = net.create_extractor();
    // ex.input("data", in);
    // ex.extract("prob", out);
    ex.input(fsrcnn_x4_param_id::BLOB_input_1, in);
    ex.extract(fsrcnn_x4_param_id::BLOB_52, out);


 

    cv::Mat fimg;
    fimg.create(out.h, out.w, CV_32F);
    memcpy((uchar*)fimg.data, out.data, out.w * out.h * sizeof(float));

    cv::Mat sr_aimg;
    fimg.convertTo(sr_aimg, CV_8U, 255.0);


    cv::Mat sr_uimg;
    cv::Mat sr_vimg;
    cv::resize(channels[1], sr_uimg, cv::Size(256, 256));
    cv::resize(channels[2], sr_vimg, cv::Size(256, 256));

    std::vector<cv::Mat> sr_channels(3);
    sr_channels[0] = sr_aimg.clone();
    sr_channels[1] = sr_uimg.clone();
    sr_channels[2] = sr_vimg.clone();

    cv::Mat fsrcnn_img;
    cv::merge(sr_channels, fsrcnn_img);
    cv::cvtColor(fsrcnn_img, fsrcnn_img, cv::COLOR_YCrCb2BGR);

    cv::imwrite("fsrcnn_img.png", fsrcnn_img);


    return 0;
}


#endif






