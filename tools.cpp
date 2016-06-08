#include <iostream>
#include <string>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

inline bool is_in_mask(const uchar val) {
    return val < 253; // TODO fix this
}

void apply_median_filter(const std::string &in_file, 
                         const std::string &out_file,
                         const int radius,
                         const int min_found = 0) {
    cv::Mat in_img = cv::imread(in_file, cv::IMREAD_GRAYSCALE);
    cv::Mat out_img = in_img.clone();
    int sum, found;
    for (int y = 0; y < in_img.rows; ++y) {
        for (int x = 0; x < in_img.cols; ++x) {
            sum = 0;
            found = 0;
            for (int y1 = std::max(y - radius, 0); 
                 y1 <= std::min(y + radius, in_img.rows - 1); ++y1) {
                for (int x1 = std::max(x - radius, 0); 
                     x1 <= std::min(x + radius, in_img.cols - 1); ++x1) {
                    size_t val = in_img.at<uchar>(y1, x1);
                    if (is_in_mask(val)) {
                        sum += val;
                        ++found;
                    }
                }
            }
            out_img.at<uchar>(y, x) = (found <= min_found ? 255 : 
                                       static_cast<double>(sum) / found);
        }
    }
    cv::imwrite(out_file, out_img);
}

int main(int argc, char **argv) {
    if (argc <= 1) {
        std::cerr << "usage: ... :(" << std::endl;
        return 0;
    }
    if (std::string(argv[1]) == std::string("--median_filter")) {
        apply_median_filter(std::string(argv[2]), 
                            std::string(argv[3]), 
                            std::stoi(std::string(argv[4])));
    }
}
