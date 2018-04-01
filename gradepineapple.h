#ifndef GRADEPINEAPPLE_H
#define GRADEPINEAPPLE_H

#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

class GradePineapple {
public:
    void extractRGB(Mat &colorImage, Mat &binaryImage, float a, int b, int result, int &R, int &G, int &B);
    void convertRGBtoHSI(int R, int G, int B, float H, float S, float I);
    void predictGrade(int R, int G, int B, string &grade);
};

#endif // GRADEPINEAPPLE_H
