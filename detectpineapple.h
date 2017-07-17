#ifndef DETECTPINEAPPLE_H
#define DETECTPINEAPPLE_H

#include "opencv2/imgproc.hpp"
using namespace cv;
using namespace std;

class DetectPineapple   {

public:
    const int USE_CUDA = 0;
    const int DONT_USE_CUDA = 1;

    // Grayscaling method
    void grayscale(Mat &image);
    // noises reduction method
    void reductNoises(Mat &image, int FILTER_SIZE);
    // binaryzation (otsu method)
    void convertToBinary(Mat &grayImage);
    // object labelling method
    vector<Rect> labelling(Mat &binaryImage, Mat &captureFrame);
    // drawing object ROI method
    void drawROIObject(Mat &originalFrame, vector<Rect> theRect);
    // extract object points
    vector<Point> getObjectPoints(Mat &binaryImage);
    // object orientation analysis method
    void objectOrientation(vector<Point> pts, vector<float> &orientations, Point &center);
    // draw object orientation
    void drawPCAAxis(Mat &img, Point center, Point q, Scalar color, const float scale);
    // transform object points
    void transformPoints(Mat &myImage, float orientation, vector<Point> &transformedPoints);
    // transform SINGLE point
    void transformPoint(Point &pointInput, Point &pointOutput, float orientation);
    // crown and fruit separation method
    void cutTheCrown(vector<Point> &transformedPoints, int &cutPoint);
    // draw sparated line
    void drawSparatedLine(Mat &myImage, float orientation, Point &cutPoint, float &a, int &b);
    // draw sparator line
    void drawSparatorLine(Mat &myImage, float &a, int &b, Point &firstPoint, Point &secondPoint);
    // fruit region recognition method
    void compactnessAnalyze(Mat &myImage, float a, int b, int &result, float &compactnessUp, float &compactnessDown);
    // write fruit and crown
    void writeLabel(Mat &image, Point firstPoint, Point secondPoint, float a, int b, int result);
    // give color for fruit
    void giveFruitColor(Mat &binaryImage, Mat &colorImage, float a, int b, int result, String givenColor);
};

#endif // DETECTPINEAPPLE_H
