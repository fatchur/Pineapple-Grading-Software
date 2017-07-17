#include <QCoreApplication>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "detectpineapple.h"

using namespace cv;
using namespace std;

int main()
{
    VideoCapture capturedDevice(0);
    if (!capturedDevice.isOpened()) {
        cout << "cannot open camera";
    }

    /*
    vector<String> filenames;
    String folder = "/media/fatchur/9E8F-67DD/pesiapan jurnal/gambar/";
    glob(folder, filenames);    */

    int a=0;
    while (true) {
        Mat cameraFrame;
        capturedDevice.read(cameraFrame);
        //cameraFrame = imread(filenames[i]);
        Mat origFrame = cameraFrame.clone();

        // build detect pineapple object
        DetectPineapple detectPineapple;
        // convert image to grayscale
        detectPineapple.grayscale(cameraFrame);
        // reduct noises of gray image
        detectPineapple.reductNoises(cameraFrame, 7);
        // convert to binary
        detectPineapple.convertToBinary(cameraFrame);
        // closing
        int morph_size = 2;
        Mat element = getStructuringElement( MORPH_RECT, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
        morphologyEx(cameraFrame, cameraFrame, MORPH_CLOSE, element, Point (-1,-1));

        // labelling
        vector<Rect> objectRect = detectPineapple.labelling(cameraFrame, origFrame);
        // draw object ROI
        detectPineapple.drawROIObject(origFrame, objectRect);

        a ++;
        cout << a ;
        cout<< objectRect.size() << endl;

        // for every object ROI in "objectRect"
        for( int i=0; i<objectRect.size(); i++){
            // crop the binary image at ROI [i]
            Mat croppedImageBinary = cameraFrame(objectRect[i]);
            // we also need to crop the color image at ROI
            Mat croppedColorImage = origFrame(objectRect[i]);
            // get points vector of this object
            vector<Point> objectPoints = detectPineapple.getObjectPoints(croppedImageBinary);

            // get object orientation
            // this method return two values
            // 1. 2 float values of orientation
            // 2. a center point of object
            vector<float> orientations;
            Point centerPoint;
            detectPineapple.objectOrientation(objectPoints, orientations, centerPoint);
            // now we get two eigen values
            // we used it as new coordinate system

            // Draw the principal components
            //Point p1 = centerPoint + 0.02 * Point(static_cast<int>(orientations[0].x * eigen_val[0]), static_cast<int>(orientations[0].y * eigen_val[0]));
            //Point p2 = centerPoint - 0.02 * Point(static_cast<int>(orientations[1].x * eigen_val[1]), static_cast<int>(orientations[1].y * eigen_val[1]));

            // transforming "objectPoints" to new coordinate system
            vector<Point> transformedPoints;
            detectPineapple.transformPoints(croppedImageBinary, orientations[0], transformedPoints);

            // analyze the cut point
            int cutPoint;
            detectPineapple.cutTheCrown(transformedPoints, cutPoint);

            // draw the spartated line
            // the cut point is based on new ordinat system
            // transform it back to the old ordinat system
            Point cutPointOriginalOrdinat, cutPointTransformedOrdinat;
            cutPointTransformedOrdinat.x = cutPoint;
            cutPointTransformedOrdinat.y = 0;
            detectPineapple.transformPoint(cutPointTransformedOrdinat, cutPointOriginalOrdinat, -orientations[0]);

            // --- NOW we start to draw the line
            // stright line equation: y = ax + b
            float a; int b;
            Point p1, p2;
            detectPineapple.drawSparatedLine(croppedColorImage, orientations[1], cutPointOriginalOrdinat, a, b);

            // determine the fruit region
            // resultCode = 0 ---> image Points > separator line are fruit region, else crown region
            // resultCode = 1 ---> image Points < separator line are fruit region, else crown region
            int result;
            float compactnessOfUpperRegion, compactnessOfLowerRegion;
            Mat binaryCopy = croppedImageBinary.clone();
            detectPineapple.compactnessAnalyze(binaryCopy, a, b, result, compactnessOfUpperRegion, compactnessOfLowerRegion);

            // draw real sparator line
            detectPineapple.drawSparatorLine(croppedColorImage, a, b, p1, p2);

            // write label (crown or fruit) in each regian
            // detectPineapple.writeLabel(croppedColorImage, p1, p2, a, b, result);

            // give color to fruit
            String givenColor = "red";
            detectPineapple.giveFruitColor(croppedImageBinary, croppedColorImage, a, b, result, givenColor);

            // now paste the "croppedImageColor" to "origframe"
            croppedColorImage.copyTo(origFrame(objectRect[i]));
        }

        //imwrite(folder + "colored" + to_string(a) + ".jpg", origFrame);

        imshow("cam", origFrame);
        //imshow("orig", origFrame);
        waitKey(100);
    }
    return 0;
}
