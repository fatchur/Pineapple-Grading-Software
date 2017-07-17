#include "detectpineapple.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/photo.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void DetectPineapple::grayscale(Mat &image){
    cvtColor(image, image, CV_BGR2GRAY);
}
void DetectPineapple::reductNoises(Mat &image, int FILTER_SIZE){
    //fastNlMeansDenoising(image, image, 3, 7, 21);

    int border = (FILTER_SIZE-1) / 2;
        int jumlahWarna = 0, rataRata = 0;

        for(int y = border; y < image.rows - border; y++)
        {
            for(int x = border ; x < image.cols-border; x++)
            {
                // applying median filter
                for (int i = -1 * border ; i <= border; i ++)
                {
                    for(int j = -1 * border; j <= border; j++)
                    {
                        int warna = image.at<uchar>(y + i , x + j);
                        jumlahWarna += warna;
                    }
                }

                rataRata = (int)(jumlahWarna / (FILTER_SIZE * FILTER_SIZE));
                image.at<uchar>(y,x)=rataRata;
                jumlahWarna = 0;
                rataRata = 0;
            }
        }
}
void DetectPineapple::convertToBinary(Mat &grayImage){
    threshold(grayImage, grayImage, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
}
vector<Rect> DetectPineapple::labelling(Mat &binaryImage, Mat &captureFrame){
    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground
    vector<Rect> areaObject;
    cv::Mat label_image;
    binaryImage.convertTo(label_image, CV_32SC1);

    int label_count = 2; // starts at 2 because 0,1 are used already
    //saat penelitian y = 60
    for(int y=0; y < label_image.rows; y++)
       {
          int *row = (int*)label_image.ptr(y);
          //saat penelitian x = 280
          for(int x=0; x < label_image.cols; x++)
             {
              if(row[x] != 255)
                 {	continue;	}

              cv::Rect rect;
              cv::floodFill(label_image, cv::Point(x,y), label_count, &rect, 0, 0, 4);

              if ((rect.width*rect.height)> 1000 & (rect.width*rect.height) < 200000 )
                 {
                   areaObject.push_back(rect);
                   rectangle (captureFrame, Point(rect.x, rect.y), Point(rect.x+rect.width, rect.y+rect.height) , cvScalar(0,255,0,0), 2 , 8 , 0);
                   //circle( captureFrame, Point(rect.x+rect.width/2, rect.y+rect.height/2), 5 , Scalar( 0, 0, 255 ), 2 );
                 }
            }
        }

    return areaObject;
}
void DetectPineapple::drawROIObject(Mat &originalImage, vector<Rect> theRect){
    for (uint i=0; i<theRect.size(); i++){
        rectangle(originalImage, theRect[i], cvScalar(255,0,0,255), 1, 8, 0);
    }
}
vector<Point> DetectPineapple::getObjectPoints(Mat &binaryImage)	{

    vector<Point> objectPoints;

    for(int i = 0; i < binaryImage.rows; i ++)	{
        for(int j = 0; j < binaryImage.cols; j++)	{
            int warna = binaryImage.at<uchar>(i,j);
            if(warna == 255)	{
                objectPoints.push_back(Point(j,i));
            }
        }
    }

    return objectPoints;
}
void DetectPineapple::objectOrientation(vector<Point> pts, vector<float> &orientations, Point &center)	{

    int sz = static_cast<int>(pts.size());
    Mat data_pts = Mat(sz, 2, CV_64FC1);

    for (int i = 0; i < data_pts.rows; ++i)
    {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }

    //Perform PCA analysis
    PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW);
    //Store the center of the object
    center = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)), static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
    //Store the eigenvalues and eigenvectors
    vector<Point2d> eigen_vecs(2);
    vector<double> eigen_val(2);
    for (int i = 0; i < 2; ++i)
    {
        eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                pca_analysis.eigenvectors.at<double>(i, 1));
        eigen_val[i] = pca_analysis.eigenvalues.at<double>(0, i);
    }

    orientations.push_back(atan2(eigen_vecs[0].y, eigen_vecs[0].x)); // orientation in radians
    orientations.push_back(atan2(eigen_vecs[1].y, eigen_vecs[1].x));

}
void DetectPineapple::transformPoints(Mat &myImage, float orientation, vector<Point> &transformedPoints)	{
    //parameter konversi
    float a = cos(-orientation);
    float b = -sin(-orientation);
    float c = sin(-orientation);
    float d = cos(-orientation);

    for(int i=0; i<myImage.rows; i++)
        {
            for (int j=0; j<myImage.cols; j++)
                {
                    int value = myImage.at<uchar>(i,j);
                    if(value ==255)	{
                        int xAksen = a*j + b*i;
                        int yAksen = c*j + d*i;
                        transformedPoints.push_back(Point(xAksen, yAksen));
                    }
                }
        }
}
void DetectPineapple::cutTheCrown(vector<Point> &transformedPoints, int &cutPoint)	{

    int maxValue = transformedPoints[0].x;
    int minValue = transformedPoints[0].x;

    for(uint i =1; i < transformedPoints.size(); i++)	{
        if (maxValue < transformedPoints[i].x)	{
            maxValue = transformedPoints[i].x;
        }
        if (minValue > transformedPoints[i].x)	{
            minValue = transformedPoints[i].x;
        }
    }

    int length = maxValue - minValue;
    int deviation = length/4;
    int center = (maxValue + minValue)/2;

    //setting semua anggota array = 0
    int numberOfPixelsPerX [1500];
    for (int i =0; i < 1500; i ++)	{
        numberOfPixelsPerX[i] = 0;
    }

    //mencari titik pemisah
    for (uint i = 0; i < transformedPoints.size(); i++)	{
        numberOfPixelsPerX[transformedPoints[i].x] += 1;
    }

    int theLeast = center-deviation;

    for(int i = theLeast; i < center + deviation; i++ )	{
        if (numberOfPixelsPerX[theLeast] > numberOfPixelsPerX[i])	{
            theLeast = i;
        }
    }
    cutPoint = theLeast;
}
void DetectPineapple::transformPoint(Point &pointInput, Point &pointOutput, float orientation)	{
    //parameter konversi
    float a = cos(-orientation);
    float b = -sin(-orientation);
    float c = sin(-orientation);
    float d = cos(-orientation);

    pointOutput.x = a*pointInput.x + b*pointInput.y;
    pointOutput.y = c*pointInput.x + d*pointInput.y;
}
void DetectPineapple::drawSparatedLine(Mat &myImage, float orientation, Point &cutPoint, float &a, int &b)	{
    a = tan(orientation);
    b = cutPoint.y - cutPoint.x * a;

    for (int i = 0; i < myImage.rows; i ++)	{
        for ( int j = 0; j < myImage.cols; j++)	{
            int value = a * j + b;
            if (i == value)	{
                myImage.at<Vec3b>(i,j)[0] = 255;
                myImage.at<Vec3b>(i,j)[1] = 255;
                myImage.at<Vec3b>(i,j)[2] = 255;
            }
        }
    }
}
void DetectPineapple::drawSparatorLine(Mat &myImage, float &a, int &b, Point &firstPoint, Point &secondPoint){
    
    // find cut point on y axis
    // it is called first point
    // case 1, if first point .y > 0 & < rows
    // case 2, if first point .y < 0: it througs top horizontal line
    // case 3, if first point .y > myImage.rows: it throughs under horizontall line
    // NOW, first case
    int x1 = 0;
    int y1 =  x1 * a + b;
    if (y1>=0 & y1<=myImage.rows){
        firstPoint.x = 0;
        firstPoint.y = y1;
    }
    else if (y1<0){
        firstPoint.y = 0;
        firstPoint.x = -b / a;
    }
    else if (y1>myImage.rows){
        firstPoint.y = myImage.rows;
        firstPoint.x = (myImage.rows -b)/a;
    }

    // NOW FOR SECOND POINT
    // find cut point on x axis
    // it is called first point
    // case 1, if first point .y > 0 & < rows
    // case 2, if first point .y < 0: it througs top horizontal line
    // case 3, if first point .y > myImage.rows: it throughs under horizontall line
    // NOW, first case
    int x2 = myImage.cols;
    int y2 = a * x2 + b;
    if (y2>=0 & y2<=myImage.rows){
        secondPoint.x = x2;
        secondPoint.y = y2;
    }
    else if (y2<0){
        secondPoint.y = 0;
        secondPoint.x = -b/a;
    }
    else if (y2>myImage.rows){
        secondPoint.y = myImage.rows;
        secondPoint.x = (myImage.rows -b)/a;
    }

    line(myImage, firstPoint, secondPoint, Scalar(255, 0,0, 255), 2, 8, 0);

}
void DetectPineapple::compactnessAnalyze(Mat &myImage, float a, int b, int &result, float &compactnessUp, float &compactnessDown)
{

    /* menghitung luas
     */

    int luasUp = 0;
    int luasDown = 0;
    for (int i = 0; i < myImage.rows; i++)
    {
        for (int j = 0; j < myImage.cols; j++)
        {
            int warna = myImage.at<uchar>(i,j);
            if (warna != 0)
            {
                if ( i >= (a*j + b))	{
                    luasUp ++;	}
                else {luasDown ++; }
            }
        }
    }

    /* deteksi tepi canny
    -treshold = 80;
    */
    Mat detected_edges;

    int edgeThresh = 1;
    int lowThreshold = 80;
    int const max_lowThreshold = 100;
    int ratio = 3;
    int kernel_size = 3;

    /// Canny detector
    Canny( myImage, myImage, lowThreshold, lowThreshold*ratio, kernel_size );

    // menghitung garis tepi objek

    int perimeterUp = 0;
    int perimeterDown = 0;
    for (int i = 0; i < myImage.rows; i++)
    {
        for (int j = 0; j < myImage.cols; j++)
        {
            int warna = myImage.at<uchar>(i,j);
            if (warna != 0)
            {
                if ( i >= (a*j + b))	{
                    perimeterUp ++;	}
                else {perimeterDown ++; }
            }
        }
    }

    compactnessUp = 4* luasUp / (3.14 * perimeterUp * perimeterUp);
    compactnessDown = 4* luasDown / (3.14 * perimeterDown * perimeterDown);

    if (compactnessUp > compactnessDown)	{result = 0;}
    else {result =1;}
    //putText(captureFrame, tostr(roundness), Point(rect2.x , rect2.y + myRect.y), FONT_HERSHEY_PLAIN, 1, cvScalar(255, 0,0, 255), 1, CV_AA);
}
void DetectPineapple::writeLabel(Mat &image, Point firstPoint, Point secondPoint, float a, int b, int result){
    vector<Point> lowerThan;
    vector<Point> upperThan;

    lowerThan.push_back(firstPoint);
    lowerThan.push_back(secondPoint);
    upperThan.push_back(firstPoint);
    upperThan.push_back(secondPoint);

    // checking for point in left and top corner (0,0)
    if (b > 0)  { lowerThan.push_back(Point (0,0)); }
    else { upperThan.push_back(Point(0,0));}
    // checking for point in left and bottom corner (0, image.rows)
    if (b > image.rows) {   lowerThan.push_back(Point(0, image.rows));}
    else {  upperThan.push_back(Point (0,image.rows));}
    // checking for point in right and top corner (imnage.cols,0)
    if ((a*image.cols + b) > 0) {   lowerThan.push_back(Point (image.cols,0)); }
    else { upperThan.push_back(Point(image.cols,0));}
    // checking for point in right and bottom corner (image.cols, image.rows)
    if ((a*image.cols + b) > image.rows)   {   lowerThan.push_back(Point (image.cols, image.rows)); }
    else { upperThan.push_back(Point(image.cols, image.rows));}

    Point centerOfLowerPart, centerOfUpperPart;

    // find center point of lower part
    int sumX=0, sumY=0;
    for (int i=0; i<lowerThan.size(); i++){
        sumX += lowerThan[i].x;
        sumY += lowerThan[i].y;
    }
    centerOfLowerPart.x = sumX / lowerThan.size()-20;
    centerOfLowerPart.y = sumY / lowerThan.size();

    // find center point of upper part
    sumX=0; sumY=0;
    for (int i=0; i<upperThan.size(); i++){
        sumX += upperThan[i].x;
        sumY += upperThan[i].y;
    }
    centerOfUpperPart.x = sumX / upperThan.size()-20;
    centerOfUpperPart.y = sumY / upperThan.size();

    // if upper part is fruit
    if (result == 0){
        putText(image, "fruit", centerOfUpperPart, CV_FONT_HERSHEY_PLAIN, 1.5, Scalar(0,0,0,255), 2, 8);
        putText(image, "crown", centerOfLowerPart, CV_FONT_HERSHEY_PLAIN, 1.5, Scalar(0,0,0,255), 2, 8);
    }
    else {
        putText(image, "crown", centerOfUpperPart, CV_FONT_HERSHEY_PLAIN, 1.5, Scalar(0,0,0,255), 2, 8);
        putText(image, "fruit", centerOfLowerPart, CV_FONT_HERSHEY_PLAIN, 1.5, Scalar(0,0,0,255), 2, 8);
    }

    /*
    for (int i=0; i<lowerThan.size(); i++){
        circle(image, lowerThan[i], 10, Scalar(0,0,255,255), 2, 8, 0);
    }*/

    //circle(image, centerOfUpperPart, 10, Scalar(0,0,255,255), 2, 8, 0);
    //circle(image, centerOfLowerPart, 10, Scalar(0,0,255,255), 2, 8, 0);
}
void DetectPineapple::giveFruitColor(Mat &binaryImage, Mat &colorImage, float a, int b, int result, String givenColor){

    if (givenColor == "red"){
        for (int i=0; i<binaryImage.rows; i++){
            for(int j=0; j<binaryImage.cols; j++){

                // checking the pixel is in fruit region or not
                if ( result == 0 & i>(a*j+b)){
                    int color = binaryImage.at<uchar>(i,j);

                    if (color == 255){
                        colorImage.at<Vec3b> (i,j)[0]=0;
                        colorImage.at<Vec3b> (i,j)[1]=0;
                        colorImage.at<Vec3b> (i,j)[2]=255;
                   }
                }
                else if (result == 1 & i<(a*j+b)){
                    int color = binaryImage.at<uchar>(i,j);

                    if (color == 255){
                        colorImage.at<Vec3b> (i,j)[0]=0;
                        colorImage.at<Vec3b> (i,j)[1]=0;
                        colorImage.at<Vec3b> (i,j)[2]=255;
                   }
                }
            }
        }
    }
}
void DetectPineapple::drawPCAAxis(Mat &img, Point center, Point q, Scalar color, const float scale)
{
    double angle;
    double hypotenuse;
    angle = atan2( (double) center.y - q.y, (double) center.x - q.x ); // angle in radians
    hypotenuse = sqrt( (double) (center.y - q.y) * (center.y - q.y) + (center.x - q.x) * (center.x - q.x));

    // double degrees = angle * 180 / CV_PI; // convert radians to degrees (0-180 range)
    // cout << "Degrees: " << abs(degrees - 180) << endl; // angle in 0-360 degrees range
    // Here we lengthen the arrow by a factor of scale

    q.x = (int) (center.x - scale * hypotenuse * cos(angle));
    q.y = (int) (center.y - scale * hypotenuse * sin(angle));
    line(img, center, q, color, 1, CV_AA);
    // create the arrow hooks
    center.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
    center.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
    line(img, center, q, color, 1, CV_AA);
    center.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
    center.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
    line(img, center, q, color, 1, CV_AA);

}
