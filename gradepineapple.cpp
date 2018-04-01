#include "gradepineapple.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/photo.hpp"

using namespace cv;
using namespace std;

void GradePineapple::extractRGB(Mat &colorImage, Mat &binaryImage, float a, int b, int result, int &R, int &G, int &B) {

    int sumR = 0, sumG = 0, sumB = 0, objPixels = 0;
    erode(binaryImage, binaryImage, 30 );

    for (int i = 0; i < binaryImage.rows; i++)
    {
        for (int j = 0; j < binaryImage.cols; j++)
        {
            if ( result == 0 & i>(a*j+b)){
                int color = binaryImage.at<uchar>(i,j);
                if (color != 0)
                {
                    Vec3b intensity = colorImage.at<Vec3b>(i , j);
                    sumR += intensity[2];
                    sumG += intensity[1];
                    sumB += intensity[0];

                    // captureFrame.at<Vec3b>(i , j )[0]=0;
                    // captureFrame.at<Vec3b>(i , j )[1]=255;
                    // captureFrame.at<Vec3b>(i , j )[2]=0;

                    objPixels ++;
                }
            }

            else if (result == 1 & i<(a*j+b)){
                int color = binaryImage.at<uchar>(i,j);
                if (color != 0)
                {
                    Vec3b intensity = colorImage.at<Vec3b>(i , j);
                    sumR += intensity[2];
                    sumG += intensity[1];
                    sumB += intensity[0];

                    // captureFrame.at<Vec3b>(i , j )[0]=0;
                    // captureFrame.at<Vec3b>(i , j )[1]=255;
                    // captureFrame.at<Vec3b>(i , j )[2]=0;

                    objPixels ++;
                }
            }

        }
    }

    if (objPixels != 0 )
    {
        R = (sumR / objPixels);
        G = (sumG / objPixels);
        B = (sumB / objPixels);
    }
}

void GradePineapple::predictGrade(int R, int G, int B, string &grade)
{
    /* interpolasi RGB
    -
    */
    int minR=44, minG=37, minB=23; //minR = 44
    int maxR=70, maxG=56, maxB=41; //max R= 70

    //interpolasi
    R =  (R-minR) * 225/(maxR-minR);
    G = (G-minG) * 225/(maxG-minG);
    B = (B-minB) * 225/(maxB-minB);

    //inisiasi bobot + Bias ANN
    int offsetInput[3] = {0, 12, 0};
    float gainInput[3] ={0.00888888889, 0.00938967136, 0.00888888889};
    int minInput= -1;

    float biasLayer1[3] = {-1.0078863051281, -1.2386389137523, 0.0145354807791};
    float bobotLayer1[3][3] =	{
                                    {4.021315087768950 , -2.152069472915780 , 0.312019373396070 },
                                    {-0.209902342097445 , 2.394709790660760 , 0.338073585445509 },
                                    {2.419569233022990 , 2.911227649837230 , 0.521151962447184 }
                                };

    float biasOutput = -0.4214106857215100;
    float bobotOutput[3] = {0.77633662851335800 , -0.51256744642715900 , 0.33176819264420700 };

    int offsetOutput = 0;
    float gainOutput = 0.666666;
    int minOutput = -1;

    float persiapanR, persiapanG, persiapanB;
    float N1, N2, N3;
    float output, outputINV;
    int pembulatan;

    //persiapan
    persiapanR = (R-offsetInput[0])*gainInput[0] + minInput;
    persiapanG = (G-offsetInput[1])*gainInput[1] + minInput;
    persiapanB = (B-offsetInput[2])*gainInput[2] + minInput;

    //lapisan 1
    N1 = 2/(1 + exp (-2 * (persiapanR * bobotLayer1[0][0] + persiapanG * bobotLayer1[0][1] + persiapanB * bobotLayer1[0][2] + biasLayer1[0] ))) - 1;
    N2 = 2/(1 + exp (-2 * (persiapanR * bobotLayer1[1][0] + persiapanG * bobotLayer1[1][1] + persiapanB * bobotLayer1[1][2] + biasLayer1[1] ))) - 1;
    N3 = 2/(1 + exp (-2 * (persiapanR * bobotLayer1[2][0] + persiapanG * bobotLayer1[2][1] + persiapanB * bobotLayer1[2][2] + biasLayer1[2] ))) - 1;


    //output
    output = N1 * bobotOutput[0] + N2 * bobotOutput[1] + N3 * bobotOutput[2] + biasOutput;

    //inv output
    outputINV = (output - minOutput) / gainOutput + offsetOutput;

    //pembulatan
    // pembulatan = (int) outputINV;

    if (outputINV < 0.3){
        grade = "A";
    }
    else if (outputINV >= 0.3 & outputINV < 1.5){
        grade = "B";
    }
    else if (outputINV >= 1.5 & outputINV < 2.5){
        grade = "C";
    }
    else if (outputINV >= 2.5)    {
        grade = "D";
    }

}
