#pragma once

#include <iostream>
#include <fstream>
#include <math.h>

#include "opencv2/stitching/stitcher.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;
using namespace std;


//#include "mocv/FindCameraMatrices.h"
//


//Listener

class FmListener
{
    public:
    virtual void newPose (float dir, float a, int match) = 0;
};

class PanoListener
{
    public:
    virtual void newFrame (Mat img) = 0;
};


class CFld
{

    /*
     * Parameters
     */

    //Maximum angle between 2 pairs of keypoints to calculate the pose.
    const static int pose_PairMaxAngle = 150;

    //Ratio between the best match and the second best match to consider it a good match
    constexpr static float pose_GoodMatchesRatio = 0.7;
    constexpr static int min_features = 60;


    public:
        CFld();
        ~CFld();

        /**
         * Proceses de training image array and adds it to the fabmap algorithm.
         *
         * @param[in] imgs Vector of images to process
         * @return The procesed data from the images. Useful for saving it and pass
         * it as the argument in addTrainData so you dont have to process it again.
         */
        Mat addTrainImgVec(vector<Mat>& imgs);

        /**
         * Proceses de training video and adds it to the fabmap algorithm.
         *
         * @param[in] vc VideoCapture to process
         * @return The procesed data from the video. Useful for saving it and pass
         * it as the argument in addTrainData so you dont have to process it again.
         */
        Mat addTrainVideo(VideoCapture& vc);

        /**
         * Adds the already procesed training data to the fabmap algprithm.
         *
         * @param[in] trainData Data to add.
         */
        void addTrainData(Mat trainData);

        /**
         * Proceses de vocabulary image array and adds it to the fabmap algorithm.
         * Vocabulary takes a long time to process.
         *
         * @param[in] imgs Vector of images to process
         * @return The procesed data from the images. Useful for saving it and pass
         * it as the argument in addVocabulary so you dont have to process it again.
         */
        Mat addVocabImgVec(vector<Mat>& imgs);

        /**
         * Proceses de vocabulary video and adds it to the fabmap algorithm.
         * Vocabulary takes a long time to process.
         *
         * @param[in] vc VideoCapture to process
         * @return The procesed data from the video. Useful for saving it and pass
         * it as the argument in addVocabulary so you dont have to process it again.
         */
        Mat addVocabVideo(VideoCapture& vc);

        /**
         * Adds the already procesed vocabulary to the fabmap algprithm.
         *
         * @param[in] vocabulary Data to add.
         */
        void addVocabulary(Mat vocabulary);

        /**
         * Adds and processes a new frame.
         *
         * @param[in] frame Image of the new frame.
         */
        void addFrame(Mat frame);


        Mat genFrameData(Mat frame);

        /**
         * Returns the matrix in the scale of 0-255 of the probability of a pair of images beeing of the same place
         *
         * @return The matrix
         */
        Mat getMatrix();

        Mat createPano(vector<Mat> &imgs, bool rotate);

        bool geometricCheck( Mat &img1, Mat &img2);

        void findOmniPose(Mat& img1, Mat& img2, float& out_angle, float& out_dir, int& num_features);

        void setListener(FmListener* l);
        void setPanoListener(PanoListener* l);

        //private:
        //

        FmListener* listener;
        PanoListener* plistener;

        float consider_match;
        float maxSigma;

        //Margin in frames thats the minimum to consider to matches to be from diferent places.
        static const int same_place_margin = 5;


        Ptr<FeatureDetector> detector;
        Ptr<DescriptorExtractor> extractor;
        Ptr<DescriptorMatcher> matcher;

        BOWImgDescriptorExtractor* bide;

        Ptr<of2::FabMap> fabmap;

        Mat tree;
        Mat trainData;
        Mat vocab;

        vector<Mat> past_images;


        vector<vector<of2::IMatch> > matches;

        int iframe;


        int iniFabMap();

        bool genData(const Ptr<FeatureDetector> &detector, BOWImgDescriptorExtractor *bide, const Mat &frame, Mat &data);
        bool genDataVideo(const Ptr<FeatureDetector> &detector, BOWImgDescriptorExtractor *bide, VideoCapture &cap, Mat &data, int steps);

        bool genVocabData(const Ptr<FeatureDetector> &detector, const Ptr<DescriptorExtractor> &extractor, VideoCapture &cap, Mat &data, int steps);
        Mat genVocab(const Ptr<FeatureDetector> &detector, const Ptr<DescriptorExtractor> &extractor, VideoCapture &cap, int steps, float radius);


        void checkMatch(vector<of2::IMatch> & v);

        vector<Mat>* rotate_vector(vector<Mat> &imgs, int angle);


        /*
         *@brief rotate image by factor of 90 degrees
         *
         *@param source : input image
         *@param dst : output image
         *@param angle : factor of 90, even it is not factor of 90, the angle
         * will be mapped to the range of [-360, 360].
         * {angle = 90n; n = {-4, -3, -2, -1, 0, 1, 2, 3, 4} }
         * if angle bigger than 360 or smaller than -360, the angle will
         * be map to -360 ~ 360.
         * mapping rule is : angle = ((angle / 90) % 4) * 90;
         *
         * ex : 89 will map to 0, 98 to 90, 179 to 90, 270 to 3, 360 to 0.
         *
         */
        void rotate_image(cv::Mat &src, cv::Mat &dst, int angle);

        float slope_kpts(KeyPoint kpt1, KeyPoint kpt2);

        float percentilInlinersKpts(vector<KeyPoint> &keyPointsRef, vector<KeyPoint> &keyPoints, vector<DMatch> &all_matches);

        void goodMatches(vector<DMatch> &all_matches, vector<DMatch> &good_matches);


        float compareHistogram(Mat &img1, Mat &img2);

        void calcPose(const Mat &K, const Mat &img1, const Mat &img2, Mat &Pout);

        void FindCameraMatrices(const Mat& K,
                const vector<KeyPoint>& kpts1,
                const vector<KeyPoint>& kpts2,
                const vector<DMatch> p_matches,
                const Mat &img1,
                const Mat &img2,
                Matx34d& P,
                Matx34d& P1
                );

        bool CheckCoherentRotation(cv::Mat_<double>& R);


        void KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps);




};



