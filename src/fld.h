#ifndef _FLD_H
#define _FLD_H

#include <iostream>
#include <fstream>


#include "opencv2/contrib.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/nonfree.hpp"

using namespace cv;
using namespace std;


class CFld
{
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

    Mat getMatrix();



  private:
    Ptr<FeatureDetector> detector;
    Ptr<DescriptorExtractor> extractor;
    Ptr<DescriptorMatcher> matcher;

    BOWImgDescriptorExtractor* bide;

    Ptr<of2::FabMap> fabmap;

    Mat tree;
    Mat trainData;
    Mat vocab;


    vector<vector<of2::IMatch> > matches;

    int iframe;


    int iniFabMap();

    bool genData(const Ptr<FeatureDetector> &detector, BOWImgDescriptorExtractor &bide, const Mat &frame, Mat &data);
    bool genDataVideo(const Ptr<FeatureDetector> &detector, BOWImgDescriptorExtractor &bide, VideoCapture &cap, Mat &data, int steps);

    bool genVocabData(const Ptr<FeatureDetector> &detector, const Ptr<DescriptorExtractor> &extractor, VideoCapture &cap, Mat &data, int steps);
    Mat genVocab(const Ptr<FeatureDetector> &detector, const Ptr<DescriptorExtractor> &extractor, VideoCapture &cap, int steps, float radius);

};

#endif
