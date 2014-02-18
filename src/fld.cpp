#include "fld.h"

CFld::CFld(){
    detector = Ptr<FeatureDetector>(
            new DynamicAdaptedFeatureDetector(
                AdjusterAdapter::create("STAR"), 130, 150, 5));
    extractor = Ptr<DescriptorExtractor>(
            new SurfDescriptorExtractor(1000, 4, 2, false, true));
    matcher = DescriptorMatcher::create("FlannBased");


    bide = new BOWImgDescriptorExtractor(extractor, matcher);


    iframe = 0;
}

CFld::~CFld(){
    delete bide;
}

void CFld::addVocabulary(Mat vocabulary){
    bide->setVocabulary(vocabulary);
}

void CFld::addTrainData(Mat td){
    trainData = td;
    of2::ChowLiuTree treeBuilder;
    treeBuilder.add(trainData);
    tree = treeBuilder.make();

    iniFabMap();
}


int CFld::iniFabMap(){
    fabmap.reset(new of2::FabMap2(tree, 0.39, 0, of2::FabMap::SAMPLED |
                of2::FabMap::CHOW_LIU));
    fabmap->addTraining(trainData);

    return 0;
}


void CFld::addFrame(Mat frame){
    iframe++;

    Mat fData;

    genData(detector,bide,frame,fData);

    vector<of2::IMatch> imatches;
    fabmap->compare(fData, imatches, true);

    matches.push_back(imatches);
}


Mat CFld::getMatrix(){
    Mat matrix;

    matrix = Mat::zeros(matches.size()+1, matches.size()+1, CV_8UC1);

    int j = 0;

    vector< vector<of2::IMatch> >::iterator le;
    vector<of2::IMatch>::iterator l;

    for(le = matches.begin(); le != matches.end(); le++) {
        j++;
        for(l = le->begin(); l != le->end(); l++) {
            if(l->imgIdx < 0) {
                matrix.at<char>(l->queryIdx+j, l->queryIdx) =
                    (char)(l->match*255);

            } else {
                matrix.at<char>(l->queryIdx+j, l->imgIdx) =
                    (char)(l->match*255);
            }
        }
    }

    return matrix;
}

/*Mat CFld::addTrainImgVec(vector<Mat>& imgs){


  }
  */

Mat CFld::addTrainVideo(VideoCapture& vc){
    Mat td;
    CFld::genDataVideo(detector,bide,vc,td,20);
    addTrainData(td);
    return td;
}

/*
   Mat CFld::addVocabImgVec(vector<Mat>& imgs){


   }
   */

Mat CFld::addVocabVideo(VideoCapture& vc){
    Mat v;
    v = CFld::genVocab(detector, extractor, vc, 20, 0.45);
    addVocabulary(v);
    return v;
}



bool CFld::genData(const Ptr<FeatureDetector> &detector, BOWImgDescriptorExtractor *bide, const Mat &frame, Mat &data){
    Mat bow;
    vector<KeyPoint> kpts;

    detector->detect(frame, kpts);

    bide->compute(frame, kpts, bow);

    data.push_back(bow);

    return true;
}


bool CFld::genDataVideo(const Ptr<FeatureDetector> &detector, BOWImgDescriptorExtractor *bide, VideoCapture &cap, Mat &data, int steps){
    if ( !data.empty() )  // if not success, exit program
    {
        cout << "data is not empty" << endl;
        return false;
    }

    if ( !cap.isOpened() )  // if not success, exit program
    {
        cout << "Cannot open the video file" << endl;
        return false;
    }

    int i = 0;

    Mat vframe;

    while(1)
    {
        i++;


        bool bSuccess = cap.read(vframe); // read a new frame from video

        if (!bSuccess) //if not success, break loop
        {
            cout << "Cannot read the frame from video file" << endl;
            break;
        }

        if(i%steps==0){

            /*imshow("MyVideo", vframe); //show the frame in "MyVideo" window

              if(waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
              {
              cout << "esc key is pressed by user" << endl;
              break; 
              }

*/
            genData(detector,bide,vframe,data);

        }

    }

    return true;

}


bool CFld::genVocabData(const Ptr<FeatureDetector> &detector, const Ptr<DescriptorExtractor> &extractor, VideoCapture &cap, Mat &data, int steps){
    if ( !data.empty() )  // if not success, exit program
    {
        cout << "data is not empty" << endl;
        return false;
    }

    if ( !cap.isOpened() )  // if not success, exit program
    {
        cout << "Cannot open the video file" << endl;
        return false;
    }

    int i = 0;

    Mat vframe, descs;

    vector<KeyPoint> kpts;

    while(1)
    {
        i++;


        bool bSuccess = cap.read(vframe); // read a new frame from video

        if (!bSuccess) //if not success, break loop
        {
            cout << "Cannot read the frame from video file" << endl;
            break;
        }

        if(i%steps==0){

            /*imshow("MyVideo", vframe); //show the frame in "MyVideo" window

              if(waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
              {
              cout << "esc key is pressed by user" << endl;
              break; 
              }

*/
            //detect & extract features
            detector->detect(vframe, kpts);
            extractor->compute(vframe, kpts, descs);

            //add all descriptors to the training data
            data.push_back(descs);


        }

    }

    return true;

}

Mat CFld::genVocab(const Ptr<FeatureDetector> &detector, const Ptr<DescriptorExtractor> &extractor, VideoCapture &cap, int steps, float radius){
    Mat vocabTrainData;


    genVocabData(detector,extractor, cap, vocabTrainData,steps);

    cout << "Performing clustering" << std::endl;

    //uses Modified Sequential Clustering to train a vocabulary
    of2::BOWMSCTrainer trainer(radius);
    trainer.add(vocabTrainData);
    return trainer.cluster();
}



Mat CFld::createPano(vector<Mat> &imgs, bool rotate){

    Mat pano;
    Stitcher::Status status;

    if(rotate){
        vector<Mat> *list;
        list = rotate_vector(imgs,90);


        vector<Mat>::iterator l;
        Mat temp;



        namedWindow( "Pano2", WINDOW_AUTOSIZE ); // Create a window for display.
        for(l = list->begin(); l != list->end(); l++) {

            imshow( "Pano2", *l);                // Show our image inside it.
            waitKey(500);

        }



        Stitcher stitcher = Stitcher::createDefault();
        status = stitcher.stitch(*list, pano);


        delete list;
    }else{
        Stitcher stitcher = Stitcher::createDefault();
        status = stitcher.stitch(imgs, pano);

    }


    if (status != Stitcher::OK)
    {
        cout << "Can't stitch images, error code = " << int(status) << endl;
    }


    return pano;
}

vector<Mat>* CFld::rotate_vector(vector<Mat> &imgs, int angle){
    vector<Mat>* rotated = new vector<Mat>();

    vector<Mat>::iterator l;
    Mat temp;
    for(l = imgs.begin(); l != imgs.end(); l++) {
        rotate_image(*l,temp,angle);
        rotated->push_back(temp);
    }

    return rotated;
}




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

void CFld::rotate_image(cv::Mat &src, cv::Mat &dst, int angle)
{   
    if(src.data != dst.data){
        src.copyTo(dst);
    }

    angle = ((angle / 90) % 4) * 90;

    //0 : flip vertical; 1 flip horizontal
    bool const flip_horizontal_or_vertical = angle > 0 ? 1 : 0;
    int const number = std::abs(angle / 90);          

    for(int i = 0; i != number; ++i){
        cv::transpose(dst, dst);
        cv::flip(dst, dst, flip_horizontal_or_vertical);
    }
}



