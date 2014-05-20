#include "fld.h"
#include "pose.cpp"


CFld::CFld(){
    detector = Ptr<FeatureDetector>(
            new DynamicAdaptedFeatureDetector(
                AdjusterAdapter::create("STAR"), 130, 150, 5));
    extractor = Ptr<DescriptorExtractor>(
            new SurfDescriptorExtractor(1000, 4, 2, false, true));
    matcher = DescriptorMatcher::create("FlannBased");


    bide = new BOWImgDescriptorExtractor(extractor, matcher);


    iframe = 0;



    //params
    consider_match = 0.9f;
    maxSigma = 0.55f;
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
    fabmap = new of2::FabMap2(tree, 0.39, 0, of2::FabMap::SAMPLED |
                of2::FabMap::CHOW_LIU);
    fabmap->addTraining(trainData);

    return 0;
}


void CFld::addFrame(Mat frame){

    past_images.push_back(frame);

    Mat fData;

    genData(detector,bide,frame,fData);

    vector<of2::IMatch> imatches;
    fabmap->compare(fData, imatches, true);

    checkMatch(imatches);

    matches.push_back(imatches);

    iframe++;
}

Mat CFld::genFrameData(Mat frame){

    Mat fData;

    genData(detector,bide,frame,fData);

    return fData;
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



void CFld::checkMatch(vector<of2::IMatch> & v  ){
    int i1,i2;
    bool ok;


    auto remover = remove_if(v.begin(), v.end(), [&](const of2::IMatch & o ) { 
            if(o.match > consider_match){
            //cout << o.queryIdx << " - " << o.imgIdx << endl;
                if(o.imgIdx < 0) {
                    i1 = o.queryIdx+iframe;
                    i2 = o.queryIdx;
                } else {
                    i1 = o.queryIdx+iframe;
                    i2 = o.imgIdx;
                }
            }else{
                return true;
            }


            if (abs(i1-i2)< same_place_margin) return true;
            //if (!geometricCheck( past_images.at(i1), past_images.at(i2))) return true;

            imwrite( "pano1.jpg", past_images.at(i1) );
            imwrite( "pano2.jpg", past_images.at(i2) );
    });




    v.erase( remover, v.end());

}


bool CFld::geometricCheck( Mat &img1, Mat &img2){

    Ptr<FeatureDetector> c_detector = FeatureDetector::create("ORB");
    Ptr<DescriptorExtractor> c_descriptor = DescriptorExtractor::create("BRIEF");
    Ptr<DescriptorMatcher> c_matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // detecting keypoints
    vector<KeyPoint> keypoints1, keypoints2;
    c_detector->detect(img1, keypoints1);
    c_detector->detect(img2, keypoints2);

    // computing descriptors
    Mat descriptors1, descriptors2;
    c_descriptor->compute(img1, keypoints1, descriptors1);
    c_descriptor->compute(img2, keypoints2, descriptors2);

    // matching descriptors
    vector<DMatch> c_matches;
    c_matcher->match(descriptors1, descriptors2, c_matches);


    percentilInlinersKpts(keypoints1, keypoints2, c_matches);
    compareHistogram(img1, img2);


    vector<float> angles;


    vector<DMatch>::iterator l;
    for(l = c_matches.begin(); l != c_matches.end(); l++) {
        //if(l->distance <= 15){
        //cout << l->distance << endl;
        angles.push_back(slope_kpts(keypoints1.at(l->queryIdx),keypoints2.at(l->trainIdx)));
        //}
    }

    if(angles.size()==0){
        angles.push_back(99);
    }

    vector<float>* test;
    test = &angles;

    double sum = accumulate(std::begin(*test), std::end(*test), 0.0);
    double m =  sum / test->size();

    double accum = 0.0;
    std::for_each (std::begin(*test), std::end(*test), [&](const double d) {
            accum += (d - m) * (d - m);
            });

    double stdev = sqrt(accum / (test->size()-1));

    cout << "Stdev: " << stdev << endl;

    cout << "-------" << endl;

    // drawing the results
    //namedWindow("matches", 1);
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, c_matches, img_matches);
    //imshow("matches", img_matches);
    waitKey(0);

    return stdev <= maxSigma;
}


float CFld::slope_kpts(KeyPoint kpt1, KeyPoint kpt2){
    float p1x = kpt1.pt.x;
    float p1y = kpt1.pt.y;
    float p2x = kpt2.pt.x;
    float p2y = kpt2.pt.y;


    return atan((p2y-p1y)/(p2x-p1x));


}

float CFld::percentilInlinersKpts(vector<KeyPoint> &keyPoints1, vector<KeyPoint> &keyPoints2, vector<DMatch> &all_matches){

    vector<DMatch> good_matches;

    //goodMatches(all_matches, good_matches);

    good_matches = all_matches;

    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for( unsigned int i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keyPoints1[ good_matches[i].queryIdx ].pt );
        scene.push_back( keyPoints2[ good_matches[i].trainIdx ].pt );
    }



    Mat mask;

    Mat H = findHomography( obj, scene, RANSAC , 10, mask);


    int size = mask.total();


    int ones;

    float a = sum(mask)[0];

    ones = a;


    cout << ones << " out of " << size << endl;

    return float(ones)/size;

}


void CFld::goodMatches(vector<DMatch> &all_matches, vector<DMatch> &good_matches){
    float min_dist, max_dist;

    vector<DMatch>::iterator l;
    for(l = all_matches.begin(); l != all_matches.end(); l++) {
        double dist = l->distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    for(l = all_matches.begin(); l != all_matches.end(); l++) {
        if( l->distance < 3*min_dist )
        { good_matches.push_back( *l); }
    }
}



float CFld::compareHistogram(Mat &img1, Mat &img2){

    Mat src_base, hsv_base;
    Mat src_test1, hsv_test1;

    src_base = img1;
    src_test1 = img2;

    /// Convert to HSV
    cvtColor( src_base, hsv_base, COLOR_RGB2HSV );
    cvtColor( src_test1, hsv_test1, COLOR_RGB2HSV );

    //hsv_half_down = hsv_base( Range( hsv_base.rows/2, hsv_base.rows - 1 ), Range( 0, hsv_base.cols - 1 ) );

    /// Using 30 bins for hue and 32 for saturation
    int h_bins = 50; int s_bins = 60;
    int histSize[] = { h_bins, s_bins };

    // hue varies from 0 to 256, saturation from 0 to 180
    float h_ranges[] = { 0, 256 };
    float s_ranges[] = { 0, 180 };

    const float* ranges[] = { h_ranges, s_ranges };

    // Use the o-th and 1-st channels
    int channels[] = { 0, 1 };

    /// Histograms
    MatND hist_base;
    MatND hist_half_down;
    MatND hist_test1;
    MatND hist_test2;

    /// Calculate the histograms for the HSV images
    calcHist( &hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false );
    normalize( hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat() );

    /*
       calcHist( &hsv_half_down, 1, channels, Mat(), hist_half_down, 2, histSize, ranges, true, false );
       normalize( hist_half_down, hist_half_down, 0, 1, NORM_MINMAX, -1, Mat() );
       */

    calcHist( &hsv_test1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false );
    normalize( hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat() );

    double base_test1;

    /// Apply the histogram comparison methods
    //double base_base = compareHist( hist_base, hist_base, compare_method );
    //double base_half = compareHist( hist_base, hist_half_down, compare_method );
    base_test1 = compareHist( hist_base, hist_test1, 0);


    cout << "hist: " << base_test1 << endl;

    return base_test1;

}



void CFld::calcPose(const Mat &K, const Mat &img1, const Mat &img2, Mat &Pout){



    //DETECTION ###########################
    Ptr<FeatureDetector> p_detector = FeatureDetector::create("SIFT");

    Ptr<DescriptorExtractor> p_descriptor = DescriptorExtractor::create("SIFT");
    Ptr<DescriptorMatcher> p_matcher = DescriptorMatcher::create("BruteForce");

    // detecting keypoints
    vector<KeyPoint> keypoints1, keypoints2;
    p_detector->detect(img1, keypoints1);
    p_detector->detect(img2, keypoints2);

    // computing descriptors
    Mat descriptors1, descriptors2;
    p_descriptor->compute(img1, keypoints1, descriptors1);
    p_descriptor->compute(img2, keypoints2, descriptors2);

    // matching descriptors
    std::vector<std::vector<cv::DMatch>> p_matches;
    p_matcher->knnMatch(descriptors1, descriptors2, p_matches, 2);  // Find two nearest matches



    vector<cv::DMatch> good_matches;

    for (int i = 0; i < p_matches.size(); ++i)
    {
        //cout << "comparison : " << p_matches[i][0].distance << " < " << pose_GoodMatchesRatio * p_matches[i][1].distance << endl;
        if (p_matches[i][0].distance < 0.8 * p_matches[i][1].distance)
        {
            good_matches.push_back(p_matches[i][0]);
        }
    }




    //FI DETECTION ###########################

    Matx34d P;
    Matx34d P1;

    FindCameraMatrices(K,
            keypoints1,
            keypoints2,
            good_matches,
            img1,
            img2,
            P,
            P1
            );


    Pout = Mat(P);
}


void CFld::FindCameraMatrices(const Mat& K,
        const vector<KeyPoint>& kpts1,
        const vector<KeyPoint>& kpts2,
        const vector<DMatch> p_matches,
        const Mat &img1,
        const Mat &img2,
        Matx34d& P,
        Matx34d& P1
        )
{
    //KEypoints to point array
    /*
       vector<Point2f> imgpts1, imgpts2;

       KeyPointsToPoints(kpts1, imgpts1);
       KeyPointsToPoints(kpts2, imgpts2);
       */
    Mat img1p, img2p;

    img1.copyTo(img1p);
    img2.copyTo(img2p);

    vector<Point2f> imgpts1,imgpts2;



    for( unsigned int i = 0; i < p_matches.size(); i++ )
    {
        imgpts1.push_back(kpts1[p_matches[i].queryIdx].pt);
        imgpts2.push_back(kpts2[p_matches[i].trainIdx].pt);
    }




    //Find camera matrices

    //Get Fundamental Matrix
    vector<uchar> status(imgpts1.size());

    double minVal,maxVal;
    cv::minMaxIdx(imgpts1,&minVal,&maxVal);
    Mat F = findFundamentalMat(imgpts1,imgpts2, FM_RANSAC, 0.006 * maxVal, 0.99, status);


    vector<Point2f> imgpts1_g ,imgpts2_g;
    for (unsigned int i = 0; i < status.size(); i++){
        if (status.at(i) == 1){
            imgpts1_g.push_back(imgpts1.at(i));
            imgpts2_g.push_back(imgpts2.at(i));
        }
    }





    cout << "number of in: " << imgpts1_g.size() << endl;


    //Essential matrix: compute then extract cameras [R|t]
    Mat_<double> E = K.t() * F * K; //according to HZ (9.12)




    /* draw the left points corresponding epipolar lines in right image */
    std::vector<cv::Vec3f> linesLeft;
    cv::computeCorrespondEpilines(
            imgpts1_g, // image points
            1,                      // in image 1 (can also be 2)
            F,            // F matrix
            linesLeft);             // vector of epipolar lines

    // for all epipolar lines
    for (vector<cv::Vec3f>::const_iterator it= linesLeft.begin(); it!=linesLeft.end(); ++it) {

        // draw the epipolar line between first and last column
        cv::line(img2p,cv::Point(0,-(*it)[2]/(*it)[1]),cv::Point(img2p.cols,-((*it)[2]+(*it)[0]*img2p.cols)/(*it)[1]),cv::Scalar(255,255,255));
    }



    // draw the left points corresponding epipolar lines in left image
    std::vector<cv::Vec3f> linesRight;
    cv::computeCorrespondEpilines(imgpts2_g,2,F,linesRight);
    for (vector<cv::Vec3f>::const_iterator it= linesRight.begin(); it!=linesRight.end(); ++it) {

        // draw the epipolar line between first and last column
        cv::line(img1p,cv::Point(0,-(*it)[2]/(*it)[1]), cv::Point(img1p.cols,-((*it)[2]+(*it)[0]*img1p.cols)/(*it)[1]), cv::Scalar(255,255,255));
    }

    // Display the images with points and epipolar lines
    cv::namedWindow("Right Image Epilines");
    cv::imshow("Right Image Epilines",img1p);
    cv::namedWindow("Left Image Epilines");
    cv::imshow("Left Image Epilines",img2p);

    waitKey(20000);


    //FILTER KEYPOINTS
    //TODO IMPLEMENT THIS
    /*
       for (unsigned int i=0; i<status.size(); i++) {
       if (status[i])
       {
       imgpts1_good.push_back(imgpts1_tmp[i]);
       imgpts2_good.push_back(imgpts2_tmp[i]);

       if (matches.size() <= 0) { //points already aligned...
       new_matches.push_back(DMatch(matches[i].queryIdx,matches[i].trainIdx,matches[i].distance));
       } else {
       new_matches.push_back(matches[i]);
       }

       }
       }	

       cout << matches.size() << " matches before, " << new_matches.size() << " new matches after Fundamental Matrix\n";
       */


    /*
    //decompose E to P' , HZ (9.19)
    SVD svd(E,SVD::MODIFY_A);
    Mat svd_u = svd.u;
    Mat svd_vt = svd.vt;
    Mat svd_w = svd.w;

    Matx33d W(0,-1,0,//HZ 9.13
    1,0,0,
    0,0,1);
    Mat_<double> R = svd_u * Mat(W) * svd_vt; //HZ 9.19
    Mat_<double> t = svd_u.col(2); //u3

    if (!CheckCoherentRotation(R)) {
    cout<<"resulting rotation is not coherent\n";
    P1 = 0;
    return;
    }
    */


    Mat_<double> R1(3,3);
    Mat_<double> R2(3,3);
    Mat_<double> t1(1,3);
    Mat_<double> t2(1,3);







    //decompose E to P' , HZ (9.19)
    {	
        if (!DecomposeEtoRandT_2(E,R1,R2,t1,t2)){
            cout<<"Error decomposing\n";
            P1 = 0;
            return;
        }

        if (!CheckCoherentRotation(R1)) {
            cout<<"resulting rotation is not coherent\n";
            P1 = 0;
            return;
        }


        if (!CheckCoherentRotation(R2)) {
            cout<<"resulting rotation is not coherent\n";
            P1 = 0;
            return;
        }

        P = Matx34d(R1(0,0),R1(0,1),R1(0,2),t1(0),
                R1(1,0),R1(1,1),R1(1,2),t1(1),
                R1(2,0),R1(2,1),R1(2,2),t1(2));

        cout << Mat(P) << endl;



        /*

           P = Matx34d(R1(0,0),R1(0,1),R1(0,2),t2(0),
           R1(1,0),R1(1,1),R1(1,2),t2(1),
           R1(2,0),R1(2,1),R1(2,2),t2(2));

           cout << Mat(P) << endl;


           P = Matx34d(R2(0,0),R2(0,1),R2(0,2),t1(0),
           R2(1,0),R2(1,1),R2(1,2),t1(1),
           R2(2,0),R2(2,1),R2(2,2),t1(2));

           cout << Mat(P) << endl;


           P = Matx34d(R2(0,0),R2(0,1),R2(0,2),t2(0),
           R2(1,0),R2(1,1),R2(1,2),t2(1),
           R2(2,0),R2(2,1),R2(2,2),t2(2));

           cout << Mat(P) << endl;

*/

    }

}


bool CFld::CheckCoherentRotation(cv::Mat_<double>& R) {
    if(fabsf(determinant(R))-1.0 > 1e-07) {
        cerr<<"det(R) != +-1.0, this is not a rotation matrix"<<endl;
        return false; 
    }
    return true;
}


void CFld::KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps) {
    ps.clear();
    for (unsigned int i=0; i<kps.size(); i++) ps.push_back(kps[i].pt);
}




