#include "fld.h"




//L'ordenació es fa amb els punts de la primera imatge. No ha de diferir molt de els punts de la segona.
bool sortByX(const pair<Point2f, Point2f>& pt1, const pair<Point2f, Point2f>& pt2) { return pt1.first.x < pt2.first.x; }

//L'ordenació es fa amb els punts de la primera imatge. No ha de diferir molt de els punts de la segona.
bool sortByAngleDif(const pair<float,float>& pt1, const pair<float,float>& pt2) { return pt1.second < pt2.second; }


//L'ordenació es fa amb els punts de la primera imatge. No ha de diferir molt de els punts de la segona.
bool sortByMeanX(pair < pair <float,float> , float > pt1 , pair < pair <float,float> , float > pt2) { return pt1.first.first < pt2.first.first; }

void CFld::findOmniPose(Mat& img1, Mat& img2){

    /*
     * Get features
     */

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



    cout << "Matches: " << p_matches.size() << endl;
    vector<cv::DMatch> good_matches;

    for (unsigned int i = 0; i < p_matches.size(); ++i)
    {
        //cout << "comparison : " << p_matches[i][0].distance << " < " << pose_GoodMatchesRatio * p_matches[i][1].distance << endl;
        if (p_matches[i][0].distance < pose_GoodMatchesRatio * p_matches[i][1].distance)
        {
            good_matches.push_back(p_matches[i][0]);
        }
    }

    cout << "Good Matches: " << good_matches.size() << endl;

    vector< pair<Point2f, Point2f> > ptsPair;
    for( unsigned int i = 0; i < good_matches.size(); i++ )
    {
        ptsPair.push_back( 
                pair<Point2f, Point2f> (    keypoints1[ good_matches[i].queryIdx ].pt,
                    keypoints2[ good_matches[i].trainIdx ].pt));

    }





    /*
     * Sort them
     */

    sort(ptsPair.begin(), ptsPair.end(), sortByX);

    /*
     * Find the relative angles of pairs. Pairs should be relatively close;
     */

    //with of the image
    float imw = img1.cols;

    //angles

    Point2f p1,p2;



    //result
    vector<float> angle1, angle2;


    cout << "Points: " << ptsPair.size() << endl;

    for (unsigned int i = 0; i < ptsPair.size(); ++i)
    {
        p1 = ptsPair.at(i).first ;
        p2 = ptsPair.at(i).second;


        angle1.push_back(p1.x /imw*360);
        angle2.push_back(p2.x /imw*360);

    }


    float mean = 0;
    float a1, a2;
    int delta = 2;
    assert(360%delta == 0);
    int ds = 360/delta;

    int minAngle;
    float meanMinAngle = 1000;

    for (int i = 0; i < ds; ++i)
    {


        for (unsigned int j = 0; j < angle1.size(); ++j){


            a2 = angle2.at(j);
            a1 = angle1.at(j)+(i*delta);

            if(a1>360)a1-=360;

            mean += abs(a1-a2);

        }


        mean = mean/angle1.size();

        //cout << mean << endl;

        if(mean < meanMinAngle){
            minAngle = i*delta;
            meanMinAngle = mean;
        }
        mean = 0;

    }

    cout << "Angle:  " << minAngle << endl;


    float minDirAngle;
    float meanMinDirAngle;
    float angleDif;

    vector< pair<float, float> > posAngle;


    for (unsigned int i = 0; i < angle1.size(); ++i){


        a2 = angle2.at(i);
        a1 = angle1.at(i)+(minAngle);
        if(a1>360)a1-=360;

        angleDif = abs(a1-a2);

        posAngle.push_back(pair<float,float>(angle1.at(i),angleDif));

    }


    sort(posAngle.begin(), posAngle.end(), sortByAngleDif);

    // DEBUG
    for (unsigned int i = 0; i < posAngle.size(); ++i){

        cout << posAngle.at(i).second << "-" << posAngle.at(i).first << endl;

    }
    
    mean = 0;

    /*
    float weightS = 0;
    float maxDif = (posAngle.at(posAngle.size()-1).second)/30;

    //cout << "maxDif: " <<  maxDif << endl;
    int c = 0;


    //calcular mean
    //for (unsigned int c = 0; c < posAngle.size()/8; ++c){
    while(1){


        if( posAngle.at(c).second > maxDif) break;

        // TODO treure outliers amb RANSAC
        mean += posAngle.at(c).first*(1/posAngle.at(c).second);
        weightS += 1/posAngle.at(c).second;

        c++;

    }

    float direction = mean/weightS;

    cout << "Direction: " << direction << endl;

*/
    delta = 60;
    assert(360%delta == 0);
    ds = 360/delta;
    
    int weightS = 0;

    for (int i = 0; i < ds; ++i)
    {


        mean = 0;
        for (unsigned int j = 0; j < posAngle.size(); ++j){
            if(posAngle.at(j).first >= i*delta && posAngle.at(j).first < (i+1)*delta){
                mean += posAngle.at(j).second;
                weightS++;
            }
        }


        mean = mean/weightS;
        weightS=0;


        cout << delta*i << " : " << mean << endl;


    }




    //show matches


    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches,Scalar::all(-1), Scalar::all(-1), vector<char>(), 2);
    imwrite( "./result.jpg", img_matches );
    //cv::resize(img_matches, img_matches, Size(), 0.5, 0.5);

    imshow("matches", img_matches);
    if(waitKey(30000) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
    {
        cout << "esc key is pressed by user" << endl;
    }




}


