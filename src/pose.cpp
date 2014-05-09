#include "fld.h"




//L'ordenació es fa amb els punts de la primera imatge. No ha de diferir molt de els punts de la segona.
bool sortByX(const pair<Point2f, Point2f>& pt1, const pair<Point2f, Point2f>& pt2) { return pt1.first.x < pt2.first.x; }


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

    for (int i = 0; i < p_matches.size(); ++i)
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
    float a1, a2, dif;


    //intermidiate memory for points
    Point2f pa_1, pa_2, pb_1, pb_2;

    //result
    vector< pair < pair <float,float> , float > > angleDifs;

    //Calculating aproximately the number of keypoits that should be in the range of pose_PairMaxAngle,
    //  I decide how many pairs back I search for every pair.
    int lookBack = (int)((pose_PairMaxAngle/360.0)*ptsPair.size());


    cout << "Points: " << ptsPair.size() << endl;

    for (int i = 0; i < ptsPair.size(); ++i)
    {
        pa_1 = ptsPair.at(i).first;
        pa_2 = ptsPair.at(i).second;

        /*cout << " -- new point -- " << endl;
        cout << "point a: " << "from 1: " << pa_1 << " - from 2: " << pa_2 << endl;
        */

        for (int j = 0; j < ptsPair.size(); ++j)
            //for (int j = (i>=lookBack? i-lookBack : 0); j < ptsPair.size()-i; ++j)
        {

            if(i==j) continue;



            pb_1 = ptsPair.at(j).first;
            pb_2 = ptsPair.at(j).second;

            //cout << "    point b: " << "from 1: " << pb_1 << " - from 2: " << pb_2 << endl;

            a1 = abs( pa_1.x - pb_1.x );
            a2 = abs( pa_2.x - pb_2.x );


            a1 = a1/imw*360;
            a2 = a2/imw*360;

            if(a1 > pose_PairMaxAngle || a2 > pose_PairMaxAngle) continue;


            dif = abs(a1-a2);


            angleDifs.push_back( 
                    pair < pair <float,float> , float > (
                        pair <float,float>(
                            abs( pa_1.x + pb_1.x )/2/imw*360,
                            abs( pa_2.x + pb_2.x )/2/imw*360
                            ),
                        dif
                        )
                    );


        }

    }

    sort(angleDifs.begin(), angleDifs.end(), sortByMeanX);


    float mean = 0;
    int count = 0;
    int last = 0;
    int current;
    float x, d;

    int divisions = 10;
    int difs[36] = {0};

    ofstream myfile;
    myfile.open ("data_pose");


    for (int i = 0; i < angleDifs.size(); ++i)
    {

        x = angleDifs.at(i).first.first;
        d = angleDifs.at(i).second;

        myfile << x << ", " << d << endl;

        current = (int)(x / divisions);


        if(current != last){
            if(count > 0) difs[last] = difs[last]/count;


            //cout << difs[last] << endl;

            count = 0;
            last = current;

        }

        difs[current] += d;
        count++;

    }


    myfile.close();




    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches);

    //cv::resize(img_matches, img_matches, Size(), 0.5, 0.5);

    imshow("matches", img_matches);
    if(waitKey(30000) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
    {
        cout << "esc key is pressed by user" << endl;
    }






}


