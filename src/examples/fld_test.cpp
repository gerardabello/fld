#include "../fld.h"

#include <iostream>
#include <string.h>

#include "sfm_test.cpp"


bool openMatFileIfExists(string path, Mat &data){
    std::ifstream file(path.c_str());
    if(file.is_open()) {
        cout << path << ": Data Already Present" <<
            std::endl;

        file.close();

        cv::FileStorage fs;    

        //load in vocab training data
        fs.open(path, cv::FileStorage::READ);
        fs["Data"] >> data;
        if (data.empty()) {
            std::cerr << path << ": Data is empty" <<
                std::endl;
            return false;
        }
        fs.release();

        return true;
    }

    return false;
}


bool saveMatFile(string path, const Mat &data){

    cv::FileStorage fs;    

    //save the vocabulary
    std::cout << "Saving Data" << std::endl;
    fs.open(path, cv::FileStorage::WRITE);
    fs << "Data" << data;
    fs.release();

    return true;
}


void testPose(CFld *test){


    string imgDir = "../dat/fountain/";


    Mat K = (Mat_<double>(3,3) << 919.8266666666666, 0.0, 506.89666666666665, 0.0, 921.8365624999999, 335.7672021484375, 0, 0, 1 );


    Mat img1, img2;

    img1 = imread(imgDir+"0001-small.png" , IMREAD_COLOR);
    img2 = imread(imgDir+"0005-small.png" , IMREAD_COLOR);


    namedWindow( "Pano", WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Pano", img1 );                // Show our image inside it.
    waitKey(500);

    imshow( "Pano", img2 );                // Show our image inside it.
    waitKey(500);

    Mat P;

    test->calcPose(K, img1, img2, P);


}

void testStrech(CFld *test){

    string imgDir = "../dat/";

    vector<Mat> images;
    images.push_back(imread(imgDir+"roofs1.jpg" , IMREAD_COLOR));
    images.push_back(imread(imgDir+"roofs2.jpg" , IMREAD_COLOR));
    /*images.push_back(imread(imgDir+"fr.jpg" , IMREAD_COLOR));
      images.push_back(imread(imgDir+"rr.jpg" , IMREAD_COLOR));
      images.push_back(imread(imgDir+"rl.jpg" , IMREAD_COLOR));*/


    Mat pano;
    pano = test->createPano(images, false);


    namedWindow( "Pano", WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Pano", pano );                // Show our image inside it.
    waitKey(500);
}

void testFabmap(CFld *test){
    string dataDir = "../dat/fabmap/";


    Mat vocab;
    bool vocabdatasaved = openMatFileIfExists(dataDir + string("vocab"),vocab); 
    if(!vocabdatasaved){
        cout << "Generating vocab" << endl;
        VideoCapture cap_train_voca(dataDir + string("stlucia_train.avi")); // open the video file for reading

        vocab = test->addVocabVideo(cap_train_voca);
        saveMatFile(dataDir + string("vocab"), vocab);
    }else{

        cout << "Vocab loaded from file" << endl;
        test->addVocabulary(vocab);
    }


    VideoCapture cap_train(dataDir + string("stlucia_train.avi")); // open the video file for reading

    test->addTrainVideo(cap_train);


    VideoCapture cap(dataDir + string("stlucia_test.avi")); // open the video file for reading

    int i = 0;
    int steps = 10;
    while(1)
    {
        i++;

        Mat vframe;

        bool bSuccess = cap.read(vframe); // read a new frame from video

        if (!bSuccess) //if not success, break loop
        {
            cout << "Cannot read the frame from video file" << endl;
            break;
        }

        if(i%steps==0){

            imshow("MyVideo", vframe); //show the frame in "MyVideo" window

            if(waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
            {
                cout << "esc key is pressed by user" << endl;
                break; 
            }

            test->addFrame(vframe);
            Mat result;
            result = test->getMatrix();

            imshow("Confusion Matrix", result);
        }

    }





    waitKey();


}





int main(int argc, char *argv[])
{

    cout << endl << "## Fld Test ##" << endl << endl;

    CFld *test;

    test = new CFld();

    //testPose(test);
    //testStrech(test);
    testSfM();




    cout << endl << "## End ##" << endl << endl;
    return 0;
}



