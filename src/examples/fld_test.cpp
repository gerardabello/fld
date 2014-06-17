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


void testOmniPose(CFld *test){
/*
    string dataDir = "../dat/frames/data";
    vector<VideoCapture> vcv;

    getVideoCaptureVector(dataDir, vcv);

    Mat frameM1, frameM2;

    int frame1 = 786;
    int frame2 = 800;


    Mat sframe;

    while(1)
    {	

        //bool bSuccess = readOmniFrame(vcv, vframe); // read a new frame from video
        bool bSuccess = combineImages(vcv, sframe);

        if (!bSuccess) //if not success, break loop
        {
            cout << "Cannot read the frame from video file" << endl;
            break;
        }
        
        if(i==frame1 || i==frame2){

            cv::resize(sframe, sframe, Size(), 0.5, 0.5);

            sframe.copyTo(



        }


    }
*/
}


void testPose(CFld *test){


    string imgDir = "../dat/sfm/";


    Mat K = (Mat_<double>(3,3) << 919.8266666666666, 0.0, 506.89666666666665, 0.0, 921.8365624999999, 335.7672021484375, 0, 0, 1 );


    Mat img1, img2;

    img1 = imread(imgDir+"frame0632.jpg" , IMREAD_COLOR);
    img2 = imread(imgDir+"frame0637.jpg" , IMREAD_COLOR);


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


class mylistener : public FmListener{

    public:

        virtual void newPose (float dir, float a){
            cout << "pose: d:" << dir << " a: " << a << endl;
        }

};



void testFabmap(CFld *test){
    string dataDir = "../dat/frames/test_2";
    string trainDir = "../dat/frames/data";
    string mainDir = "../dat/frames";




    test->setListener(new mylistener);

    Mat vocab;
    Mat vocab1, vocab2, vocab3, vocab4, vocab5;
    bool vocabdatasaved = openMatFileIfExists(mainDir + string("/vocab"),vocab); 
    if(!vocabdatasaved){
        cout << "Generating vocab" << endl;

        VideoCapture cap_train_voca1(trainDir + string("/fc/frame%4d.jpg")); // open the video file for reading
        VideoCapture cap_train_voca2(trainDir + string("/rr/frame%4d.jpg")); // open the video file for reading
        VideoCapture cap_train_voca3(trainDir + string("/rl/frame%4d.jpg")); // open the video file for reading
        VideoCapture cap_train_voca4(trainDir + string("/fr/frame%4d.jpg")); // open the video file for reading
        VideoCapture cap_train_voca5(trainDir + string("/fl/frame%4d.jpg")); // open the video file for reading


        vocab1 = test->addVocabVideo(cap_train_voca1);
        vocab2 = test->addVocabVideo(cap_train_voca2);
        vocab3 = test->addVocabVideo(cap_train_voca3);
        vocab4 = test->addVocabVideo(cap_train_voca4);
        vocab5 = test->addVocabVideo(cap_train_voca5);

        vocab.push_back(vocab1);
        vocab.push_back(vocab2);
        vocab.push_back(vocab3);
        vocab.push_back(vocab4);
        vocab.push_back(vocab5);

        test->addVocabulary(vocab);

        saveMatFile(mainDir + string("/vocab"), vocab);
    }else{

        cout << "Vocab loaded from file" << endl;
        test->addVocabulary(vocab);
    }

    Mat train_data;

    bool traindatasaved = openMatFileIfExists(mainDir + string("/train_data"),train_data); 
    //bool traindatasaved = false;
    if(!traindatasaved){
        cout << "Generating train data" << endl;
        vector<VideoCapture> vcv_train;

        getVideoCaptureVector(trainDir, vcv_train);

        //VideoCapture cap_train(dataDir + string("stlucia_train.avi")); // open the video file for reading

        //train_data = test->addTrainVideo(vcv_train);

        Mat temp_train_data;
        int i_train = 0;
        int steps_train = 40;
        while(1)
        {
            i_train++;


            //vector<Mat> vframe;

            Mat sframe_train;

            //bool bSuccess = readOmniFrame(vcv, vframe); // read a new frame from video
            bool bSuccess_train = combineImages(vcv_train, sframe_train);

            if (!bSuccess_train) //if not success, break loop
            {
                cout << "Cannot read the frame from video file" << endl;
                break;
            }

            cv::resize(sframe_train, sframe_train, Size(), 0.5, 0.5);



            if(i_train%steps_train==0){

                cout << "Train: generating frame " << i_train << endl;
                /*
                   imshow("MyVideo", vframe.at(1)); //show the frame in "MyVideo" window

                   if(waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
                   {
                   cout << "esc key is pressed by user" << endl;
                   break; 
                   }
                   */

                temp_train_data = test->genFrameData(sframe_train);
                train_data.push_back(temp_train_data);

            }

        }


        saveMatFile(mainDir + string("/train_data"), train_data);
    }else{

        cout << "Train loaded from file" << endl;
        cout << "Adding train data ..." << endl;
        test->addTrainData(train_data);
    }




    vector<VideoCapture> vcv;

    getVideoCaptureVector(dataDir, vcv);


    namedWindow( "Test Data", WINDOW_AUTOSIZE ); // Create a window for display.

    cout << "Initilizing FabMap stream" << endl;
    int i = 0;
    int steps = 10;
    while(1)
    {
        i++;

        //vector<Mat> vframe;

        Mat sframe;

        //bool bSuccess = readOmniFrame(vcv, vframe); // read a new frame from video
        bool bSuccess = combineImages(vcv, sframe);

        if (!bSuccess) //if not success, break loop
        {
            cout << "Cannot read the frame from video file" << endl;
            break;
        }

        //cv::resize(sframe, sframe, Size(), 0.5, 0.5);

        if((i+3)%steps==0){


            if(waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
            {
                cout << "esc key is pressed by user" << endl;
                break; 
            }

            cout << "FabMap: Adding Frame " << i << endl;
            test->addFrame(sframe);

            cv::resize(sframe, sframe, Size(), 0.2, 0.2);
            imshow("Test Data", sframe); //show the frame in "MyVideo" window

            Mat result;
            result = test->getMatrix();

            imshow("Confusion Matrix", result);
        }

    }




    /*
       cout << "FabMap: saving results "  << endl;
       saveMatFile(mainDir + string("/result"), result);
       */


    waitKey();


}


void testMyPose(CFld *test){

    string trainDir = "../dat/frames/train_copy";

    Mat img1, img2;

    /*
       getOnePano(trainDir, "0687", img1);
       getOnePano(trainDir, "0700", img2);
       */


    getOnePano(trainDir, "0050", img1);
    getOnePano(trainDir, "0092", img2);


    float scale = 0.7;

    cv::resize(img1, img1, Size(), scale, scale);
    cv::resize(img2, img2, Size(), scale, scale);

    float angle, direction;
    int features;

    test->findOmniPose(img1,img2, angle, direction, features);

}



int main(int argc, char *argv[])
{

    cout << endl << "## Fld Test ##" << endl << endl;


    CFld *test;

    test = new CFld();

    //testPose(test);
    //testStrech(test);
    //testSfM();
    testFabmap(test);
    //testMyPose(test);







    cout << endl << "## End ##" << endl << endl;
    return 0;
}



