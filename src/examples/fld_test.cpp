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


bool readOmniFrame(vector<VideoCapture> vcv, vector<Mat>& iv){

    vector<VideoCapture>::iterator l;
    for(l = vcv.begin(); l != vcv.end(); l++) {
        Mat m;
        bool bSuccess = l->read(m);
        if(!bSuccess){
            return false;
        }

        iv.push_back(m);
    }

    return true;
}


Mat combineImages(vector<VideoCapture> vcv){

    vector<Mat> iv;
    readOmniFrame(vcv, iv);

    int h,w;
    int i;

    h = iv.at(0).rows;
    w = iv.at(0).cols;

    Mat combine(h, w*iv.size(), CV_8UC3);

    i = 0;
    vector<Mat>::iterator l;
    for(l = iv.begin(); l != iv.end(); l++) {

        Mat roi(combine, Rect(w*i, 0, w, h));
        l->copyTo(roi);
        i++;
    }

    imshow("SideBySide", combine); //show the frame in "MyVideo" window

    if(waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
    {
        cout << "esc key is pressed by user" << endl;
        return combine;
    }

    return combine;
}


bool getVideoCaptureVector(string path, vector<VideoCapture>& vcv){


    VideoCapture cap1(path + string("/fc/frame%4d.jpg")); // open the video file for reading
    VideoCapture cap2(path + string("/fr/frame%4d.jpg")); // open the video file for reading
    VideoCapture cap3(path + string("/rr/frame%4d.jpg")); // open the video file for reading
    VideoCapture cap4(path + string("/rl/frame%4d.jpg")); // open the video file for reading
    VideoCapture cap5(path + string("/rl/frame%4d.jpg")); // open the video file for reading


    vcv.push_back(cap1);
    vcv.push_back(cap2);
    vcv.push_back(cap3);
    vcv.push_back(cap4);
    vcv.push_back(cap5);

}



void testFabmap(CFld *test){
    string dataDir = "../dat/frames/data";
    string trainDir = "../dat/frames/train";
    string mainDir = "../dat/frames";


    Mat vocab;
    Mat vocab1, vocab2;
    bool vocabdatasaved = openMatFileIfExists(mainDir + string("/vocab"),vocab); 
    if(!vocabdatasaved){
        cout << "Generating vocab" << endl;
        VideoCapture cap_train_voca1(trainDir + string("/fc/frame%4d.jpg")); // open the video file for reading
        VideoCapture cap_train_voca2(trainDir + string("/rr/frame%4d.jpg")); // open the video file for reading

        vocab1 = test->addVocabVideo(cap_train_voca1);
        vocab2 = test->addVocabVideo(cap_train_voca2);

        vocab.push_back(vocab1);
        vocab.push_back(vocab2);

        test->addVocabulary(vocab);

        saveMatFile(mainDir + string("/vocab"), vocab);
        saveMatFile(mainDir + string("/vocab1"), vocab1);
        saveMatFile(mainDir + string("/vocab2"), vocab2);
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

        train_data = test->addTrainVideo(vcv_train);


        saveMatFile(mainDir + string("/train_data"), train_data);
    }else{

        cout << "Train loaded from file" << endl;
        cout << "Adding train data ..." << endl;
        test->addTrainData(train_data);
    }




    //VideoCapture cap(dataDir + string("stlucia_test.avi")); // open the video file for reading


    vector<VideoCapture> vcv;

    getVideoCaptureVector(dataDir, vcv);



    cout << "Initilizing FabMap stream" << endl;
    int i = 0;
    int steps = 20;
    while(1)
    {
        i++;

        vector<Mat> vframe;

        Mat sframe;

        //bool bSuccess = readOmniFrame(vcv, vframe); // read a new frame from video
        sframe = combineImages(vcv);
        bool bSuccess = true;

        if (!bSuccess) //if not success, break loop
        {
            cout << "Cannot read the frame from video file" << endl;
            break;
        }

        if(i%steps==0){

            /*
               imshow("MyVideo", vframe.at(1)); //show the frame in "MyVideo" window

               if(waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
               {
               cout << "esc key is pressed by user" << endl;
               break; 
               }
               */

            cout << "FabMap: Adding Frame " << i << endl;
            test->addFrame(vframe);



            /*
               imshow("Confusion Matrix", result);
               */
        }

    }


    cout << "FabMap: getting results " << endl;
    Mat result;
    result = test->getMatrix();

    cout << result << endl;
    /*
       cout << "FabMap: saving results "  << endl;
       saveMatFile(mainDir + string("/result"), result);

*/

    cout << "FabMap: printing results "  << endl;
    imshow("Confusion Matrix", result);

    waitKey();


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




    cout << endl << "## End ##" << endl << endl;
    return 0;
}



