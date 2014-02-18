#include "../fld.h"


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

}


int main(int argc, char *argv[])
{
    CFld *test;

    test = new CFld();

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


    return 0;
}



