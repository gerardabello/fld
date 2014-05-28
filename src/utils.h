void TakeSVDOfE_2(Mat_<double>& E, Mat& svd_u, Mat& svd_vt, Mat& svd_w) {
    //Using OpenCV's SVD
    SVD svd(E,SVD::MODIFY_A);
    svd_u = svd.u;
    svd_vt = svd.vt;
    svd_w = svd.w;

    cout << "----------------------- SVD ------------------------\n";
    cout << "U:\n"<<svd_u<<"\nW:\n"<<svd_w<<"\nVt:\n"<<svd_vt<<endl;
    cout << "----------------------------------------------------\n";
}



bool DecomposeEtoRandT_2(
        Mat_<double>& E,
        Mat_<double>& R1,
        Mat_<double>& R2,
        Mat_<double>& t1,
        Mat_<double>& t2)
{
    //Using HZ E decomposition
    Mat svd_u, svd_vt, svd_w;
    TakeSVDOfE_2(E,svd_u,svd_vt,svd_w);

    //check if first and second singular values are the same (as they should be)
    double singular_values_ratio = fabsf(svd_w.at<double>(0) / svd_w.at<double>(1));
    if(singular_values_ratio>1.0) singular_values_ratio = 1.0/singular_values_ratio; // flip ratio to keep it [0,1]
    if (singular_values_ratio < 0.7) {
        cout << "singular values are too far apart\n";
        return false;
    }

    Matx33d W(0,-1,0,	//HZ 9.13
            1,0,0,
            0,0,1);
    Matx33d Wt(0,1,0,
            -1,0,0,
            0,0,1);
    R1 = svd_u * Mat(W) * svd_vt; //HZ 9.19
    R2 = svd_u * Mat(Wt) * svd_vt; //HZ 9.19
    t1 = svd_u.col(2); //u3
    t2 = -svd_u.col(2); //u3
    return true;
}


void cropImgMargins(Mat& src, int margin){

    int w = src.cols;
    int h = src.rows;

    auto myRoi = cvRect(margin, 0,w-(margin*2),h);

    cv::Mat croppedImage;
    cv::Mat(src, myRoi).copyTo(croppedImage);


    croppedImage.copyTo(src);

}


bool readOmniFrame(vector<VideoCapture> vcv, vector<Mat>& iv){

    vector<VideoCapture>::iterator l;
    for(l = vcv.begin(); l != vcv.end(); l++) {
        Mat m;
        bool bSuccess = l->read(m);
        if(!bSuccess){
            return false;
        }

        cropImgMargins(m, 50);
        iv.push_back(m);
    }

    return true;
}


void combinator( vector<Mat>& iv, Mat &result){
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

    result = combine.clone();

}



bool combineImages(vector<VideoCapture> vcv, Mat &result){


    vector<Mat> iv;
    bool a = readOmniFrame(vcv, iv);

    if(!a) return false;

    combinator(iv,result);

    return true;
}


void getOnePano(string path, string num, Mat& result){

    vector<Mat> iv;

    iv.push_back( imread(path+"/fl/frame"+num+".jpg" , IMREAD_COLOR));
    iv.push_back( imread(path+"/fc/frame"+num+".jpg" , IMREAD_COLOR));
    iv.push_back( imread(path+"/fr/frame"+num+".jpg" , IMREAD_COLOR));
    iv.push_back( imread(path+"/rr/frame"+num+".jpg" , IMREAD_COLOR));
    iv.push_back( imread(path+"/rl/frame"+num+".jpg" , IMREAD_COLOR));

    combinator(iv,result);
}


void getVideoCaptureVector(string path, vector<VideoCapture>& vcv){


    VideoCapture cap5(path + string("/fl/frame%4d.jpg")); // open the video file for reading
    VideoCapture cap1(path + string("/fc/frame%4d.jpg")); // open the video file for reading
    VideoCapture cap2(path + string("/fr/frame%4d.jpg")); // open the video file for reading
    VideoCapture cap3(path + string("/rr/frame%4d.jpg")); // open the video file for reading
    VideoCapture cap4(path + string("/rl/frame%4d.jpg")); // open the video file for reading


    vcv.push_back(cap1);
    vcv.push_back(cap2);
    vcv.push_back(cap3);
    vcv.push_back(cap4);
    vcv.push_back(cap5);

}
