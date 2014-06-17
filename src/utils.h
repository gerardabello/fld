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


void cropImgMargins(Mat& src, int margin, int bottom){

    int w = src.cols;
    int h = src.rows;

    auto myRoi = cvRect(margin, 0,w-(margin*2),h-bottom);

    cv::Mat croppedImage;
    cv::Mat(src, myRoi).copyTo(croppedImage);


    croppedImage.copyTo(src);

}

void initUndistortRectifyMap3( InputArray _cameraMatrix, InputArray _distCoeffs,
                              InputArray _matR, InputArray _newCameraMatrix,
                              Size size, int m1type, OutputArray _map1, OutputArray _map2 )
{
    Mat cameraMatrix = _cameraMatrix.getMat(), distCoeffs = _distCoeffs.getMat();
    Mat matR = _matR.getMat(), newCameraMatrix = _newCameraMatrix.getMat();

    if( m1type <= 0 )
        m1type = CV_16SC2;
    CV_Assert( m1type == CV_16SC2 || m1type == CV_32FC1 || m1type == CV_32FC2 );
    _map1.create( size, m1type );
    Mat map1 = _map1.getMat(), map2;
    if( m1type != CV_32FC2 )
    {
        _map2.create( size, m1type == CV_16SC2 ? CV_16UC1 : CV_32FC1 );
        map2 = _map2.getMat();
    }
    else
        _map2.release();

    Mat_<double> R = Mat_<double>::eye(3, 3);
    Mat_<double> A = Mat_<double>(cameraMatrix), Ar;

    if( newCameraMatrix.data )
        Ar = Mat_<double>(newCameraMatrix);
    else
        Ar = getDefaultNewCameraMatrix( A, size, true );

    if( matR.data )
        R = Mat_<double>(matR);

    if( distCoeffs.data )
        distCoeffs = Mat_<double>(distCoeffs);
    else
    {
        distCoeffs.create(12, 1, CV_64F);
        distCoeffs = 0.;
    }

    CV_Assert( A.size() == Size(3,3) && A.size() == R.size() );
    CV_Assert( Ar.size() == Size(3,3) || Ar.size() == Size(4, 3));
    Mat_<double> iR = (Ar.colRange(0,3)*R).inv(DECOMP_LU);
    const double* ir = &iR(0,0);

    double u0 = A(0, 2), v0 = A(1, 2);
    double fx = A(0, 0), fy = A(1, 1);

    CV_Assert( distCoeffs.size() == Size(1, 4) || distCoeffs.size() == Size(4, 1) ||
               distCoeffs.size() == Size(1, 5) || distCoeffs.size() == Size(5, 1) ||
               distCoeffs.size() == Size(1, 8) || distCoeffs.size() == Size(8, 1) ||
               distCoeffs.size() == Size(1, 12) || distCoeffs.size() == Size(12, 1));

    if( distCoeffs.rows != 1 && !distCoeffs.isContinuous() )
        distCoeffs = distCoeffs.t();

    double k1 = ((double*)distCoeffs.data)[0];
    double k2 = ((double*)distCoeffs.data)[1];
    double p1 = ((double*)distCoeffs.data)[2];
    double p2 = ((double*)distCoeffs.data)[3];
    double k3 = distCoeffs.cols + distCoeffs.rows - 1 >= 5 ? ((double*)distCoeffs.data)[4] : 0.;
    double k4 = distCoeffs.cols + distCoeffs.rows - 1 >= 8 ? ((double*)distCoeffs.data)[5] : 0.;
    double k5 = distCoeffs.cols + distCoeffs.rows - 1 >= 8 ? ((double*)distCoeffs.data)[6] : 0.;
    double k6 = distCoeffs.cols + distCoeffs.rows - 1 >= 8 ? ((double*)distCoeffs.data)[7] : 0.;
    double s1 = distCoeffs.cols + distCoeffs.rows - 1 >= 12 ? ((double*)distCoeffs.data)[8] : 0.;
    double s2 = distCoeffs.cols + distCoeffs.rows - 1 >= 12 ? ((double*)distCoeffs.data)[9] : 0.;
    double s3 = distCoeffs.cols + distCoeffs.rows - 1 >= 12 ? ((double*)distCoeffs.data)[10] : 0.;
    double s4 = distCoeffs.cols + distCoeffs.rows - 1 >= 12 ? ((double*)distCoeffs.data)[11] : 0.;

    for( int i = 0; i < size.height; i++ )
    {
        float* m1f = (float*)(map1.data + map1.step*i);
        float* m2f = (float*)(map2.data + map2.step*i);
        short* m1 = (short*)m1f;
        ushort* m2 = (ushort*)m2f;
        double _x = i*ir[1] + ir[2], _y = i*ir[4] + ir[5], _w = i*ir[7] + ir[8];

        for( int j = 0; j < size.width; j++, _x += ir[0], _y += ir[3], _w += ir[6] )
        {
            double w = 1./_w, x = _x*w, y = _y*w;
            double x2 = x*x, y2 = y*y;
            double r2 = x2 + y2, _2xy = 2*x*y;
            double kr = (1 + ((k3*r2 + k2)*r2 + k1)*r2)/(1 + ((k6*r2 + k5)*r2 + k4)*r2);
            double u = fx*(x*kr + p1*_2xy + p2*(r2 + 2*x2) + s1*r2+s2*r2*r2) + u0;
            //double u = fx*kr*x + u0;
            //double v = fy*(y*kr + p1*(r2 + 2*y2) + p2*_2xy + s3*r2+s4*r2*r2) + v0;
            double v = fy*y + v0;
            if( m1type == CV_16SC2 )
            {
                int iu = saturate_cast<int>(u*INTER_TAB_SIZE);
                int iv = saturate_cast<int>(v*INTER_TAB_SIZE);
                m1[j*2] = (short)(iu >> INTER_BITS);
                m1[j*2+1] = (short)(iv >> INTER_BITS);
                m2[j] = (ushort)((iv & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (iu & (INTER_TAB_SIZE-1)));
            }
            else if( m1type == CV_32FC1 )
            {
                m1f[j] = (float)u;
                m2f[j] = (float)v;
            }
            else
            {
                m1f[j*2] = (float)u;
                m1f[j*2+1] = (float)v;
            }
        }
    }
}

void undistort3( InputArray _src, OutputArray _dst, InputArray _cameraMatrix,
        InputArray _distCoeffs, InputArray _newCameraMatrix )
{
    Mat src = _src.getMat(), cameraMatrix = _cameraMatrix.getMat();
    Mat distCoeffs = _distCoeffs.getMat(), newCameraMatrix = _newCameraMatrix.getMat();

    _dst.create( src.size(), src.type() );
    Mat dst = _dst.getMat();

    CV_Assert( dst.data != src.data );

    int stripe_size0 = std::min(std::max(1, (1 << 12) / std::max(src.cols, 1)), src.rows);
    Mat map1(stripe_size0, src.cols, CV_16SC2), map2(stripe_size0, src.cols, CV_16UC1);

    Mat_<double> A, Ar, I = Mat_<double>::eye(3,3);

    cameraMatrix.convertTo(A, CV_64F);
    if( distCoeffs.data )
        distCoeffs = Mat_<double>(distCoeffs);
    else
    {
        distCoeffs.create(5, 1, CV_64F);
        distCoeffs = 0.;
    }

    if( newCameraMatrix.data )
        newCameraMatrix.convertTo(Ar, CV_64F);
    else
        A.copyTo(Ar);

    double v0 = Ar(1, 2);
    for( int y = 0; y < src.rows; y += stripe_size0 )
    {
        int stripe_size = std::min( stripe_size0, src.rows - y );
        Ar(1, 2) = v0 - y;
        Mat map1_part = map1.rowRange(0, stripe_size),
            map2_part = map2.rowRange(0, stripe_size),
            dst_part = dst.rowRange(y, y + stripe_size);


        initUndistortRectifyMap3( A, distCoeffs, I, Ar, Size(src.cols, stripe_size),map1_part.type(), map1_part, map2_part );
        //initUndistortRectifyMap( A, distCoeffs, I, Ar, Size(src.cols, stripe_size),map1_part.type(), map1_part, map2_part );
        remap( src, dst_part, map1_part, map2_part, INTER_LINEAR, BORDER_CONSTANT );
    }
}

bool rectify(Mat& sr){

    //double d[5] = {-0.307221, 0.065639, -0.001296, 0.000508, 0};
    double d[5] = {-0.307221, 0.065639, -0.001296, 0.000508, 0};
    Mat dist = Mat(1, 5, CV_64F, d);

    double m[3][3] = {{554.330365, 0, 515.043549},{ 0, 552.271147, 398.77278},{ 0, 0, 1}};
    Mat camera_mat = Mat(3, 3, CV_64F, m);

    Mat temp;

    Mat ncm;
    undistort3(sr, temp, camera_mat, dist, ncm);

    temp.copyTo(sr);

}


bool readOmniFrame(vector<VideoCapture> vcv, vector<Mat>& iv){

    vector<VideoCapture>::iterator l;
    for(l = vcv.begin(); l != vcv.end(); l++) {
        Mat m;
        bool bSuccess = l->read(m);
        if(!bSuccess){
            return false;
        }

        rectify(m);
        cropImgMargins(m, 20, 70);
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
