void TakeSVDOfE(Mat_<double>& E, Mat& svd_u, Mat& svd_vt, Mat& svd_w) {
    //Using OpenCV's SVD
    SVD svd(E,SVD::MODIFY_A);
    svd_u = svd.u;
    svd_vt = svd.vt;
    svd_w = svd.w;

    cout << "----------------------- SVD ------------------------\n";
    cout << "U:\n"<<svd_u<<"\nW:\n"<<svd_w<<"\nVt:\n"<<svd_vt<<endl;
    cout << "----------------------------------------------------\n";
}



bool DecomposeEtoRandT(
        Mat_<double>& E,
        Mat_<double>& R1,
        Mat_<double>& R2,
        Mat_<double>& t1,
        Mat_<double>& t2)
{
    //Using HZ E decomposition
    Mat svd_u, svd_vt, svd_w;
    TakeSVDOfE(E,svd_u,svd_vt,svd_w);

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


