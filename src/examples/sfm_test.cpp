#include "../fld.h"

#include "SfMToyLib/MultiCameraPnP.h"
#include "SfMToyLib/Distance.h"
#include "SfMToyLib/SfMUpdateListener.h"

#include "SFML/Graphics.hpp"
#include "SFML/OpenGL.hpp"

#include "Eigen/Eigen"

sf::RenderWindow* window;

std::vector<cv::Point3d> m_pcld;
std::vector<cv::Vec3b> m_pcldrgb;
std::vector<cv::Matx34d> m_cameras;
std::vector<Eigen::Affine3d> m_cameras_transforms;
Eigen::Affine3d m_global_transform;

// QThreadPool qtp;

float vizScale;
double scale_cameras_down;
float r_x,r_y,r_z;

bool drag = false;

int lastx, lasty;


void updateGL();



class sul : public SfMUpdateListener{

    public:

        void update(std::vector<cv::Point3d> pcld,
                std::vector<cv::Vec3b> pcldrgb,
                std::vector<cv::Point3d> pcld_alternate,
                std::vector<cv::Vec3b> pcldrgb_alternate,
                std::vector<cv::Matx34d> cameras){
            vizScale = 0.2;

            cout << "Cameras:" << endl;


            vector<Matx34d>::iterator l;
            for(l = cameras.begin(); l != cameras.end(); l++) {
                cout << *l << endl;
            }

/*
            cout << "Points:" << endl;


            vector<Point3d>::iterator li;
            for(li = pcld.begin(); li != pcld.end(); li++) {
                cout << *li << endl;
            }
*/

            //opengl


            m_pcld = pcld;
            m_pcldrgb = pcldrgb;
            m_cameras = cameras;

            //get the scale of the result cloud using PCA
            {
                cv::Mat_<double> cldm(pcld.size(), 3);
                for (unsigned int i = 0; i < pcld.size(); i++) {
                    cldm.row(i)(0) = pcld[i].x;
                    cldm.row(i)(1) = pcld[i].y;
                    cldm.row(i)(2) = pcld[i].z;
                }
                cv::Mat_<double> mean; //cv::reduce(cldm,mean,0,CV_REDUCE_AVG);
                cv::PCA pca(cldm, mean, CV_PCA_DATA_AS_ROW);
                scale_cameras_down = 1.0;// / (3.0 * sqrt(pca.eigenvalues.at<double> (0)));
                // std::cout << "emean " << mean << std::endl;
                //m_global_transform = Eigen::Translation<double,3>(-Eigen::Map<Eigen::Vector3d>(mean[0]));
            }

            //compute transformation to place cameras in world
            m_cameras_transforms.resize(m_cameras.size());
            Eigen::Vector3d c_sum(0,0,0);
            for (int i = 0; i < m_cameras.size(); ++i) {
                Eigen::Matrix<double, 3, 4> P = Eigen::Map<Eigen::Matrix<double, 3, 4,
                    Eigen::RowMajor> >(m_cameras[i].val);
                Eigen::Matrix3d R = P.block(0, 0, 3, 3);
                cout << "R:" << R << endl;
                Eigen::Vector3d t = P.block(0, 3, 3, 1);
                cout << "T:" << t << endl;
                Eigen::Vector3d c = -R.transpose() * t;
                c_sum += c;
                m_cameras_transforms[i] =
                    Eigen::Translation<double, 3>(c) *
                    Eigen::Quaterniond(R) *
                    Eigen::UniformScaling<double>(scale_cameras_down)
                    ;
            }

            m_global_transform = Eigen::Translation<double,3>(-c_sum / (double)(m_cameras.size()));
            // m_global_transform = m_cameras_transforms[0].inverse();

            updateGL();

        }
};


void updateGL(){


    ///////////////////////////////////////////
    
    
    glClearColor(0.1f, 0.15f, 0.2f, 1.0f );
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	// Clean the screen and the depth buffer
    glLoadIdentity();									// Reset The Projection Matrix /

    glPushMatrix();

    glRotatef(r_x, 1.0f, 0.0f, 0.0f); 
    glRotatef(r_y, 0.0f, 1.0f, 0.0f); 
    glRotatef(r_z, 0.0f, 0.0f, 1.0f); 

    glScaled(vizScale,vizScale,vizScale);
    glMultMatrixd(m_global_transform.data());

    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_LIGHTING);
    glBegin(GL_POINTS);
    for (int i = 0; i < m_pcld.size(); ++i) {
        glColor3ub(m_pcldrgb[i][2],m_pcldrgb[i][1],m_pcldrgb[i][0]);
        glVertex3dv(&(m_pcld[i].x));
    }
    glEnd();

    // glScaled(scale_cameras_down,scale_cameras_down,scale_cameras_down);
    glEnable(GL_RESCALE_NORMAL);
    //glEnable(GL_LIGHTING);
    for (int i = 0; i < m_cameras_transforms.size(); ++i) {

        glPushMatrix();
        glMultMatrixd(m_cameras_transforms[i].data());

        glLineWidth(2.0f);

        glBegin(GL_LINES);
        glColor3f(1, 0, 0);
        glVertex3f(0, 0, 0);
        glVertex3f(1, 0, 0);

        glColor3f(0, 1, 0);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 1, 0);

        glColor3f(0, 0, 1);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 0, 1);
        glEnd();

        glPopMatrix();
    }

    glPopAttrib();
    glPopMatrix();
}


void setupSFML(){

    window = new sf::RenderWindow(sf::VideoMode(1024, 720), "SFML works!");
    sf::CircleShape shape(100.f);
    shape.setFillColor(sf::Color::Green);


}


void testSfM(){



    setupSFML();


    std::vector<cv::Mat> images;
    std::vector<std::string> images_names;

    open_imgs_dir("../dat/sfm/",images,images_names,1.0);


    cv::Ptr<MultiCameraPnP> distance = new MultiCameraPnP(images,images_names,"../dat/sfm/");
    distance->use_rich_features = true;
    distance->use_gpu = false;


    sul *l = new sul();

    distance->attach(l);


    distance->RecoverDepthFromImages();


    int mx, my;



    glOrtho(-1.0, 1.0, -1.0, 1.0, 0.1, 1000);

    while (window->isOpen())
    {
        sf::Event event;
        while (window->pollEvent(event))
        {

            switch (event.type)
            {
                case sf::Event::Closed:
                    window->close();
                    break;

                case sf::Event::MouseButtonPressed:
                    drag = true;
                    break;
                case sf::Event::MouseButtonReleased:
                    drag = false;
                    break;
                case sf::Event::MouseWheelMoved:
                    vizScale += (vizScale*0.1)*event.mouseWheel.delta;
                    break;


                default:
                    break;
            }
        }

        mx = sf::Mouse::getPosition().x;
        my = sf::Mouse::getPosition().y;

        if (drag) {
            r_x += (lasty - my)/4;
            r_y += (lastx - mx)/4;
        }

        lastx = mx;
        lasty = my;

        updateGL();
        //window->draw(shape);
        window->display();
    }


}
