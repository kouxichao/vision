#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing.h"
#include "dlib/image_io.h"
#include "iostream"
 
using namespace dlib;
using namespace std;
 
int image_preprocess(const char* imagepath, array2d<rgb_pixel>&  face_chips)
{
 
	try
	{

		clock_t start,finsh;
		start = clock();
		frontal_face_detector detector = get_frontal_face_detector();
                finsh = clock();
                cout << "det_get_fro cost " << (finsh-start)/1000 << "ms" << endl;

		start = clock();
		shape_predictor sp;
		deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
                finsh = clock();
                cout << "deserialize cost " << (finsh-start)/1000 << "ms" << endl;

                
                cout << "processing image " << imagepath << endl;
		array2d<rgb_pixel> img;
		load_image(img, imagepath);
//                img.set_size(600,400);
//                resize_image(image, img);
//		pyramid_up(img);
	
		start = clock();
         	std::vector<rectangle> dets = detector(img);
 
		finsh = clock();

                cout << "det cost " << (finsh-start)/1000 << "s" << endl;
			// Now we will go ask the shape_predictor to tell us the pose of
			// each face we detected.
		start = clock();
                int index = 0;
                float area = 0; 
                if(dets.size() == 0)
                {
                    cout << "no face is detected" <<endl;
	            return -1;
		}
		for (unsigned long j = 0; j < dets.size(); ++j)
		{
                    float width = dets[j].right() - dets[j].left();
                    float height = dets[j].bottom() - dets[j].top();
                    if(width * height > area)
                    {
                        index = j;
                        area = width * height;
                    }
                    
		}
                finsh = clock();
                cout << "area cost " << (finsh - start)/1000 << "ms" << endl;             
		start = clock();
                full_object_detection shape = sp(img, dets[index]);
		extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chips);
		finsh = clock();
                cout << "sp cost " << (finsh-start)/1000 << "ms" << endl;
                
		cout << "image process finished" << endl;
                
	}
	catch (exception& e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
        return 0;
}

