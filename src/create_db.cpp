#include <stdio.h>
#include "dlib/image_io.h"
#include <dirent.h>
#include <sqlite3.h>
#include <vector>
#include <string>
#include "net.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing.h"
#include <iostream>
 
using namespace dlib;
using namespace std;

 
int preprocess(const char* imagepath, array2d<rgb_pixel>&  face_chips, frontal_face_detector& de, shape_predictor& sp)
{
 
	try
	{


                clock_t start, finsh; 
                cout << "processing image " << imagepath << endl;
		array2d<rgb_pixel> img;
		load_image(img, imagepath);
//                img.set_size(600,400);
//                resize_image(image, img);
//		pyramid_up(img);
		start = clock();
         	std::vector<rectangle> dets = de(img);
 
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


int normalize(ncnn::Mat& fc1);

int get_face_feature(array2d<rgb_pixel>&  face_chips, ncnn::Mat& fc);
int image_preprocess(const char* imagepath, array2d<rgb_pixel>&  face_chips);

void getAllImages(const char* path, std::vector< std::pair<std::string, ncnn::Mat> >& entry, frontal_face_detector& detector, shape_predictor& sp)
{

    DIR* pDir;
    struct dirent* ptr;

    pDir = opendir(path);
   
    //插入不同人的脸部特征记录
    if(pDir)
    {
        int con=1;
        while((ptr = readdir(pDir)) != 0 && con)
        {
           if (ptr->d_type == DT_DIR)
           {
               
               if (strcmp(ptr -> d_name,".") != 0 && strcmp(ptr -> d_name,"..") != 0) 
               {
                   getAllImages((std::string(path) + "/" + ptr -> d_name).data(), entry, detector, sp);
               }
              
           }
           else if(ptr->d_type == DT_REG)
           {
               std::size_t found = std::string(path).find_last_of("/");
               std::string name = std::string(path).substr(found+1);
              
               ncnn::Mat prob; 
               array2d<rgb_pixel>  face_chips;
               int i;
               for(i=0; i<(int)entry.size(); i++)
               {
                   if(name.compare(entry[i].first)  == 0)
                   {
                       if(entry[i].second.h == 10)
                      { con = 0;
                       break;
                       }                       
                       if(preprocess((std::string(path) + '/' + ptr -> d_name).data(), face_chips, detector, sp) != 0)
                       {fprintf(stdout, "stop insert this one, continue next one!\n");return;}
                       get_face_feature(face_chips, prob);
                       normalize(prob);
                       ncnn::Mat preds(prob.w, entry[i].second.h+1, 1);
                       for(int h=0; h<preds.h; h++)
                       {
                           for(int w=0; w<prob.w; w++)
                           {
                               if(h == preds.h-1)
                               *(preds.row(entry[i].second.h) + w) =  *(prob.row(0) + w); 
                               else
                               *(preds.row(h) + w) =  *(entry[i].second.row(h) + w); 
                           }
                       }
                       entry[i] = std::make_pair(name, preds);
                       printf("%s\n", ptr -> d_name);
                       break;
                  }
               }
               if(i == entry.size())
               {
                   
                   if(preprocess((std::string(path) + '/' + ptr -> d_name).data(), face_chips, detector, sp) != 0)
                   {fprintf(stdout, "stop insert this one, continue next one!\n");return;}

                   get_face_feature(face_chips, prob);
                   normalize(prob);
                   entry.push_back(std::make_pair(name, prob));
                   printf("__________add_entry_%s___________:\n%s\n",name.data(), ptr -> d_name);
               }            
           }   
        }
    }
}

//创建脸部特征数据库
int create_db(const char* image_dir)
{
		clock_t start;
                clock_t finsh;
		start = clock();
		frontal_face_detector detector = get_frontal_face_detector();
                finsh = clock();
                cout << "det_get_fro cost " << (finsh-start)/1000 << "ms" << endl;

		start = clock();
		shape_predictor sp;
		deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
                finsh = clock();
                cout << "deserialize cost " << (finsh-start)/1000 << "ms" << endl;

    //获取图片数据
    std::vector< std::pair<std::string, ncnn::Mat> > entry;
    getAllImages(image_dir, entry, detector, sp);
    printf("entry_size: %d\n", entry.size());
    
    sqlite3 *db;
    char *zErrMsg = 0;
    int  rc;
    char *sql;
    sqlite3_stmt* stmt;

    //打开数据库文件
    rc = sqlite3_open("face_feature.db", &db);
    if( rc ){
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
        exit(0);
    }else{
        fprintf(stdout, "Opened database successfully\n");
    }

    //创建脸部特征表
    sql = "CREATE TABLE FEATURES("  \
         "NAME           VARCHAR(256)    NOT NULL," \
         "NUMFEA         INT     NOT NULL," \
         "FEAOFFACE      BLOB    NOT NULL );";
    rc = sqlite3_exec(db, sql, NULL, 0, &zErrMsg);
    if( rc != SQLITE_OK ){
    fprintf(stderr, "SQL error: %s\n", zErrMsg);
       sqlite3_free(zErrMsg);
    }else{
       fprintf(stdout, "Table created successfully\n");
    }
    
    //插入脸部特征数据条目
    for(int i=0; i<entry.size(); i++)
    {
        if(entry[i].second.h == 10)
        {
        sqlite3_prepare(db,"INSERT INTO FEATURES (NAME,NUMFEA,FEAOFFACE) VALUES(?,?,?);",-1,&stmt,NULL);
        sqlite3_bind_text(stmt,1,entry[i].first.data(),-1,SQLITE_STATIC);
        sqlite3_bind_int(stmt,2,entry[i].second.h);
        sqlite3_bind_blob(stmt,3,entry[i].second.data,entry[i].second.w*entry[i].second.h*4,SQLITE_TRANSIENT);
        rc = sqlite3_step(stmt);
        if( rc != SQLITE_DONE ){
            printf("%s",sqlite3_errmsg(db));
        }
        else{
            fprintf(stdout, "Records created successfully\n");
        }
        printf("%s\n", entry[i].first.data());
        sqlite3_finalize(stmt);
        }
    }   

    sqlite3_close(db); 
    return 0;
}
