#include <stdio.h>
#include <vector>
#include "net.h"
#include <string>
#include <sqlite3.h>
#include <math.h>
#include "dlib/image_io.h"

using namespace dlib;

int get_face_feature(array2d<rgb_pixel>& rgb_img, ncnn::Mat& fc);
int image_preprocess(const char* imagepath, array2d<rgb_pixel>&  face_chips);

float dot(float* fc1, float* fc2)
{
    float sum = 0;
    for(int i=0; i<128; i++)
    {
        sum += (*(fc1+i)) * (*(fc2+i));
    }
    return sum;
}

int normalize(ncnn::Mat& fc1)
{
    float sq = sqrt(dot((float*)fc1.data, (float*)fc1.data));
    for(int i=0; i<fc1.w*fc1.h*fc1.c; i++)
    {
        *(fc1.row(0)+i) = (*(fc1.row(0)+i))/sq;
    }
    return 0;
}

int knn(std::vector< std::pair<int, float> >& re, int k)
{
    std::pair<int, float> temp;
    for(int i=0; i<re.size(); i++)
    {
	for(int j=i+1; j<re.size(); j++)
	{
            if(re[j].second > re[i].second)
            {
            	temp = re[i];
                re[i] = re[j];
                re[j] = temp;
            }
        }
    }
   
    if(re[0].second > 0.6)
    { 
    std::vector< std::pair<int, int> > vote;
    vote.push_back(std::make_pair(re[0].first, 1));
    for(int i=1; i<k; i++)
    {
        int j=0;
	for(; j<vote.size(); j++)
        {
            if(vote[j].first == re[i].first)
            {
                vote[j].second += 1;
                break;
            }
        }
        if(j == vote.size())
        {
            vote.push_back(std::make_pair(re[i].first, 1));
        } 
    }
    
    float max = 0;
   // printf("(ID:%d)__ballot:%d\n", vote[0].first, vote[0].second);
    for(int j=1; j<vote.size(); j++)
    {
        if(vote[j].second > vote[0].second)
        {
            max = j;
        }
   //     printf("(ID:%d)__ballot:%d\n", vote[j].first, vote[j].second);
    }
    printf("results:\n(ID:%d)__ballot:%d\n", vote[max].first, vote[max].second);
    
    return vote[max].first;
    }
    else
    {
   		return -1; 
    }
}

int face_recognise(const char* imagepath, std::string& recog_name)
{
    //打开数据库并读出所有name 
    sqlite3* facefeatures;
    char* zErrMsg = 0;
    char* sql;
    sqlite3_stmt* stat;

    int rc = sqlite3_open("face_feature.db", &facefeatures);    
    if(rc)
    {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(facefeatures));
        exit(0);
    }
    else
    {
        fprintf(stdout, "Opened database successfully\n");
    }

    std::vector<std::string> name;
    rc = sqlite3_prepare_v2(facefeatures, "SELECT NAME FROM FEATURES", -1, &stat, NULL);
    if(rc!=SQLITE_OK) {
                fprintf(stderr,"read name error\n");
                sqlite3_free(zErrMsg);
                return 0;
    }
    while(sqlite3_step(stat)==SQLITE_ROW)
    {
         name.push_back((std::string)(char*)sqlite3_column_text(stat, 0));
  //       fprintf(stderr, "name _:%s\n", sqlite3_column_text(stat, 0));
    }
    fprintf(stderr, "numRows:%d\n", name.size());
    sqlite3_finalize(stat); 

    clock_t start,finsh;
    start = clock();   
    ncnn::Mat fc;
    array2d<rgb_pixel> rgb_img;
    
    if(image_preprocess(imagepath, rgb_img) != 0)
    {
        fprintf(stdout, "stop recognition.......\n");
        return -1;
    }
    finsh = clock();
    fprintf(stderr, "dlib_pre cost %d ms\n", (finsh-start)/1000);

    start = clock();
    get_face_feature(rgb_img, fc);
    normalize(fc);
    finsh = clock();
    fprintf(stderr, "get_feature cost %d ms\n", (finsh-start)/1000);
   
    //读取数据库中脸部特征数据
    start = clock();
    int i=0;
    float similarity;
    std::vector< std::pair<int, float> > results;
    for(; i<name.size(); i++)
    {
        sqlite3_blob* blob = NULL;
        rc = sqlite3_blob_open(facefeatures,  "main", "FEATURES", "FEAOFFACE", i+1, 0, &blob);
        if (rc != SQLITE_OK)
        {
            printf("Failed to open BLOB: %s \n", sqlite3_errmsg(facefeatures));
            return -1;
        }
        int blob_length = sqlite3_blob_bytes(blob);
        printf("blob_length_%d\n", blob_length);
    
        float buf[128] = {0.f};
        int offset = 0;
        while (offset < blob_length)
        {
            int size = blob_length - offset;
            if (size > 128*4) size = 128*4;
            rc = sqlite3_blob_read(blob, buf, size, offset);
            if (rc != SQLITE_OK)
            {
                printf("failed to read BLOB!\n");
                break;
            }
        
            offset += size;
            similarity = dot((float*)fc.data, buf);
            fprintf(stderr, "%d_similarity:%f\n",i, similarity);
            results.push_back(std::make_pair(i, similarity));
        }
        sqlite3_blob_close(blob);
    }
    
    int ID;
    ID = knn(results, 3);
    if(ID+1)
    {
        recog_name = name[ID];
    }
    else
    {
        recog_name = (std::string)"unknown!";
    }
    finsh = clock();
    fprintf(stderr, "knn cost %d ms\n", (finsh - start)/1000);
    sqlite3_close(facefeatures);

    return 0;
}
