#include <stdio.h>
#include <vector>
#include "net.h"
#include <string>
#include <sqlite3.h>

float dot(float* fc1, float* fc2);
int update_feature(sqlite3* facefeatures, char* name, float* fc, int dyUP)
{
    sqlite3_stmt* stat;
    int rc = sqlite3_prepare_v2(facefeatures, "SELECT rowid,NAME,NUMFEA FROM FEATURES WHERE (NAME) = (?)", -1, &stat, NULL);
    if(rc!=SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", sqlite3_errmsg(facefeatures));
        return -1;
    } 
    
    sqlite3_bind_text(stat, 1, name, -1, SQLITE_STATIC);
    if(sqlite3_step(stat)==SQLITE_ROW)
    {
        int rid = sqlite3_column_int(stat, 0);
        int numfea = sqlite3_column_int(stat, 2);
        sqlite3_finalize(stat); 

        sqlite3_blob* blob = NULL;

      //  int r = strcmp((const char*)sqlite3_column_text(stat, 0), name);
      //  if(!r)     
        if(numfea < 10)
        {
            rc = sqlite3_blob_open(facefeatures, "main", "FEATURES", "FEAOFFACE", rid, 1, &blob);
            if (rc != SQLITE_OK)
            {
                printf("Failed to open BLOB: %s \n", sqlite3_errmsg(facefeatures));             
                return -1;
            }

            sqlite3_blob_write(blob, fc, 512, numfea*512);
            if (rc != SQLITE_OK)
            {
                printf("failed to write feature_BLOB!\n");
                return -1;
            }
               
            sqlite3_blob_close(blob);
            rc = sqlite3_prepare_v2(facefeatures, " UPDATE FEATURES SET (NUMFEA) = (?) WHERE (rowid) = (?)", -1, &stat, NULL);
            sqlite3_bind_int(stat, 1, numfea+1);
            sqlite3_bind_int(stat, 2, rid);
            rc = sqlite3_step(stat);
            if( rc != SQLITE_DONE ){
                printf("%s",sqlite3_errmsg(facefeatures));
                return -1;
            }
  
            sqlite3_finalize(stat);  
            fprintf(stderr, "Records %s(rowid:%d)_insert successfully!\n%d features have inserted\n", name, rid, numfea+1);
        } 
        else if(dyUP)
        {
            rc = sqlite3_blob_open(facefeatures,  "main", "FEATURES", "FEAOFFACE", rid, 1, &blob);
            if (rc != SQLITE_OK)
            {
                printf("Failed to open BLOB: %s \n", sqlite3_errmsg(facefeatures));             
                return -1;
            }
   
            float buf[128] = {0.f};
            int offset = 0;
            int minSimID = 0;
            int readSize = 512;
            float similarity = 0;
            float minSim = 0;
            for(int i=0; i<10; i++)
            {
                rc = sqlite3_blob_read(blob, buf, readSize, readSize*i);
                if (rc != SQLITE_OK)
                {
                    printf("failed to read BLOB!\n");
                    break;
                }
               
                similarity = dot(fc,buf);
                if(i == 0)
                minSim = similarity;
                else if(similarity < minSim)
                {
                    minSim = similarity;
                    minSimID = i;
                }
                         
                printf("%d_similarity:%f\n",i, similarity);   
            }
                
            sqlite3_blob_write(blob, fc, 512, minSimID*512);
            sqlite3_blob_close(blob);    
                       
            fprintf(stderr, "%s(rowid:%d)_update successfully!\n", name, rid);
        } 
    }
    return 0;
}
