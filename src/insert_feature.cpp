#include <stdio.h>
#include <vector>
#include "net.h"
#include <string>
#include <sqlite3.h>

int update_feature(sqlite3* facefeatures, char* name, float* fc, int dyUP);

int insert_feature(sqlite3* facefeatures, char* name, float* fc)
{
    char* zErrMsg;
    char* sql;
    sql = "CREATE TABLE IF NOT EXISTS FEATURES("  \
         "NAME           VARCHAR(256)    NOT NULL," \
         "NUMFEA         INT     NOT NULL," \
         "FEAOFFACE      BLOB    NOT NULL );";
    int  rc;
    rc = sqlite3_exec(facefeatures, sql, NULL, 0, &zErrMsg);
    if( rc != SQLITE_OK ){
       fprintf(stderr, "SQL error: %s\n", zErrMsg);
       sqlite3_free(zErrMsg);
    }else{
       fprintf(stdout, "Operate on Table FEATURES\n");
    }

    sqlite3_stmt* stat;
    rc = sqlite3_prepare_v2(facefeatures, "SELECT NAME FROM FEATURES WHERE (NAME) = (?)", -1, &stat, NULL);
    if(rc!=SQLITE_OK) {
        printf("%s",sqlite3_errmsg(facefeatures));
        return -1;
    }
    
    int flag = 0;
    sqlite3_bind_text(stat, 1, name, -1, SQLITE_STATIC);
    
    if(sqlite3_step(stat)==SQLITE_ROW)
    {
        sqlite3_finalize(stat);
        update_feature(facefeatures, name, fc, 0);
    }
    else
    {    
        sqlite3_prepare_v2(facefeatures,"INSERT INTO FEATURES (NAME,NUMFEA,FEAOFFACE) VALUES(?,?,?);",-1,&stat, NULL);
        sqlite3_bind_text(stat,1,name, -1, SQLITE_STATIC);
        sqlite3_bind_int(stat,2,1);
        float fe[1280] = {0.f};
        for(int j=0; j<128; j++)
        { 
            fe[j] = *(fc + j);
        }
        sqlite3_bind_blob(stat, 3, fe, 5120, SQLITE_STATIC);
        rc = sqlite3_step(stat);
        if( rc != SQLITE_DONE ){
            printf("%s",sqlite3_errmsg(facefeatures));
            return -1;
        }
        else{
            fprintf(stderr, "Records %s created successfully!\n%d features have inserted\n", name, 1);
        }
        sqlite3_finalize(stat);
    }

//    sqlite3_close(facefeatures); 
    return 0;
}
