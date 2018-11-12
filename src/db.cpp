
#include<stdio.h>
#include<string>
#include<sqlite3.h>
#include"net.h"
#include"dlib/image_io.h"
int create_db(const char* image_dir);
int main(int argc, char* argv[])
{
    const char* image_dir = argv[1];
    create_db(image_dir);  
}
