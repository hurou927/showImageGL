/*

size_t Max,Width,Length,SamplePerPixel;
unsigned char * img = ReadPNM(filename,NULL, &Max,&Width,&Length,&SamplePerPixel);
printf("Max,%d,Width,%d,Length,%d,SamplePerPixel,%d\n",Max,Width,Length,SamplePerPixel);

 */

#ifndef _MYPGM
#define _MYPGM
#include <stdio.h>
#include <stdlib.h>


unsigned char *ReadPNM   (const char *filename, char *Format, size_t *Max , size_t *Width, size_t *Length,size_t *SamplePerPixel);
void WritePNM (const char *filename, unsigned char *image, size_t Max ,size_t Width,size_t Length,size_t SamplePerPixel);


unsigned char *ReadPGM   (const char *filename, char *Format, size_t *Max , size_t *Width, size_t *Length);
unsigned char *ReadPGM_P2(const char *filename, size_t *Max ,size_t *Width, size_t *Length);
unsigned char *ReadPGM_P5(const char *filename, size_t *Max ,size_t *Width, size_t *Length);
unsigned char *ReadPPM_P3(const char *filename, size_t *Max ,size_t *Width, size_t *Length);
unsigned char *ReadPPM_P6(const char *filename, size_t *Max ,size_t *Width, size_t *Length);

void WritePGM_P2(const char *filename, unsigned char *image, size_t Max ,size_t Width,size_t Length);
void WritePGM_P5(const char *filename, unsigned char *image, size_t Max ,size_t Width,size_t Length);
void WritePGM_P3(const char *filename, unsigned char *image, size_t Max ,size_t Width,size_t Length);
void WritePGM_P6(const char *filename, unsigned char *image, size_t Max ,size_t Width,size_t Length);

unsigned char *ReadPNM(const char *filename, char *Format, size_t *Max, size_t *Width, size_t *Length,size_t *SamplePerPixel){



    char format[256];
    unsigned char *image = NULL;
    size_t W,L,M;
    FILE *fp;

    if((fp = fopen(filename , "r" ))==NULL){
        fprintf(stderr,"Error::%s::%s::%d::fopen\n",__FILE__,__FUNCTION__,__LINE__);
        exit(1);
    }

    fgets(format,256,fp);
    fclose(fp);
    if(format[0]=='P' && format[1]=='2'){
        image=ReadPGM_P2(filename,&M,&W,&L);
        *SamplePerPixel=1;
    }else if(format[0]=='P' && format[1]=='5'){
        image=ReadPGM_P5(filename,&M,&W,&L);
        *SamplePerPixel=1;
    }else if(format[0]=='P' && format[1]=='3'){
        image=ReadPPM_P3(filename,&M,&W,&L);
        *SamplePerPixel=3;
    }else if(format[0]=='P' && format[1]=='6'){
        image=ReadPPM_P6(filename,&M,&W,&L);
        *SamplePerPixel=3;
    }else{
        printf("undefined format %s\n",format);
    }

    if(Format!=NULL){
        Format[0]=format[0];Format[1]=format[1];Format[2]='\0';
    }

    *Width  = W;
    *Length = L;
    *Max=M;

    return image;

}



void WritePNM(const char *filename, unsigned char *image, size_t Max ,size_t Width,size_t Length,size_t SamplePerPixel){

    if(SamplePerPixel==1)
        WritePGM_P5(filename,image,Max ,Width,Length);
    else if(SamplePerPixel==3)
        WritePGM_P6(filename,image,Max ,Width,Length);

}



unsigned char *ReadPGM(const char *filename, char *Format, size_t *Max, size_t *Width, size_t *Length){


    char format[256];
    unsigned char *image = NULL;
    size_t W,L,M;
    FILE *fp;
    if((fp = fopen(filename , "r" ))==NULL){
            fprintf(stderr,"fopen FAILURE\n");
        exit(1);
    }
    fgets(format,256,fp);
    fclose(fp);
    if(format[0]=='P' && format[1]=='2'){
            image=ReadPGM_P2(filename,&M,&W,&L);
    }else if(format[0]=='P' && format[1]=='5'){
            image=ReadPGM_P5(filename,&M,&W,&L);
    }else if(format[0]=='P' && format[1]=='3'){
            image=ReadPPM_P3(filename,&M,&W,&L);
    }else if(format[0]=='P' && format[1]=='6'){
            image=ReadPPM_P6(filename,&M,&W,&L);
    }else{
            printf("undefined format %s\n",format);
    }
    if(Format!=NULL){
            Format[0]=format[0];Format[1]=format[1];Format[2]='\0';
    }
    *Width  = W;
    *Length = L;
    *Max=M;
    return image;
}
unsigned char *ReadPGM_P2(const char *filename, size_t *Max, size_t *Width, size_t *Length){
        /**
     * P2
     * #comment
     * Width Length
     * image data(text)......
     */
    char format[256];
    char comment[256];
    size_t W,L,max;
    unsigned char *image;
    FILE *fp;
    if((fp = fopen(filename , "r" ))==NULL){
            fprintf(stderr,"fopen FAILURE\n");
        exit(1);
    }
    fgets(format,256,fp);
    if(format[0]!='P' || format[1]!='2'){
            fprintf(stderr,"format error. File must be pgm(P2) format. Your file is %s\n",format);
        exit(1);
    }
    int commentCountAdd1=1;
    while(1){
    		fgets(comment,256,fp);
		if(comment[0]!='#'){
    			fseek(fp, 0, SEEK_SET);
            while(commentCountAdd1--)
			    fgets(comment,256,fp);
			break;
		}
        commentCountAdd1++;
	}
    fscanf(fp,"%zu",&W);
    fscanf(fp,"%zu",&L);
    fscanf(fp,"%zu",&max);
    if(max!=255){
            printf("P2 : max value warining : This function expects 8bit pgm file. Your file has max value:%zu \n",max);
    }
    image=(unsigned char*)malloc(sizeof(unsigned char)*W*L);
    size_t i,j;
    for(i=0;i<L;i++){
            for(j=0;j<W;j++){
                fscanf(fp,"%d",image+(i*W+j));
        }
    }
    // printf("%s",format);
    // printf("#%s",comment);
    // printf("%d %d\n",W,L);
    // printf("%d\n",max);
    // for(i=0;i<L;i++){
        //     for(j=0;j<W;j++){
        //         printf("%d ",image[i*W+j]);
    //     }
    //     printf("\n");
    // }
    *Length=L;
    *Width =W;
    *Max = max;
    fclose(fp);
    return image;
}
unsigned char *ReadPGM_P5(const char *filename, size_t *Max, size_t *Width, size_t *Length){
        /**
     * P5
     * #comment
     * Width Length
     * image data(text)......
     */
    char format[256];
    char comment[256];
    size_t W,L,max;
    unsigned char *image;
    FILE *fp;
    if((fp = fopen(filename , "rb" ))==NULL){
            fprintf(stderr,"fopen FAILURE\n");
        exit(1);
    }
    fgets(format,256,fp);
    if(format[0]!='P' || format[1]!='5'){
            fprintf(stderr,"format error. File must be pgm(P5) format. Your file is %s\n",format);
        exit(1);
    }
    int commentCountAdd1=1;
    while(1){
    		fgets(comment,256,fp);
		if(comment[0]!='#'){
    			fseek(fp, 0, SEEK_SET);
            while(commentCountAdd1--)
			    fgets(comment,256,fp);
			break;
		}
        commentCountAdd1++;
	}
    fscanf(fp,"%zu",&W);
    fscanf(fp,"%zu",&L);
    fscanf(fp,"%zu",&max);
    if(max!=255){
            printf("P5 : max value warining : This function expects 8bit pgm file. Your file has max value:%zu \n",max);
    }
    image=(unsigned char*)malloc(sizeof(unsigned char)*W*L);
    fread(image,sizeof(char),1,fp);
    fread(image,sizeof(char),W*L,fp);
    *Length=L;
    *Width =W;
    *Max = max;
    fclose(fp);
    return image;
}
unsigned char *ReadPPM_P3(const char *filename, size_t *Max, size_t *Width, size_t *Length){
        /**
     * P3
     * #comment
     * Width Length
     * image data(text)......
     */
    char format[256];
    char comment[256];
    size_t W,L,max;
    unsigned char *image;
    FILE *fp;
    if((fp = fopen(filename , "r" ))==NULL){
            fprintf(stderr,"fopen FAILURE\n");
        exit(1);
    }
    fgets(format,256,fp);
    if(format[0]!='P' || format[1]!='3'){
            fprintf(stderr,"format error. File must be pgm(P2) format. Your file is %s\n",format);
        exit(1);
    }
    int commentCountAdd1=1;
    while(1){
    		fgets(comment,256,fp);
		if(comment[0]!='#'){
    			fseek(fp, 0, SEEK_SET);
            while(commentCountAdd1--)
			    fgets(comment,256,fp);
			break;
		}
        commentCountAdd1++;
	}
    fscanf(fp,"%zu",&W);
    fscanf(fp,"%zu",&L);
    fscanf(fp,"%zu",&max);
    if(max!=255){
            printf("P2 : max value warining : This function expects 8bit pgm file. Your file has max value:%zu \n",max);
    }
    image=(unsigned char*)malloc(sizeof(unsigned char)*W*L*3);
   size_t i,j;
    for(i=0;i<L*3;i++){
            for(j=0;j<W;j++){
                fscanf(fp,"%d",image+(i*W+j));
        }
    }
    *Length=L;
    *Width =W;
    *Max = max;
    fclose(fp);
    return image;
}
unsigned char *ReadPPM_P6(const char *filename, size_t *Max, size_t *Width, size_t *Length){
        /**
     * P6
     * #comment
     * Width Length
     * image data(text)......
     */
    char format[256];
    char comment[256];
    size_t W,L,max;
    unsigned char *image;
    FILE *fp;
    if((fp = fopen(filename , "rb" ))==NULL){
            fprintf(stderr,"fopen FAILURE\n");
        exit(1);
    }
    fgets(format,256,fp);
    if(format[0]!='P' || format[1]!='6'){
            fprintf(stderr,"format error. File must be pgm(P5) format. Your file is %s\n",format);
        exit(1);
    }
    int commentCountAdd1=1;
    while(1){
    		fgets(comment,256,fp);
		if(comment[0]!='#'){
    			fseek(fp, 0, SEEK_SET);
            while(commentCountAdd1--)
			    fgets(comment,256,fp);
			break;
		}
        commentCountAdd1++;
	}
    fscanf(fp,"%zu",&W);
    fscanf(fp,"%zu",&L);
    fscanf(fp,"%zu",&max);
    if(max!=255){
            printf("P5 : max value warining : This function expects 8bit pgm file. Your file has max value:%zu \n",max);
    }
    image=(unsigned char*)malloc(sizeof(unsigned char)*W*L*3);
    fread(image,sizeof(char),1,fp);
    fread(image,sizeof(char),W*L*3,fp);
    *Length=L;
    *Width =W;
    *Max = max;
    fclose(fp);
    return image;
}
void WritePGM_P2(const char *filename, unsigned char *image, size_t Max ,size_t Width,size_t Length){
        FILE *fp;
    if((fp = fopen(filename , "w" ))==NULL){
            fprintf(stderr,"fopen FAILURE\n");
        exit(1);
    }
    char format[]="P2";
    char comment[]="";
    fprintf(fp,"%s\n",format);
    fprintf(fp,"#%s\n",comment);
    fprintf(fp,"%zu %zu\n",Width,Length);
    fprintf(fp,"%zu\n",Max);
    size_t i,j;
    for(i=0;i<Length;i++){
            for(j=0;j<Width;j++){
                fprintf(fp,"%u ",image[i*(Width)+j]);
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
}
void WritePGM_P5(const char *filename, unsigned char*image, size_t Max ,size_t Width,size_t Length){
        FILE *fp;
    if((fp = fopen(filename , "w" ))==NULL){
            fprintf(stderr,"fopen FAILURE\n");
        exit(1);
    }
    char format[]="P5";
    char comment[]="";
    fprintf(fp,"%s\n",format);
    fprintf(fp,"#%s\n",comment);
    fprintf(fp,"%zu %zu\n",Width,Length);
    fprintf(fp,"%zu\n",Max);
    fwrite(image,sizeof(char),(Width)*(Length),fp);
    fclose(fp);
}
void WritePGM_P3(const char *filename, unsigned char *image, size_t Max ,size_t Width,size_t Length){
    FILE *fp;
    if((fp = fopen(filename , "w" ))==NULL){
            fprintf(stderr,"fopen FAILURE\n");
        exit(1);
    }
    char format[]="P3";
    char comment[]="";
    fprintf(fp,"%s\n",format);
    fprintf(fp,"#%s\n",comment);
    fprintf(fp,"%zu %zu\n",Width,Length);
    fprintf(fp,"%zu\n",Max);
    size_t i;
    for(i=0;i<Length*Width*3;i++){
            //for(j=0;j<Width;j++){
                fprintf(fp,"%u ",image[i]);
        //}
        //fprintf(fp,"\n");
    }
    fclose(fp);
}
void WritePGM_P6(const char *filename, unsigned char*image, size_t Max ,size_t Width,size_t Length){
    FILE *fp;
    if((fp = fopen(filename , "w" ))==NULL){
            fprintf(stderr,"fopen FAILURE\n");
        exit(1);
    }
    char format[]="P6";
    char comment[]="";
    fprintf(fp,"%s\n",format);
    fprintf(fp,"#%s\n",comment);
    fprintf(fp,"%zu %zu\n",Width,Length);
    fprintf(fp,"%zu\n",Max);
    fwrite(image,sizeof(char),(Width)*(Length)*3,fp);
    fclose(fp);
}
#endif
