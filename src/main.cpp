//============================================================================
// Name        : main.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <cv.h>
#include <highgui.h>
#include "klt.hpp"
#include "kltTest.hpp"
#include "imageConversion.hpp"
#include "cluster.hpp"
#include "tracker.hpp"
#include "kinectCapturer.h"
#include "StereoCapturer.h"
#include "MonoCamera.h"
#include "../AContrario3D/include/AContrario.hpp"

#define mindist 10
//#define lastnum 863
#define lastnum 179

using namespace klt;
using namespace cluster;

//using namespace tracker;

// PRUEBAS ACTUALES
//#if 0
int main(void)
{
	int plIm, movF=0, pMov=0, modu=4, ini=1; //read the ini position in the table, pMov: moving points in total
	int c=0, init=140;
	int c4= init + modu;
	int nF = 100, nFaux=0, sizeMaskProb = mindist*2+1;
	double dt=0.1;
	bool _3d = true, _kinect = true;

	/***************************************************************/
	/********** Initialization and Reading of Images****************/
	/***************************************************************/

	char nbuffer[90];

	cv::Mat image, imageprec, imagecopy, imaR, *gray_image;
	capture::Capturer *capturer;
	uchar *mask, *maskprob;
	int iC, iR, pasIma=1;
	int iniTrack=0;  // Flag to init tracking object

	//sprintf(buffer,"../images/image.l.%05i.tiff",firstnum);
	//sprintf(buffer,"/home/luzdora/images/TriShop1cor/serie01/images/image.%04i.jpg",init);
	//sprintf(buffer,"../images/RGB_Dani_1_%i.tiff",init);
	//sprintf(buffer,"/home/luzdora/images/palomitaRGB/-000000%04i.ppm",init);



	if(_3d)
		if(_kinect)
			//capturer = new capture::kinectCapturer("../../../../../TestsTesis/imagesDani2/colortest1_","tiff","../../../../../TestsTesis/imagesDani2/positiontest1_","M3D",10,73,1); //Dani Fantasma
			capturer = new capture::kinectCapturer("../../../../../TestsTesis/images2Dani/colortest1_","tiff","../../../../../TestsTesis/images2Dani/positiontest1_","M3D",0,62,1); // 2 Dani Fantasma
			//capturer = new capture::kinectCapturer("../../../../../TestsTesis/imagesNoObject/colortest1_","tiff","../../../../../TestsTesis/imagesNoObject/positiontest1_","M3D",0,23,1); //No objetos
		else //Stereo Vision
			//capturer = new capture::StereoCapturer("../../../../../TestsTesis/singleDaniStereo/images/testStereoNearRightRectified","png","../../../../../TestsTesis/singleDaniStereo/disparity/depthImage","DMX",10,213,3,"intrinsics.yml","extrinsics.yml"); //Dani solo estereo
			//capturer = new capture::StereoCapturer("../../../../../TestsTesis/stereo2Dani1/images/testStereoNearRightRectified","png","../../../../../TestsTesis/stereo2Dani1/disparity/depthImage","DMX",292,416,2,"intrinsics.yml","extrinsics.yml"); // 2 Dani estereo 1
			capturer = new capture::StereoCapturer("../../../../../TestsTesis/stereo2Dani2/images/testStereoNearRightRectified","png","../../../../../TestsTesis/stereo2Dani2/disparity/depthImage","DMX",255,367,2,"intrinsics.yml","extrinsics.yml"); // 2 Dani estereo 2
	else // MonoCamera
	capturer = new capture::MonoCamera("../../../../../TestsTesis/imagesDani2/colortest1_","tiff",10,50,1); //Dani Fantasma

	if(!capturer->initilize(0.0666f))
		return 0;

	if(capturer->get_frame(image,&dt)!=capture::CAPTURE_SUCESSFUL)
		return 0;

	//Image Pre-Processing
	imagecopy = image;
	gray_image = klt::convertImage(image);
	FILE *imagefile;
	/*** Initialize image dimensions **/
	iC=gray_image->cols;
	iR=gray_image->rows;

	//cv::namedWindow("treated frame",CV_WINDOW_AUTOSIZE);
	//cv::namedWindow("Result" , CV_WINDOW_AUTOSIZE);
	//cv::namedWindow("Boxes" , CV_WINDOW_AUTOSIZE);

	//cv::imshow("treated frame",image);


	/***************************************************************/
	/********* Initialization of KLT class *************************/
	/***************************************************************/
	klt::Klt *track_er;
	KLT_FeatureList fl;
	KLT_TrackingContext tc;
	track_er = new klt::Klt();
	tc = track_er->CreateTrackingContext();
	initTrackContext(tc,2);
	fl = track_er->CreateFeatureList(nF);
	mask = reserveOccupationMask(iC, iR);
	maskprob = reserveOccupationMask(sizeMaskProb,sizeMaskProb);
	initialiseProbaMask(maskprob,sizeMaskProb,mindist);
	initialiseOccupationMask(mask,maskprob,iC,iR);

	/***************************************************************/
	/********* Initialization of cluster class *********************/
	/***************************************************************/

	cluster::AContrario clust_er;

	/***************************************************************/
	/********* Initialization of object *************************/
	/***************************************************************/
	int initialization=0, dim=6, extradim = 0;
	int snkFlag=0; // 0: No snake process

	if(_3d)
		dim = 10;

	mat xp(1,dim);
	mat obfus(1,dim);
	tracker::lstclas *trOb;
	trOb = new tracker::lstclas();

	/***************************************************************/
	/********* Init Kalman Matrix and Vector*************************/
	/***************************************************************/
	vec X0(4);
	vec Z(2);
	vec R(2);
	mat P0(4,4);
	mat F(4,4);
	mat H(2,4);

	H(0,0) = 1; H(0,1) = 0; H(0,2) = 0; H(0,3) = 0;
	H(1,0) = 0; H(1,1) = 1; H(1,2) = 0; H(1,3) = 0;

	F(0,0) = 1; F(0,1) = 0; F(0,2) = dt; F(0,3) = 0;
	F(1,0) = 0; F(1,1) = 1; F(1,2) = 0; F(1,3) = dt;
	F(2,0) = 0; F(2,1) = 0; F(2,2) = 1; F(2,3) = 0;
	F(3,0) = 0; F(3,1) = 0; F(3,2) = 0; F(3,3) = 1;

	/*********Initialize timer*********************************/
	boost::posix_time::ptime startTime, stopTime;
	startTime = boost::posix_time::microsec_clock::local_time();

	/********* Select Initial Features*************************/
	track_er->SelectGoodFeatures(tc,*gray_image,fl,-1.0,0,mask,maskprob);

	/********* Project to 3D if needed*************************/
	capturer->projectFeatureList(fl);

	/********* Save Initial Features***************************/
	track_er->StoreFeatureList(fl);
	track_er->StoreFeatureList_Trail(fl);
	track_er->WriteFeatureListToImg(fl,imagecopy);


	//********Feature Table and Feature Map display**********//
	//cv::Mat pMaskImage(iR, iC, CV_8UC1,mask);
	//cv::namedWindow("Probability Mask",cv::WINDOW_AUTOSIZE);
	//cv::imshow("Probability Mask",pMaskImage);
	//cv::Mat pMaskImage2(sizeMaskProb, sizeMaskProb, CV_8UC1,maskprob);
	//cv::namedWindow("Probability Mask2",cv::WINDOW_AUTOSIZE);
	//cv::imshow("Probability Mask2",pMaskImage2);

	//********First Feature points display**********//
	//cv::imshow("Result",imagecopy);
	//cv::waitKey(0);
	/***************************************************************/
	/********   Init KLT tracker, cluster process and MOT  *********/
	/***************************************************************/

	try{
		for(int i=init+pasIma;;i++)
		{
			// Reading new image
			//sprintf(buffer,"../images/image.l.%05i.tiff",init+i);
			//sprintf(buffer,"/home/luzdora/images/TriShop1cor/serie01/images/image.%04i.jpg",i);
			//sprintf(buffer,"/home/luzdora/images/palomitaRGB/-000000%04i.ppm",i);

			imageprec = *gray_image;

			if(capturer->get_frame(image,&dt)!=capture::CAPTURE_SUCESSFUL)
				break;

			imagecopy = image;
			gray_image = klt::convertImage(image);

			//************** SHOW IMAGE *****************
			//imshow("treated frame",*gray_image);
			// ****************** TRACKING OF FEATURES ******************

			track_er->TrackFeatures(tc,imageprec,*gray_image,fl,1,mask,maskprob,dt);
			capturer->projectFeatureList(fl);
			track_er->featuresVelocity(fl,dt,0,_3d);
			plIm = countMovingTrackingFeatures(fl);
			movF+=plIm;
			std::cout << "Count Moving Features " << plIm << std::endl;
			track_er->StoreFeatureList(fl);
			//	track_er->StoreFeatureList_Trail(fl);   por el momento no le veo utilidad

			/*********** THIS CODE WRITE FEATURES ON THE IMAGE AND SHOW IT ******/
			/*** generate result image ****/

			//CvSize imaSize = cvSize(iR, iC);
			//imaR = cv
			//imaR = cvCreateMat(iR, iC, CV_8UC1);
			imaR = imagecopy.clone();
			track_er->WriteFeatureListToImg(fl,imagecopy);
			//KLTwriteImageToImg(imagecopy, imaR);  //imaR *gray_image
			//cvSaveImage(nbuffer, image.data);
			//****************************Show image ande feature points**************//
			//imshow("Result", imagecopy);
			//cv::waitKey(0);

			//	cvSaveImage(nbuffer, image.data);

			/***************************************************************/
			/********   Tracking Section  *********/
			/***************************************************************/
			if (iniTrack>0){

				nFaux = nF - objectTracking(*trOb, tc, imageprec, *gray_image, Z, H, R, iC, iR, imaR, mask, maskprob, dt, pasIma);
				iniTrack = trOb->totalClusters;
				if(iniTrack>0)
					tracker::deleteObjectZone(*trOb, mask, maskprob,tc, iC, iR, pasIma);
				else
					initialization = 0;
				//imshow("Boxes", imaR);
				//cv::waitKey(0);
			}


			/***************************************************************/
			/******   Clustering Evaluation Only if it is the 4th image ****/
			/***************************************************************/

			if (i==c4)
			{
				//********Display Feature table**********//
//				cv::Mat testImage(imagecopy.size(),CV_8UC3,cv::Scalar(255,255,255));
//				track_er->writeFeatureTableToImg(track_er->ft,track_er->curFrame,testImage);
//				cv::namedWindow("Feature Table",CV_WINDOW_AUTOSIZE);
//				imshow("FeatureTable", testImage);
//				cv::waitKey(0);
				pMov = countMovingFeaturesTotal(fl,modu); // moving points in complete sequence

				// If there is a minimal quantity of points then cluster it
				std::cout << "Number of moving points in total " << pMov << std::endl;
				if (movF>5 && pMov>0){
					movF=track_er->ExtractMovingPoints(xp, ini, modu, fl,_3d);

					//c= clusteringTrackingPoints(xp, iC, iR); //Old Method
					c = clust_er.clusteringTrackingPoints(xp,4,4,0.02,10,0.01); //New Method
					//std::cout << "Number of clusters " << c << std::endl;
					WriteClustersToImg( xp , imagecopy ,c);
					cv::namedWindow(nbuffer,CV_WINDOW_AUTOSIZE);
					//imshow(nbuffer, imagecopy);
					//cv::waitKey(0);
					//c1 = c;
					if (c>0 && plIm>1){
						if (initialization == 0){
							plIm = initTrackingObjects(*trOb, xp, X0, P0, iC, iR, tc, 0, c, plIm, dt, snkFlag);
							if (plIm>0){ //total number of object points found
								c= trOb->totalClusters;
								std::cout << "Number of objects initialization " << c << std::endl;
								iniTrack = c;
								setOccupationZoneInMask(mask, maskprob, fl, iC, iR, mindist, 0);
								nFaux= nF - plIm;
								//fl=track_er->ResizeFeatureList(fl, nFaux);
								tracker::kalmanFilterPrediction(*trOb, dt, iC, iR, tc, imaR, mask);
							}
							c=0;
							initialization=plIm;
						}
						// Starting fusion evaluation of clusters
						if(c>0){
							initialiseOccupationMask(mask, maskprob, iC,iR);
							obfus.resize(nF-nFaux,dim);
							// c is the number of clusters with points detected
							c=jointObjectData(xp,*trOb,obfus,c,iniTrack+c,plIm);

							if(c>=2){ // only if 2 or more clusters were found
								std::cout << "Number of clusters " << c << "Number of clusters before"<< iniTrack<< std::endl;
								c=tracker::mergingObjects(obfus, c, iniTrack);
								std::cout << "Number of clusters after merging" << c << std::endl;

								if (c>0){ // New merged objects were found
									plIm=initTrackingObjects(*trOb, obfus,X0,P0,iC,iR,tc,iniTrack,c,obfus.size1(),dt,snkFlag);
									iniTrack=trOb->totalClusters;
									std::cout << "Number of objects" << iniTrack << std::endl;
									// 1 indicate clusters that do not move
									setOccupationZoneInMask(mask, maskprob,fl,iC,iR, mindist, 1);
									nFaux=nF-plIm;
									//fl=track_er->ResizeFeatureList(fl,nFaux); No entiendo porque resize fl??
									kalmanFilterPrediction(*trOb,dt,iC,iR,tc,imaR,mask);
									//		tracker::deleteObjectZone(*trOb,mask,maskprob,tc,iC,iR,plIm);
								}
								else{
									std::cout << "There is not new objects by merging" << std::endl;
									setOccupationZoneInMask(mask, maskprob, fl,iC, iR, mindist, 1);
								}
							}
							else{ // Less of 2 clusters were found
								std::cout << "There is not 2 clusters at least, no fussion is possible" << std::endl;
								setOccupationZoneInMask(mask, maskprob, fl,iC, iR, mindist, 1);
							}
						}
						std::cout << "Delete cluster zone and select new features" << std::endl;
						deleteObjectZone(*trOb, mask,maskprob,tc,iC,iR,plIm);
						track_er->SelectGoodFeatures(tc,*gray_image,fl,-1,0,mask,maskprob);
						capturer->projectFeatureList(fl);
						track_er->StoreFeatureList(fl);
						track_er->StoreFeatureList_Trail(fl);
						/******************* AQUI VA CODIGO PARA VISUALIZACION EN IMAGEN RESULTANTE *************/

					} else {
						std::cout << "THERE IS NO CLUSTERS OR INTERESTING POINTS" << std::endl;
						initialiseOccupationMask(mask, maskprob, iC,iR);
						deleteObjectZone(*trOb, mask,maskprob,tc,iC,iR,plIm);
						setOccupationZoneInMask(mask, maskprob, fl,iC, iR, mindist, 0);
						track_er->SelectGoodFeatures(tc,*gray_image,fl,-1,0,mask,maskprob);
						capturer->projectFeatureList(fl);
						track_er->StoreFeatureList(fl);
						track_er->StoreFeatureList_Trail(fl);
					}
				} else {
					std::cout << "NO MINIMAL MOBILE POINTS, FEATLIST TO MASK" << std::endl;
					setOccupationZoneInMask(mask, maskprob, fl,iC, iR, mindist, 1);
					track_er->SelectGoodFeatures(tc,*gray_image,fl,-1,0,mask,maskprob);
					capturer->projectFeatureList(fl);
					track_er->StoreFeatureList(fl);
					track_er->StoreFeatureList_Trail(fl);
				}

				ini = i-init+1;
				if(ini>=96){
					init += 96;
					ini= i -init+1;
				}
				c4 = i + modu;
				movF=0;
				//cv::imshow("KalmanPrediction",imaR);
				//cv::waitKey(0);
				/****** SALVAR RESULTADOS EN IMAGEN ACCUMULADO *******/
			}  /********   END OF 4TH IMAGE EVALUATION ****/


			/***************************************************************/
			/********   SAVE IMAGE FEATURE LIST ****/
			/***************************************************************/
			// SAVE FEATURE LIST IN ACCUMULATE IMAGE
			//sprintf(nbuffer,"imagen%i.jpg",i);
			// COMO SALVAR LA IMAGEN!!! imaR
			//imagefile = fopen(nbuffer, "wb");
			//if (imagefile == NULL)
			{
			//	perror ("Can't create image file");
			//	exit(1);
			}

			//fprintf(imagefile, "P6\n%u %u\n 255\n", iC, iR);
			//fwrite(imagecopy.data, 1, iR*iC*3, imagefile);
			//sprintf(nbuffer,"imagen%i.ppm",i);
			//cv::imwrite(nbuffer,imagecopy);
			//fclose(imagefile);
			//printf("Wrote image file \n");
			//cv::waitKey(0);
		} // end of image sequence reader
		// SAVE FEATURE TABLE
		sprintf(nbuffer,"Points3D.txt");
		track_er->WriteFeatureTableReduced(nbuffer, "%6.2f",_3d);
		sprintf(nbuffer,"Points.txt");
		track_er->WriteFeatureTable(nbuffer, "%5.1f",_3d);

		printf("escritura de archivo \n");
		clust_er.writeTrackingPointstoFile(xp,"Data3dexample.txt");
		stopTime = boost::posix_time::microsec_clock::local_time();
		boost::posix_time::time_duration dur = stopTime - startTime;
		double milliseconds = dur.total_milliseconds();
		cout<<"Time Elpased "<<1000/milliseconds<<" milliseconds"<<endl;
	}
	catch(char * str)
	{
		printf("%s",str);
	}

	catch(char const* str)
	{
		cerr<<str<<endl;
	}
	cv::waitKey(0);

	track_er->FreeFeatureList(fl);
	track_er->FreeTrackingContext(tc);
	freeOccupationMask(mask);
	freeOccupationMask(maskprob);
	delete track_er;
	delete gray_image;
	tracker::eraserlstclass(*trOb);
	delete trOb;
	printf("fin\n");
	return 0;
}
//#endif

