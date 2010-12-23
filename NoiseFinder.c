/** \file NoiseFinder.c
 *  Author: Abhinav S Bhatele (bhatelele@illinois.edu)
 *  Previous contributors: Authors: Eric Bohm, Chee Wai L
 *
 *  A simple benchmark to quantify computational noise
 */

#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define UNIT_WORK_FACTOR	10000
#define TAG			255
#define NUMBINS			200

int doUnitWork(int);

int main(int argc, char **argv) {
  int myrank, size;
  int i, j, pos;
  int iterations = 100;
  int iterationsOuter;
  int worksize;

  /* Statistics data */
  double sum = 0.0;
  double sum_of_squares = 0.0;
  double min = 0.0;
  double max = 0.0;
  double globalmin = 0.0;
  double globalmax = 0.0;
  double mean = 0.0;
  double variance = 0.0;
  double stddev = 0.0;
  long long smallHist[NUMBINS+1];	/* 5 us each */
  long long smallHistSum[NUMBINS+1];
  long long largeHist[NUMBINS+1];	/* 5 ms each */
  long long largeHistSum[NUMBINS+1];

  for(i=0; i<NUMBINS+1; i++) {
    smallHist[i] = 0;
    smallHistSum[i] = 0;
    largeHist[i] = 0;
    largeHistSum[i] = 0;
  }

  /* used to prevent the compiler from optimizing the delay loop away */
  double dummydata = 0.0;

  /* Timing related variables */
  double prevtime, currtime, intime = 0.0;
  double overhead = 0.0;
  double prevouterttime, outtime;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  long long noiseProcHist[size];
  long long noiseProcHistCat[size];
  for (i = 0; i < size; i++){
    noiseProcHist[i] = 0;
    noiseProcHistCat[i] = 0;
  }
  int errlen;
  char usage[100+MPI_MAX_ERROR_STRING]="Usage: NoiseFinder NumIterations Grainsize\n";

  if(argc<3) {
    printf(usage);
    fflush(stdout);
    MPI_Abort(MPI_COMM_WORLD,10);
  }
  iterationsOuter = atoi(argv[1]);
  worksize = atoi(argv[2]);

  /* Walltime overheads */
  prevtime = MPI_Wtime();
  for (i=0; i<1000; i++) {
    MPI_Wtime();
  }
  overhead = (MPI_Wtime() - prevtime)/1000;


  int donework = 0;
  struct valpair {
    double val;
    int rank;
  } oneminp, onemaxp, *minpair, *maxpair, maxdevpair, *outminpair, *outmaxpair;

  minpair = (struct valpair *) malloc (iterationsOuter * sizeof(struct valpair));
  maxpair = (struct valpair *) malloc (iterationsOuter * sizeof(struct valpair));
  outminpair = (struct valpair *) malloc (iterationsOuter * sizeof(struct valpair));
  outmaxpair = (struct valpair *) malloc (iterationsOuter * sizeof(struct valpair));

  prevouterttime = MPI_Wtime();
  globalmin = prevouterttime*1000;
  globalmax = 0.0;

  for(j=0; j<iterationsOuter; j++) {
    prevtime = MPI_Wtime();
    min = prevtime*1000.0;
    max = 0.0;
    /* outtime = 0.0; */

    for(i=0; i<iterations; i++) {
      prevtime = MPI_Wtime();
      donework = doUnitWork(worksize);
      currtime = MPI_Wtime();
      intime =  currtime - prevtime;
      /* outtime += intime; */
      sum += intime;
      sum_of_squares += intime * intime;
      min = (min > intime) ? intime : min;
      max = (max < intime) ? intime : max;
      globalmin = (globalmin > intime) ? intime : globalmin;
      globalmax = (globalmax < intime) ? intime : globalmax;
      pos = (int) (intime / 0.000005);	/* size of each bin = 5 us */
      if(pos < NUMBINS)
	smallHist[pos]++;
      else
	smallHist[NUMBINS]++;
      pos = (int) (intime / 0.005);	/* size of each bin = 5 ms */
      if(pos < NUMBINS)
	largeHist[pos]++;
      else
	largeHist[NUMBINS]++;
    }
    outtime = currtime - prevouterttime;
    prevouterttime = currtime;

    oneminp.val = min;
    oneminp.rank = myrank;
    onemaxp.val = max;
    onemaxp.rank = myrank;
    MPI_Allreduce(&oneminp, &minpair[j], 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
    MPI_Allreduce(&onemaxp, &maxpair[j], 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    oneminp.val = outtime;
    oneminp.rank = myrank;
    onemaxp.val = outtime;
    onemaxp.rank = myrank;
    MPI_Allreduce(&oneminp, &outminpair[j], 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
    MPI_Allreduce(&onemaxp, &outmaxpair[j], 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    /* Barrier is pretty redundant
    MPI_Barrier(MPI_COMM_WORLD); */
  }

  /* Compute Statistics */
  mean = sum/(iterationsOuter*iterations);
  stddev = sqrt((sum_of_squares - 2*mean*sum + mean*mean*iterationsOuter*iterations) / 
		(iterationsOuter*iterations - 1));
  onemaxp.val=stddev;
  MPI_Allreduce(&onemaxp, &maxdevpair, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
  
  noiseProcHist[myrank] = smallHist[NUMBINS];
  MPI_Reduce(&noiseProcHist, &noiseProcHistCat, size, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Reduce(&smallHist, &smallHistSum, NUMBINS+1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&largeHist, &largeHistSum, NUMBINS+1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  
  /* Print Summary */
  char name[30];
  sprintf(name, "noise_pe%d", myrank);
  FILE *outf = fopen(name, "w");
  fprintf(outf, "Rank %d : Mean = %f Min = %f Max = %f StdDev = %f\n", myrank, mean, globalmin, globalmax, stddev);
  fflush(NULL);
  fclose(outf);

  if(myrank==0)
  { /** so the compiler can't wish away your work */
     printf("donework %d\n", donework);
  }
  fflush(NULL);

  MPI_Barrier(MPI_COMM_WORLD);    
  if(myrank==0) {
    outf = fopen(name, "a");
    fprintf(outf, "Rank %d : Num Steps = %d X %d\n", myrank, iterationsOuter, iterations);
    fprintf(outf, "Rank %d : MPI_Wtime overhead per step = %f\n", myrank, overhead);
    fprintf(outf, "Rank %d : worksize %d\n", myrank, worksize);
    for(i=0;i<iterations; i++)
      fprintf(outf, "Iter[%d] max %f on rank %d min %f on rank %d\n", i, maxpair[i].val, maxpair[i].rank, minpair[i].val, minpair[i].rank);
    for(i=0;i<iterationsOuter; i++)
      fprintf(outf, "Iter[%d] outer max %f on rank %d min %f on rank %d\n", i, outmaxpair[i].val, outmaxpair[i].rank, outminpair[i].val, outminpair[i].rank);
    fprintf(outf, "Rank[%d] had max dev %f \n", maxdevpair.rank, maxdevpair.val);

    for(i=0; i<NUMBINS+1; i++)
      fprintf(outf, "smallHist %d %lld\n", i, smallHistSum[i]);
    for(i=0; i<NUMBINS+1; i++)
      fprintf(outf, "largeHist %d %lld\n", i, largeHistSum[i]);

    for(i=0; i<size; i++)
      fprintf(outf, "noiseProcHist %d %lld\n", i, noiseProcHistCat[i]);

    fclose(outf);
  }

  /*
  printf("Rank %d : Average steptime = %f\n", myrank, mean);
  printf("Rank %d : donework %d\n",myrank, donework);
  */

  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}

int doUnitWork(int repeat) {
  double dummydata = 0.0;
  int i, repeatcount;

  for (repeatcount=0; repeatcount<repeat; repeatcount++) {
    for (i=0; i<UNIT_WORK_FACTOR; i++) {
      dummydata += i*repeatcount;
    }
  }

  return (((int)dummydata)*repeat);
}

