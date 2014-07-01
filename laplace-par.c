#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>
#include "laplace-common.h"

#define OPTION_VERBOSE "--verbose"



static void printUsage(char const * progName)
{
    fprintf(stderr, "Usage:\n"
        "    %s [--verbose] <N>\n"
        "Where:\n"
        "   <N>         The number of points in each dimension (at least 2).\n"
        "   " OPTION_VERBOSE "   Prints the input and output systems.\n",
        progName); 
}



static void freePointRow(double * currentPointRow)
{
    if (currentPointRow != NULL)
    {
        free(currentPointRow);
    }
}



static double * allocatecurrentPointRow(int numPointsPerDimension)
{
    return (double *)malloc(sizeof(double) * numPointsPerDimension);
}

/* allocate buffer of doubles of length buffSize
 * initialize it with zeroes */
static double * initializeBuffer(int buffSize)
{
    double * res;
    int i;

    res = allocatecurrentPointRow(buffSize);
    
    if (res == NULL)
    {
        free(res);
        fprintf(stderr, "ERROR: Malloc failed!\n");
        exit(1);
    }
    for(i = 0; i < buffSize; ++i)
    {
        res[i] = 0.0;
    } 

    return res;
}

static void freePoints(double * * points, int width, int myRank)
{
    if (points != NULL)
    {
        int i;
        for (i = 0; i < width; ++i)
        {
            freePointRow(points[i]);
        }
        free(points);
    }
}

static double * * allocatePoints(int numPointsPerDimension, int width, int numProcesses)
{
    double * * points;
    int i;

    points = (double * *)malloc(sizeof(double *) * width);

    if (points == NULL)
    {
        goto FAILURE;
    }
    for (i = 0; i < width; ++i)
    {
        points[i] = NULL;
    }
    for (i = 0; i < width; ++i)
    {
        points[i] = allocatecurrentPointRow(numPointsPerDimension);
        if (points[i] == NULL)
        {
            goto FAILURE;
        }
    }   
          
    return points;
  
FAILURE:
    freePoints(points, width, -1);
    return NULL;
}



static void initializePoints(double * * points, int rowFrom, int howMany, int numPointsPerDimension)
{
    int i, j;
    for (i = rowFrom; i < rowFrom + howMany; ++i)
    {
        for (j = 0; j < numPointsPerDimension; ++j)
        {
            points[i-rowFrom][j] = getInitialValue(i, j, numPointsPerDimension);
        }
    }
}
   
static void setDataBuffer(double * buff, double * * points, int dataRow, int startFrom, int numPointsPerDimension)
{
    int i;
    int idx = 0;
    for(i = startFrom; i < numPointsPerDimension - 1; i += 2)
    {
        buff[idx] = points[dataRow][i];
        idx++;
    }
}
	
/*static void setPoints(double * result, double * * points, int startRow, int width, 
        int color, int numPointsPerDimension, int row)
{
    int idx = 0;
    int ii;
    for(ii = 1 + ((startRow + width) % 2 == color ? 1 : 0); ii < numPointsPerDimension - 1; ii += 2){
        result[idx] = points[row][ii];
        idx++;
    }
}*/


static void printPoints(double * * points, int width, int numPointsPerDimension)
{
    int i, j;
    for (i = 0; i < width; ++i)
    {
        printf("%.10f", points[i][0]);
        for (j = 1; j < numPointsPerDimension; ++j)
        {
            printf(" %.10f", points[i][j]);
        }
        printf("\n");
    }
}

/*static void print1DArray(double * points, int size)
{
    int i;
    for(i = 0; i < size; ++i)
    {
        printf("%f ", points[i]);
    }
    printf("\n");
}*/

int main(int argc, char * argv[])
{
    int numPointsPerDimension;
    int verbose = 0;
    double omega;
    double epsilon;
    double * * points;
    struct timeval startTime;
    struct timeval endTime;
    double duration;
    double breakdown = 0;
    int numIterations;
    double maxDiff, tmpMaxDiff;

    int numProcesses;
    int workingProcesses;
    int myRank;
    MPI_Status status;
    MPI_Request requestUpSend, requestUpRecv;
    MPI_Request requestDownSend, requestDownRecv;
    int partitions;
    int remainder;
    int width;
    int i, k;
    int buffSize;
    int startRow;

    double * upPointsSend, * upPointsRecv;
    double * downPointsSend, * downPointsRecv;

    int upperProc, lowerProc;
    struct timeval startInterval;
    struct timeval endInterval;

    if (argc < 2)
    {
        fprintf(stderr, "ERROR: Too few arguments!\n");
        printUsage(argv[0]);
        exit(1);
    }
    else if (argc > 3)
    {
        fprintf(stderr, "ERROR: Too many arguments!\n");
        printUsage(argv[0]);
        exit(1);
    }
    else
    {
        int argIdx = 1;
        if (argc == 3)
        {
            if (strncmp(argv[argIdx], OPTION_VERBOSE, strlen(OPTION_VERBOSE)) != 0)
            {
                fprintf(stderr, "ERROR: Unexpected option '%s'!\n", argv[argIdx]);
                printUsage(argv[0]);
                exit(1);
            }
            verbose = 1;
            ++argIdx;
        }
        numPointsPerDimension = atoi(argv[argIdx]);
        if (numPointsPerDimension < 2)
        {
            fprintf(stderr, "ERROR: The number of points, '%s', should be "
                "a numeric value greater than or equal to 2!\n", argv[argIdx]);
            printUsage(argv[0]);
            exit(1);
        }
    }
    
    MPI_Init(&argc, &argv);

    /* get info about how may processes are running 
     * and what is your rank number */
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    /* calculate nominal size of data per each process */
    partitions = numPointsPerDimension / numProcesses;

    /* calculate number of processes with the additional row of data */
    remainder = numPointsPerDimension % numProcesses;

    /* according to myRank, set the width of the table */
    width = (myRank < remainder) ? partitions + 1 : partitions;
    
    /* decide how many processes are required to do the calculation */
    workingProcesses = (numProcesses > numPointsPerDimension) ? numPointsPerDimension : numProcesses;

    /* terminate processes that won't be used */
    /* start of copied part of code */
    MPI_Comm MY_WORLD = MPI_COMM_WORLD;
    if(workingProcesses < numProcesses)
    {
        MPI_Group world_group;
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);
        
        // Remove all unnecessary ranks
        MPI_Group new_group;
        int ranges[1][3] = {{workingProcesses, (numProcesses - 1), 1}};
        MPI_Group_range_excl(world_group, 1, ranges, &new_group);

        // Create a new communicator
        MPI_Comm_create(MPI_COMM_WORLD, new_group, &MY_WORLD);

        if (MY_WORLD == MPI_COMM_NULL)
        {
            // Bye bye cruel world
            MPI_Finalize();
            exit(0);
        }
    }
    /* end of copied part of code */
    /* source: http://stackoverflow.com/questions/13774968/mpi-kill-unwanted-processes */


    /* set the calculation parameters */
    omega = getOmega(numPointsPerDimension);
    epsilon = getEpsilon(numPointsPerDimension);
    
    /* allocate points table for each process */
    points = allocatePoints(numPointsPerDimension, width, numProcesses);
    if (points == NULL)
    {
        freePoints(points, width, myRank);
        fprintf(stderr, "ERROR: Malloc failed!\n");
        exit(1);
    }
    
    /* size of the table to send per each iteration */
    buffSize = numPointsPerDimension / 2 + numPointsPerDimension % 2 ;
    
    /* initialize additional buffers for communication */
    upPointsSend = initializeBuffer(buffSize);
    upPointsRecv = initializeBuffer(buffSize);
    downPointsSend = initializeBuffer(buffSize);
    downPointsRecv = initializeBuffer(buffSize);
    
    /* process #0 sends to others separate parts of the table
     * others wait for incoming data */
    if (myRank == 0)
    { 
        startRow = numPointsPerDimension;
        for(k = workingProcesses - 1; k >= 0 ; --k)
        {
            width = (k < remainder) ? partitions + 1 : partitions;
            
            /* initialize points */
            initializePoints(points, startRow - width, width, numPointsPerDimension);
        
            /* send table to k-th process */
            if(k != 0)
            {
                for(i = 0; i < width; ++i)
                {
                    MPI_Send(points[i], numPointsPerDimension, MPI_DOUBLE, k, 123, MY_WORLD);
                }
            }
            startRow -= width;
        }        
    } 
    else 
    {
        if(myRank < workingProcesses)
        {
            for(i = 0; i < width; ++i)
            {
                MPI_Recv(points[i], numPointsPerDimension, MPI_DOUBLE, 0, 123, MY_WORLD, &status);
            }
        }
    }

    /* remember with which processes you comunicate */ 
    upperProc = myRank == 0 ? MPI_PROC_NULL : myRank - 1;
    lowerProc = myRank == workingProcesses - 1 ? MPI_PROC_NULL : myRank + 1;
    
    /* here each process has it's own data set for computations */

    if(remainder > 0)
    {
        startRow = (myRank < remainder) ? myRank * (partitions + 1) : myRank * partitions + remainder;
    }
    else 
    {
        startRow = myRank * partitions;
    }

    if(gettimeofday(&startTime, NULL))
    {
        freePoints(points, width, myRank);
        fprintf(stderr, "ERROR: Gettimeofday failed!\n");
        exit(1);
    }
    
    /* Start of computations. */
    
    numIterations = 0;
    do
    {
        int i, j, color;
        maxDiff = 0.0;
        for (color = 0; color < 2; ++color)
        {

            /* fill downPointsSend with the last row of points data */
            setDataBuffer(downPointsSend, points, width - 1, 1 + ((startRow + width) % 2 == color ? 1 : 0), numPointsPerDimension);

            if(gettimeofday(&startInterval, NULL))
	    {
		        freePoints(points, width, myRank);
		        fprintf(stderr, "ERROR: Gettimeofday failed!\n");
		        exit(1);
	    }
            
            MPI_Isend(downPointsSend, buffSize, MPI_DOUBLE, lowerProc, color, MY_WORLD, &requestDownSend);
            MPI_Irecv(downPointsRecv, buffSize, MPI_DOUBLE, lowerProc, color, MY_WORLD, &requestDownRecv);
            
            if(gettimeofday(&endInterval, NULL))
	    {
		        freePoints(points, width, myRank);
		        fprintf(stderr, "ERROR: Gettimeofday failed!\n");
		        exit(1);
	    }
	
	    breakdown += ((double)endInterval.tv_sec + ((double)endInterval.tv_usec / 1000000.0)) - 
	                 ((double)startInterval.tv_sec + ((double)startInterval.tv_usec / 1000000.0));

            /* fill upPointsSend with the last row of points data */
            setDataBuffer(upPointsSend, points, 0, 1 + ((startRow - 1) % 2 == color ? 1 : 0), numPointsPerDimension);

	    if(gettimeofday(&startInterval, NULL))
	    {
		        freePoints(points, width, myRank);
		        fprintf(stderr, "ERROR: Gettimeofday failed!\n");
		        exit(1);
	    }

	    MPI_Isend(upPointsSend, buffSize, MPI_DOUBLE, upperProc, color, MY_WORLD, &requestUpSend);
            MPI_Irecv(upPointsRecv, buffSize, MPI_DOUBLE, upperProc, color, MY_WORLD, &requestUpRecv);
			
            /* computations of the first row requires data that has to be recieved from other process */
	    MPI_Wait(&requestUpRecv, &status);
	    
	    if(gettimeofday(&endInterval, NULL))
	    {
			freePoints(points, width, myRank);
	       	    	fprintf(stderr, "ERROR: Gettimeofday failed!\n");
		    	exit(1);
            }
	    breakdown += ((double)endInterval.tv_sec + ((double)endInterval.tv_usec / 1000000.0)) - 
                 	((double)startInterval.tv_sec + ((double)startInterval.tv_usec / 1000000.0));


            for (i = 0; i < width; ++i)
            {
 
                /* before computing the last row of its data, 
                 * process has to be sure that it has required
                 * row from process rank+1 */
		if(i == width - 1)
                {
                	
			if(gettimeofday(&startInterval, NULL))
	    		{
		        	freePoints(points, width, myRank);
		        	fprintf(stderr, "ERROR: Gettimeofday failed!\n");
		        	exit(1);
	    		}

	    
			MPI_Wait(&requestDownRecv, &status);
	    
	    		if(gettimeofday(&endInterval, NULL))
	    		{
				freePoints(points, width, myRank);
	       	    		fprintf(stderr, "ERROR: Gettimeofday failed!\n");
		    		exit(1);
            		}
			breakdown += ((double)endInterval.tv_sec + ((double)endInterval.tv_usec / 1000000.0)) - 
                     		((double)startInterval.tv_sec + ((double)startInterval.tv_usec / 1000000.0));

		}

                for (j = 1 + ((startRow+i) % 2 == color ? 1 : 0); j < numPointsPerDimension - 1; j += 2)
                {
                    if( (myRank != 0 || i != 0 ) && (myRank != workingProcesses - 1 || i != width - 1) )
                    {
                        
                        double tmp, diff;
                        double down, up;
                        int jIdx = (j - 1 - ((startRow + i) % 2 == color ? 1 : 0))/ 2;
                        
                        /* decide if up or down value should be taken from additional buffers */
                        up = (i == 0) ? upPointsRecv[jIdx] : points[i-1][j];
                        down = (i == width - 1) ? downPointsRecv[jIdx] : points[i+1][j];
                        
                        /* calculate final value */
                        tmp = (up + down + points[i][j - 1] + points[i][j + 1]) / 4.0;
                        diff = points[i][j];
                        points[i][j] = (1.0 - omega) * points[i][j] + omega * tmp;
                        
                        diff = fabs(diff - points[i][j]);
                        if (diff > maxDiff)
                        {
                            maxDiff = diff;
                        }
                    }
                }
            }
	    if(gettimeofday(&startInterval, NULL))
	    {
	    	freePoints(points, width, myRank);
	       	fprintf(stderr, "ERROR: Gettimeofday failed!\n");
		exit(1);
	    }

            MPI_Barrier(MY_WORLD);
	    if(gettimeofday(&endInterval, NULL))
	    {
		freePoints(points, width, myRank);
	      	fprintf(stderr, "ERROR: Gettimeofday failed!\n");
	  	exit(1);
            }
	    breakdown += ((double)endInterval.tv_sec + ((double)endInterval.tv_usec / 1000000.0)) - 
                     ((double)startInterval.tv_sec + ((double)startInterval.tv_usec / 1000000.0));

	
        }
    	
        if(gettimeofday(&startInterval, NULL))
        {
            freePoints(points, width, myRank);
            fprintf(stderr, "ERROR: Gettimeofday failed!\n");
            exit(1);
        }
        
        /* find new maxDiff among all processes */
        MPI_Allreduce(&maxDiff, &tmpMaxDiff, 1, MPI_DOUBLE, MPI_MAX, MY_WORLD );
        maxDiff = tmpMaxDiff;

        if(gettimeofday(&endInterval, NULL))
        {
            freePoints(points, width, myRank);
            fprintf(stderr, "ERROR: Gettimeofday failed!\n");
                exit(1);
        }
        
        breakdown += ((double)endInterval.tv_sec + ((double)endInterval.tv_usec / 1000000.0)) - 
                     ((double)startInterval.tv_sec + ((double)startInterval.tv_usec / 1000000.0));

        ++numIterations;
    }
    while (maxDiff > epsilon);

    /* End of computations. */
 
    if(gettimeofday(&endTime, NULL))
    {
        freePoints(points, width, myRank);
        fprintf(stderr, "ERROR: Gettimeofday failed!\n");
        exit(1);
    }

    /* calculate how long did the computation lasted */
    duration =
        ((double)endTime.tv_sec + ((double)endTime.tv_usec / 1000000.0)) - 
        ((double)startTime.tv_sec + ((double)startTime.tv_usec / 1000000.0));

    /* we choose the process whose execution lasted for the longest time */     
    double maxDuration;
    MPI_Allreduce(&duration, &maxDuration, 1, MPI_DOUBLE, MPI_MAX, MY_WORLD);
   
    if(myRank==0)
    {
        fprintf(stderr,
            "Statistics: duration(s)=%.10f breakdown=%.10f #iters=%d diff=%.10f epsilon=%.10f\n",
            maxDuration, breakdown, numIterations, maxDiff, epsilon);
    }
  
    if (verbose) {
        
        MPI_Barrier(MY_WORLD);
    
        /* process #0 is responsible for printing results of computation 
         * others send their data straight to it */
        if(myRank != 0 && myRank < workingProcesses) 
        {
            for(k = 0; k < width ; ++k)
            {
                MPI_Send(points[k], numPointsPerDimension, MPI_DOUBLE, 0, 123, MY_WORLD);
            }
        }
        else if(myRank == 0)
        {
            printPoints(points, width, numPointsPerDimension);
            for(i = 1; i < workingProcesses; ++i)
            {
                width = (i < remainder) ? partitions + 1 : partitions;
                
                for (k = 0 ; k < width ; ++k)
                {
                    MPI_Recv(points[k], numPointsPerDimension, MPI_DOUBLE, i, 123, MY_WORLD, &status);
                }

                printPoints(points, width, numPointsPerDimension);
            }
        }
    }
    
    /* free all the memory that was allocated */
    freePoints(points, width, myRank);
    free(downPointsSend);
    free(upPointsSend);
    free(downPointsRecv);
    free(upPointsRecv);
     
    MPI_Finalize();
    
    return 0;
}

