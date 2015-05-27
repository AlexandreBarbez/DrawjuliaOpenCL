#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <OpenCL/cl.h>
#include "ocl_macros.h"
#include "../common/cpu_bitmap.h"

//Common defines
#define DEVICE_TYPE CL_DEVICE_TYPE_GPU
#define DIM 1000
#define MAX_SOURCE_SIZE (0x100000)

int main(void) {
    
    
    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;
    
    fp = fopen("/Users/Alx/Workspace/c++/DrawjuliaOpenCL/DrawjuliaOpenCL/kernel.cl", "r");
    if (!fp) {
        perror("/Users/Alx/Workspace/c++/DrawjuliaOpenCL/DrawjuliaOpenCL/kernel.cl");
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
    
    const char *kernelGpu = source_str;
    
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *ptr = bitmap.get_ptr();
    
    cl_int clStatus; //Keeps track of the error values returned.
    
    // Get platform and device information
    cl_platform_id * platforms = NULL;
    
    // Set up the Platform. Take a look at the MACROs used in this file.
    // These are defined in common/ocl_macros.h
    OCL_CREATE_PLATFORMS(platforms);
    
    // Get the devices list and choose the type of device you want to run on
    cl_device_id *device_list = NULL;
    OCL_CREATE_DEVICE(platforms[0], DEVICE_TYPE, device_list);
    
    // Create OpenCL context for devices in device_list
    cl_context context;
    
    // An OpenCL context can be associated to multiple devices, either CPU or GPU
    // based on the value of DEVICE_TYPE defined above.
    context = clCreateContext(NULL, num_devices, device_list, NULL, NULL, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateContext Failed...");
    
    // Create a command queue for the first device in device_list
    cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateCommandQueue Failed...");

    
    // Create memory buffers on the device for each vector
    cl_mem bufferJulia_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char)*DIM*DIM*4, NULL, &clStatus);
    
    // Copy the Buffer to the device. We do a blocking write to the device buffer.
    clStatus = clEnqueueWriteBuffer(command_queue, bufferJulia_clmem, CL_TRUE, 0, sizeof(unsigned char)*DIM*DIM*4, ptr, 0, NULL, NULL);
    LOG_OCL_ERROR(clStatus, "clEnqueueWriteBuffer Failed...");
    
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelGpu, NULL, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateProgramWithSource Failed...");
    
    // Build the program
    clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
    if (clStatus != CL_SUCCESS)
        LOG_OCL_COMPILER_ERROR(program, device_list[0]);
    
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "kernelGpu", &clStatus);
    
    // Set the arguments of the kernel. Take a look at the kernel definition in sum_event
    // variable. First parameter is a constant and the other three are buffers.
    clStatus |= clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bufferJulia_clmem);
 
    LOG_OCL_ERROR(clStatus, "clSetKernelArg Failed...");
    
    // Execute the OpenCL kernel on the list
    size_t global_size[2] = {DIM,DIM};
    size_t local_size = 1;
    cl_event sum_event;
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, NULL, 0, NULL, &sum_event);
    LOG_OCL_ERROR(clStatus, "clEnqueueNDRangeKernel Failed...");
    
    // Read the memory buffer C_clmem on the device to the host allocated buffer C
    // This task is invoked only after the completion of the event sum_event
    clStatus = clEnqueueReadBuffer(command_queue, bufferJulia_clmem, CL_TRUE, 0, sizeof(unsigned char)*DIM*DIM*4, ptr, 1, &sum_event, NULL);
    LOG_OCL_ERROR(clStatus, "clEnqueueReadBuffer Failed...");
    
    // Clean up and wait for all the comands to complete.
    clStatus = clFinish(command_queue);
    
    
    // Finally release all OpenCL objects and release the host buffers.
    clStatus = clReleaseKernel(kernel);
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseMemObject(bufferJulia_clmem);

    clStatus = clReleaseCommandQueue(command_queue);
    clStatus = clReleaseContext(context);
    free(platforms);
    free(device_list);
    
    bitmap.display_and_exit();
    
    return 0;
}