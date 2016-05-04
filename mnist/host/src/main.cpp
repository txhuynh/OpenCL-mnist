// Copyright (C) 2013-2015 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

///////////////////////////////////////////////////////////////////////////////////
// This host program executes a matrix multiplication kernel to perform:
//  C = A * B
// where A is a N x K matrix, B is a K x M matrix and C is a N x M matrix.
// All dimensions must be a multiple of BLOCK_SIZE, which affects the
// underlying kernel.
//
// This host program supports partitioning the problem across multiple OpenCL
// devices if available. If there are M available devices, the problem is
// divided so that each device operates on N/M rows (with
// processed by each device is . The host program
// assumes that all devices are of the same type (that is, the same binary can
// be used), but the code can be generalized to support different device types
// easily.
//
// Verification is performed against the same computation on the host CPU.
///////////////////////////////////////////////////////////////////////////////////

#ifndef _main_cpp
#define _main_cpp

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include <iostream>
#include "dnn.h"
#include "main.h"
#include "matrixMult.h"

using namespace aocl_utils;

// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> device; // num_devices elements
cl_context context = NULL;
scoped_array<cl_command_queue> queue; // num_devices elements
cl_program program = NULL;
scoped_array<cl_kernel> kernel; // num_devices elements
scoped_array<cl_mem> input_a_buf; // num_devices elements
scoped_array<cl_mem> input_b_buf; // num_devices elements
scoped_array<cl_mem> output_buf; // num_devices elements

// Problem data.
unsigned A_height = 1 * BLOCK_SIZE; //fill in 0's for other rows
unsigned A_width  = 13 * BLOCK_SIZE; //64 * 12 = 768
const unsigned &B_height = A_width;
unsigned B_width  =  1 * BLOCK_SIZE; //fill in 0's for other cols
const unsigned &C_height = A_height;
const unsigned &C_width  = B_width;

scoped_array<scoped_aligned_ptr<float> > input_a; // num_devices elements
scoped_aligned_ptr<float> input_b;
scoped_array<scoped_aligned_ptr<float> > output; // num_devices elements
scoped_array<float> ref_out; // num_devices elements
scoped_array<unsigned> m_per_device; // num_devices elements

LayerDefinition *layerDefs;
Network *nn;

// Function prototypes
bool init_opencl();

/**
 * @brief Trains a network on the MNIST training set
 * @details Trains the network by feeding input, calculating and backpropaging the error, updating weights
 * @param nn A pointer to the network
 */

void trainNetwork(Network *nn){
    
    // open MNIST files
    FILE *imageFile, *labelFile;
    imageFile = openMNISTImageFile(MNIST_TRAINING_SET_IMAGE_FILE_NAME);
    labelFile = openMNISTLabelFile(MNIST_TRAINING_SET_LABEL_FILE_NAME);
    
    int errCount = 0;
    
    // Loop through all images in the file
    for (int imgCount=0; imgCount<MNIST_MAX_TRAINING_IMAGES; imgCount++){
        
        // Reading next image and its corresponding label
        MNIST_Image img = getImage(imageFile);
        MNIST_Label lbl = getLabel(labelFile);
        
        // Convert the MNIST image to a standardized vector format and feed into the network
        Vector *inpVector = getVectorFromImage(&img);
        feedInput(nn, inpVector);

        // Feed forward all layers (from input to hidden to output) calculating all nodes' output
        feedForwardNetwork(nn);

        // Back propagate the error and adjust weights in all layers accordingly
        backPropagateNetwork(nn, lbl);

        // Classify image by choosing output cell with highest output
        int classification = getNetworkClassification(nn);
        if (classification!=lbl) errCount++;

        // Display progress during training
        //displayTrainingProgress(imgCount, errCount);
        if (imgCount % 10 == 0) //TODO: delete
          printf("Images trained = %d\n", imgCount);
    }
    
    // Close files
    fclose(imageFile);
    fclose(labelFile);
    
}




/**
 * @brief Tests an already trained network on the MNIST testing set
 * @details Follows same steps as training process but without backpropagation and updating weights
 * @param nn A pointer to the network
 */

void testNetwork(Network *nn){
    
    // open MNIST files
    FILE *imageFile, *labelFile;
    imageFile = openMNISTImageFile(MNIST_TESTING_SET_IMAGE_FILE_NAME);
    labelFile = openMNISTLabelFile(MNIST_TESTING_SET_LABEL_FILE_NAME);
    
    int errCount = 0;
    
    // Loop through all images in the file
    for (int imgCount=0; imgCount<MNIST_MAX_TESTING_IMAGES; imgCount++){
        
        // Reading next image and its corresponding label
        MNIST_Image img = getImage(imageFile);
        MNIST_Label lbl = getLabel(labelFile);
        
        // Convert the MNIST image to a standardized vector format and feed into the network
        Vector *inpVector = getVectorFromImage(&img);
        feedInput(nn, inpVector);
        
        // Feed forward all layers (from input to hidden to output) calculating all nodes' output
        feedForwardNetwork(nn);
        
        // Classify image by choosing output cell with highest output
        int classification = getNetworkClassification(nn);
        if (classification!=lbl) errCount++;
        
        // Display progress during testing
        //displayTestingProgress(imgCount, errCount);
    }
    printf("Testing: imgCount = %d, errCount = %d\n", MNIST_MAX_TESTING_IMAGES, errCount);
    // Close files
    fclose(imageFile);
    fclose(labelFile);
    
}

// Entry point.
int main(int argc, char **argv) {
  Options options(argc, argv);

  // Spot check matrix sizes. They all must be a multiple of BLOCK_SIZE,
  // although it is relatively straightforward to handle non-multiples
  // by adding padding. For simplicity, this example does not pad.
  if((A_height % BLOCK_SIZE) != 0 || (A_width % BLOCK_SIZE) != 0 ||
     (B_height % BLOCK_SIZE) != 0 || (B_width % BLOCK_SIZE) != 0 ||
     (C_height % BLOCK_SIZE) != 0 || (C_width % BLOCK_SIZE) != 0) {
    printf("Matrix sizes must be a multiple of %d.\n", BLOCK_SIZE);
    return -1;
  }

 // Initialize OpenCL.
  if(!init_opencl()) {
    return -1;
  }

  int numberOfLayers = 2;
     LayerDefinition inputLayer = {
         INPUT,
         NONE,
         (Volume){MNIST_IMG_WIDTH, MNIST_IMG_HEIGHT},
         0
     };
 
     LayerDefinition outputLayer = {
         OUTPUT,
         RELU,
         (Volume){10}
     };
   
    // Create an array to hold all of the above layer definitions (for easier reference throught the code)
    layerDefs = setLayerDefinitions(numberOfLayers, inputLayer, outputLayer);
    
    // Display details of the network definition/architecture on the screen
    // outputNetworkDefinition(numberOfLayers, layerDefs);
    
    // Create a neural network based on the above definition
    nn = createNetwork(numberOfLayers, layerDefs);
    
    // Define additional hyper-parameters (optional)
    nn->learningRate = 0.0004;

    // Train the network
    trainNetwork(nn);
    
    // Test the network
    testNetwork(nn);
    
    // Free the manually allocated memory for this network
    free(nn);
    free(layerDefs);

  // Free the resources allocated
  cleanup2();

  return 0;
}

/////// HELPER FUNCTIONS ///////

// Initializes the OpenCL objects.
bool init_opencl() {
  cl_int status;

  printf("Initializing OpenCL\n");

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  //platform = findPlatform("Altera");
  platform = findPlatform("Altera");
  if(platform == NULL) {
    printf("ERROR: Unable to find Altera OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL device.
  device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
  printf("Platform: %s\n", getPlatformName(platform).c_str());
  printf("Using %d device(s)\n", num_devices);
  for(unsigned i = 0; i < num_devices; ++i) {
    printf("  %s\n", getDeviceName(device[i]).c_str());
  }

  // Create the context.
  context = clCreateContext(NULL, num_devices, device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the program for all device. Use the first device as the
  // representative device (assuming all device are of the same type).
  std::string binary_file = getBoardBinaryFile("mnist", device[0]);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Create per-device objects.
  queue.reset(num_devices);
  kernel.reset(num_devices);
  m_per_device.reset(num_devices);
  input_a_buf.reset(num_devices);
  input_b_buf.reset(num_devices);
  output_buf.reset(num_devices);
 
  const unsigned num_block_rows = C_height / BLOCK_SIZE;

  for(unsigned i = 0; i < num_devices; ++i) {
    // Command queue.
    queue[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    // Kernel.
    const char *kernel_name = "mnist";
    kernel[i] = clCreateKernel(program, kernel_name, &status);
    checkError(status, "Failed to create kernel");

    // Determine the number of elements processed by this device.
    m_per_device[i] = num_block_rows / num_devices; // number of elements handled by this device

    // Spread out the remainder of the elements over the first
    if(i < (num_block_rows % num_devices)) {
      m_per_device[i]++;
    }
    m_per_device[i] *= BLOCK_SIZE;

    // Input buffers.
    input_a_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA, 
        m_per_device[i] * A_width * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input 1");

    input_b_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA, 
        B_height * B_width * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input 2");

    // Output buffer.
    output_buf[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA, 
        m_per_device[i] * C_width * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for out");
  }

  return true;
}
// Initialize the data for the problem. Requires num_devices to be known.
void init_problem2(Node* node) {
  if(num_devices == 0) {
    checkError(-1, "No devices");
  }

  input_a.reset(num_devices);
  input_b.reset(B_height * B_width);
  output.reset(num_devices);

  for(unsigned i = 0; i < num_devices; ++i) {
    input_a[i].reset(m_per_device[i] * A_width);
    output[i].reset(m_per_device[i] * C_width);

    for(unsigned j = 0; j < node->backwardConnCount; ++j){
        Node *targetNode = node->connections[j].nodePtr;
        if (targetNode != NULL){
          input_a[i][j] = *node->connections[j].weightPtr;
          input_b[j * B_width + i] = targetNode->output;
        } else {  
          input_a[i][j] = 0; 
          input_b[j * B_width + i] = 0;
        }
        std::cout << j << ". w = " << input_a[i][j] << " ; o = " << input_b[j * B_width + i] << std::endl; //TODO: delete

    }
    if (node->backwardConnCount > m_per_device[i] * A_width) { printf("ERROR: data > input_matrix %d %d\n",node->backwardConnCount,m_per_device[i] * A_width); exit(1); }

    printf(" here 2: %d %d\n", node->backwardConnCount, m_per_device[i] * A_width); exit(0);
    for(unsigned j = node->backwardConnCount; j < m_per_device[i] * A_width; ++j) { 
      input_a[i][j] = 0;
      input_b[j * B_width + i] = 0;
  printf("here 3-----\n");
    }
  }
}
//-----------------------------------------------------------------------------
void compute_reference() {
  // Compute the reference output.
  printf("Computing reference output\n");
  ref_out.reset(C_height * C_width);

  for(unsigned y = 0, dev_index = 0; y < C_height; ++dev_index) {
    for(unsigned yy = 0; yy < m_per_device[dev_index]; ++yy, ++y) {
      for(unsigned x = 0; x < C_width; ++x) {
        // Compute result for C(y, x)
        float sum = 0.0f;
        for(unsigned k = 0; k < A_width; ++k) {
          sum += input_a[dev_index][yy * A_width + k] * input_b[k * B_width + x];
        }
        ref_out[y * C_width + x] = sum;
      }
    }
  }
}
//-----------------------------------------------------------------------------
void verify() {
  printf("Verifying\n");

  // Compute the L^2-Norm of the difference between the output and reference
  // output matrices and compare it against the L^2-Norm of the reference.
  float diff = 0.0f;
  float ref = 0.0f;
  for(unsigned y = 0, dev_index = 0; y < C_height; ++dev_index) {
    for(unsigned yy = 0; yy < m_per_device[dev_index]; ++yy, ++y) {
      for(unsigned x = 0; x < C_width; ++x) {
        const float o = output[dev_index][yy * C_width + x];
        const float r = ref_out[y * C_width + x];
        const float d = o - r;
        diff += d * d;
        ref += r * r;
      }
    }
  }

  const float diff_l2norm = sqrtf(diff);
  const float ref_l2norm = sqrtf(ref);
  const float error = diff_l2norm / ref_l2norm;
  const bool pass = error < 1e-6;
  printf("Verification: %s\n", pass ? "PASS" : "FAIL");
  if(!pass) {
    printf("Error (L^2-Norm): %0.3g\n", error);
  }
}
//-----------------------------------------------------------------------------
float run2() {
  cl_int status;

  // Transfer inputs to each device. Each of the host buffers supplied to
  // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
  // for the host-to-device transfer.
  for(unsigned i = 0; i < num_devices; ++i) {
    status = clEnqueueWriteBuffer(queue[i], input_a_buf[i], CL_FALSE,
        0, m_per_device[i] * A_width * sizeof(float), input_a[i], 0, NULL, NULL);
    checkError(status, "Failed to transfer input 1");

    status = clEnqueueWriteBuffer(queue[i], input_b_buf[i], CL_FALSE,
        0, B_width * B_height * sizeof(float), input_b, 0, NULL, NULL);
    checkError(status, "Failed to transfer input 2");
  }
  // Wait for all queues to finish.
  for(unsigned i = 0; i < num_devices; ++i) {
    clFinish(queue[i]);
  }

  // Launch kernels.
  // This is the portion of time that we'll be measuring for throughput
  // benchmarking.
  scoped_array<cl_event> kernel_event(num_devices);

  const double start_time = getCurrentTimestamp();
  for(unsigned i = 0; i < num_devices; ++i) {
    // Set kernel arguments.
    unsigned argi = 0;

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &output_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &input_a_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &input_b_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(A_width), &A_width);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(B_width), &B_width);
    checkError(status, "Failed to set argument %d", argi - 1);

    // Enqueue kernel.
    // Use a global work size corresponding to the size of the output matrix.
    // Each work-item computes the result for one value of the output matrix,
    // so the global work size has the same dimensions as the output matrix.
    // 
    // The local work size is one block, so BLOCK_SIZE x BLOCK_SIZE.
    //
    // Events are used to ensure that the kernel is not launched until
    // the writes to the input buffers have completed.
    const size_t global_work_size[2] = {C_width, m_per_device[i]};
    const size_t local_work_size[2]  = {BLOCK_SIZE, BLOCK_SIZE};
    //printf("Launching for device %d (global size: %d, %d)\n", i, global_work_size[0], global_work_size[1]);

    status = clEnqueueNDRangeKernel(queue[i], kernel[i], 2, NULL,
        global_work_size, local_work_size, 0, NULL, &kernel_event[i]);
    checkError(status, "Failed to launch kernel");
  }

  // Wait for all kernels to finish.
  clWaitForEvents(num_devices, kernel_event);

  const double end_time = getCurrentTimestamp();
  const double total_time = end_time - start_time;

  // Wall-clock time taken.
  printf("\nTime: %0.3f ms\n", total_time * 1e3);

  // Get kernel times using the OpenCL event profiling API.
  for(unsigned i = 0; i < num_devices; ++i) {
    cl_ulong time_ns = getStartEndTime(kernel_event[i]);
    printf("Kernel time (device %d): %0.3f ms\n", i, double(time_ns) * 1e-6);
  }

  // Compute the throughput (GFLOPS).
  // There are C_width * C_height output values, with each value
  // computed using A_width multiplies and adds.
  const float flops = (float)(2.0f * C_width * C_height * A_width / total_time);
  printf("\nThroughput: %0.2f GFLOPS\n\n", flops * 1e-9);

  // Release kernel events.
  for(unsigned i = 0; i < num_devices; ++i) {
    clReleaseEvent(kernel_event[i]);
  }

  // Read the result.
  for(unsigned i = 0; i < num_devices; ++i) {
    status = clEnqueueReadBuffer(queue[i], output_buf[i], CL_TRUE,
        0, m_per_device[i] * C_width * sizeof(float), output[i], 0, NULL, NULL);
    checkError(status, "Failed to read output matrix");
  }

  // Verify results.
  compute_reference();
  verify();
  return output[0][0];
}
//-----------------------------------------------------------------------------
void cleanup2() {
  for(unsigned i = 0; i < num_devices; ++i) {
    if(kernel && kernel[i]) {
      clReleaseKernel(kernel[i]);
    }
    if(queue && queue[i]) {
      clReleaseCommandQueue(queue[i]);
    }
    if(input_a_buf && input_a_buf[i]) {
      clReleaseMemObject(input_a_buf[i]);
    }
    if(input_b_buf && input_b_buf[i]) {
      clReleaseMemObject(input_b_buf[i]);
    }
    if(output_buf && output_buf[i]) {
      clReleaseMemObject(output_buf[i]);
    }
  }

  if(program) {
    clReleaseProgram(program);
  }
  if(context) {
    clReleaseContext(context);
  }
}
void cleanup() { cleanup2(); }
#endif
