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
// This host program executes a vector addition kernel to perform:
//  C = A + B
// where A, B and C are vectors with N elements.
//
// This host program supports partitioning the problem across multiple OpenCL
// devices if available. If there are M available devices, the problem is
// divided so that each device operates on N/M points. The host program
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

using namespace aocl_utils;

// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 1;
unsigned tmp_n_dev; //dummy variable
scoped_array<cl_device_id> device; // num_devices elements
cl_context context = NULL;
scoped_array<cl_command_queue> queue; // num_devices elements
cl_program program = NULL;
scoped_array<cl_kernel> kernel; // num_devices elements
scoped_array<cl_mem> in1_buf; // num_devices elements
scoped_array<cl_mem> in2_buf; // num_devices elements
scoped_array<cl_mem> in3_buf; // num_devices elements
scoped_array<cl_mem> out_buf; // num_devices elements

unsigned M = 50; // problem size
scoped_array<scoped_aligned_ptr<double> > in1, in2; // num_devices elements
scoped_array<scoped_aligned_ptr<double> > out; // num_devices elements
scoped_array<scoped_array<double> > ref_out; // num_devices elements
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
  platform = findPlatform("Altera");
  if(platform == NULL) {
    printf("ERROR: Unable to find Altera OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL device.
  device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &tmp_n_dev));
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
  in1_buf.reset(num_devices);
  in2_buf.reset(num_devices);
  out_buf.reset(num_devices);
 
  for(unsigned i = 0; i < num_devices; ++i) {
    // Command queue.
    queue[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    // Kernel.
    const char *kernel_name = "mnist";
    kernel[i] = clCreateKernel(program, kernel_name, &status);
    checkError(status, "Failed to create kernel");

    // Determine the number of elements processed by this device.
    m_per_device[i] = M / num_devices; // number of elements handled by this device

    // Input buffers.
    in1_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, 
        m_per_device[i] * sizeof(double), NULL, &status);
    checkError(status, "Failed to create buffer for input 1");

    in2_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, 
        m_per_device[i] * sizeof(double), NULL, &status);
    checkError(status, "Failed to create buffer for input 2");

    // Output buffer.
    out_buf[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
        m_per_device[i] * sizeof(double), NULL, &status);
    checkError(status, "Failed to create buffer for out");
  }

  return true;
}
// Initialize the data for the problem. Requires num_devices to be known.
void init_problem2(Node* node, int iteration) {
  if(num_devices == 0) {
    checkError(-1, "No devices");
  }

  in1.reset(num_devices);
  in2.reset(num_devices);
  out.reset(num_devices);
  ref_out.reset(num_devices);

  for(unsigned i = 0; i < num_devices; ++i) {
    in1[i].reset(m_per_device[i]);
    in2[i].reset(m_per_device[i]);
    out[i].reset(m_per_device[i]);
    ref_out[i].reset(m_per_device[i]);

    int count = iteration * m_per_device[i];
    for(unsigned j = iteration * m_per_device[i]; j < (iteration+1) * m_per_device[i]; ++j) {
      unsigned k = j % m_per_device[i];
      if (count < node->backwardConnCount){
        Node *targetNode = node->connections[j].nodePtr;
        //std::cout << "here count = " << count << " of " << node->backwardConnCount << "\n"; //TODO: delete
        if (targetNode != NULL){
          in1[i][k] = *node->connections[j].weightPtr;
          in2[i][k] = targetNode->output;
          ref_out[i][0] += in1[i][k] * in2[i][k];
        } else {  in1[i][k] = 0; in2[i][k] = 0; }
      } else { 
        in1[i][k] = 0; in2[i][k] = 0;
      }
      std::cout << j << ". w = " << in1[i][k] << " ; o = " << in2[i][k] << std::endl; //TODO: delete
      
      count += 1;
    }
  }
}
//-----------------------------------------------------------------------------
double run2() {
  cl_int status;

  const double start_time = getCurrentTimestamp();

  // Launch the problem for each device.
  scoped_array<cl_event> kernel_event(num_devices);
  scoped_array<cl_event> finish_event(num_devices);

  for(unsigned i = 0; i < num_devices; ++i) {

    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event[2];
    status = clEnqueueWriteBuffer(queue[i], in1_buf[i], CL_FALSE,
        0, m_per_device[i] * sizeof(double), in1[i], 0, NULL, &write_event[0]);
    checkError(status, "Failed to transfer input 1");

    status = clEnqueueWriteBuffer(queue[i], in2_buf[i], CL_FALSE,
        0, m_per_device[i] * sizeof(double), in2[i], 0, NULL, &write_event[1]);
    checkError(status, "Failed to transfer input 2");

    // Set kernel arguments.
    unsigned argi = 0;

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &in1_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &in2_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &out_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    // Enqueue kernel.
    // Use a global work size corresponding to the number of elements to add
    // for this device.
    // 
    // We don't specify a local work size and let the runtime choose
    // (it'll choose to use one work-group with the same size as the global
    // work-size).
    //
    // Events are used to ensure that the kernel is not launched until
    // the writes to the input buffers have completed.
    const size_t global_work_size = m_per_device[i];
    //printf("Launching for device %d (%d elements)\n", i, global_work_size);

    status = clEnqueueNDRangeKernel(queue[i], kernel[i], 1, NULL,
        &global_work_size, NULL, 2, write_event, &kernel_event[i]);
    checkError(status, "Failed to launch kernel");

    // Read the result. This the final operation.
    status = clEnqueueReadBuffer(queue[i], out_buf[i], CL_FALSE,
        0, m_per_device[i] * sizeof(double), out[i], 1, &kernel_event[i], &finish_event[i]);

    // Release local events.
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);
  }

  // Wait for all devices to finish.
  clWaitForEvents(num_devices, finish_event);

  const double end_time = getCurrentTimestamp();

  // Wall-clock time taken.
  //printf("\nTime: %0.3f ms\n", (end_time - start_time) * 1e3);

  // Get kernel times using the OpenCL event profiling API.
  /*
  for(unsigned i = 0; i < num_devices; ++i) {
    cl_ulong time_ns = getStartEndTime(kernel_event[i]);
    printf("Kernel time (device %d): %0.3f ms\n", i, double(time_ns) * 1e-6);
  }*/

  // Release all events.
  for(unsigned i = 0; i < num_devices; ++i) {
    clReleaseEvent(kernel_event[i]);
    clReleaseEvent(finish_event[i]);
  }
/*
  // Verify results.
  bool pass = true;
  for(unsigned i = 0; i < num_devices && pass; ++i) {
    //for(unsigned j = 0; j < m_per_device[i] && pass; ++j) {
      if(fabsf(out[i][0] - ref_out[i][0]) > 1.0e-5f) {
        printf("Failed verification @ device %d, index %d\nOutput: %f\nReference: %f\n",
            i, 0, out[i][0], ref_out[i][0]);
        pass = false;
      }
    //}
  } 
  //printf("\nVerification: %s\n", pass ? "PASS" : "FAIL");
*/
  return out[0][0];
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
    if(in1_buf && in1_buf[i]) {
      clReleaseMemObject(in1_buf[i]);
    }
    if(in2_buf && in2_buf[i]) {
      clReleaseMemObject(in2_buf[i]);
    }
    if(out_buf && out_buf[i]) {
      clReleaseMemObject(out_buf[i]);
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
