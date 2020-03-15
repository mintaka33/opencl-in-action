#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "interp.cl"
#define KERNEL_FUNC "interp"

#define SCALE_FACTOR 4

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {

    cl_platform_id platform;
    cl_device_id dev;
    int err;

    /* Identify a platform */
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err < 0) {
        perror("Couldn't identify a platform");
        exit(1);
    }

    /* Access a device */
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err == CL_DEVICE_NOT_FOUND) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    }
    if (err < 0) {
        perror("Couldn't access any devices");
        exit(1);
    }

    return dev;
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

    cl_program program;
    FILE* program_handle;
    char* program_buffer, * program_log;
    size_t program_size, log_size;
    int err;
    char arg[20];

    /* Read program file and place content into buffer */
    program_handle = fopen(filename, "r");
    if (program_handle == NULL) {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    /* Create program from file */
    program = clCreateProgramWithSource(ctx, 1,
        (const char**)&program_buffer, &program_size, &err);
    if (err < 0) {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);

    /* Create build argument */
    sprintf(arg, "-DSCALE=%u", SCALE_FACTOR);

    /* Build program */
    err = clBuildProgram(program, 0, NULL, arg, NULL, NULL);
    if (err < 0) {

        /* Find size of log and print to std output */
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
            0, NULL, &log_size);
        program_log = (char*)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
            log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return program;
}

int main(int argc, char** argv) {

    /* Host/device data structures */
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;
    size_t global_size[2];

    /* Image data */
    float input_pixels[16], output_pixels[16* SCALE_FACTOR];
    cl_image_format png_format;
    cl_mem input_image, output_image;
    size_t width = 4, height = 4;
    size_t origin[3], region[3];

    /* Create a device and context */
    device = create_device();
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err < 0) {
        perror("Couldn't create a context");
        exit(1);
    }

    /* Create kernel */
    program = build_program(context, device, PROGRAM_FILE);
    kernel = clCreateKernel(program, KERNEL_FUNC, &err);
    if (err < 0) {
        printf("Couldn't create a kernel: %d", err);
        exit(1);
    };

    /* Create input image object */
    png_format.image_channel_order = CL_LUMINANCE;
    png_format.image_channel_data_type = CL_UNORM_INT16;
    input_image = clCreateImage2D(context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        &png_format, width, height, 0, (void*)input_pixels, &err);
    if (err < 0) {
        perror("Couldn't create the image object");
        exit(1);
    };

    /* Create output image object */
    output_image = clCreateImage2D(context,
        CL_MEM_WRITE_ONLY, &png_format, SCALE_FACTOR * width,
        SCALE_FACTOR * height, 0, NULL, &err);
    if (err < 0) {
        perror("Couldn't create the image object");
        exit(1);
    };

    /* Create buffer */
    if (err < 0) {
        perror("Couldn't create a buffer");
        exit(1);
    };

    /* Create kernel arguments */
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
    if (err < 0) {
        printf("Couldn't set a kernel argument");
        exit(1);
    };

    /* Create a command queue */
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err < 0) {
        perror("Couldn't create a command queue");
        exit(1);
    };

    /* Enqueue kernel */
    global_size[0] = width; global_size[1] = height;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size,
        NULL, 0, NULL, NULL);
    if (err < 0) {
        perror("Couldn't enqueue the kernel");
        exit(1);
    }

    /* Read the image object */
    origin[0] = 0; origin[1] = 0; origin[2] = 0;
    region[0] = SCALE_FACTOR * width; region[1] = SCALE_FACTOR * height; region[2] = 1;
    err = clEnqueueReadImage(queue, output_image, CL_TRUE, origin,
        region, 0, 0, (void*)output_pixels, 0, NULL, NULL);
    if (err < 0) {
        perror("Couldn't read from the image object");
        exit(1);
    }

    /* Deallocate resources */
    free(input_pixels);
    free(output_pixels);
    clReleaseMemObject(input_image);
    clReleaseMemObject(output_image);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 0;
}
