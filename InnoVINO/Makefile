# Makefile template for shared library

CC = g++
CFLAGS = -fPIC -Wall -Wextra -O3 -g -m64 -std=gnu99
LDFLAGS = -linference_engine -lopencv_core -lopencv_dnn -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_video -lpthread
RM = rm
TARGET_LIB = libInnoVINO.so
LIB_SRCS = InnoVINO.cpp COPVO.cpp
LIB_NAME = InnoVINO
SAMPLE_EXC = UsingInnoVINO
SAMPLE_SRC = main.cpp
INCLUDE_PATH = -I./ -I/opt/intel/openvino_2021/deployment_tools/inference_engine/include/ -I/opt/intel/openvino_2021/deployment_tools/inference_engine/include/cpp/ -I/opt/intel/openvino_2021/opencv/include/
LIB_PATH = -L. -L/opt/intel/openvino_2021/deployment_tools/inference_engine/lib/intel64 -L/opt/intel/openvino_2021/opencv/lib -L~/inference_engine_cpp_samples_build/intel64/Release/lib

.PHONY: all
all: $(TARGET_LIB) $(SAMPLE_EXC)

$(TARGET_LIB): $(LIB_SRCS)
		$(CC) -shared -o $@ $^ $(CFLAGS) $(INCLUDE_PATH) $(LIB_PATH) ${LDFLAGS}
$(SAMPLE_EXC):
		$(CC) $(SAMPLE_SRC) -I/opt/intel/openvino_2021/opencv/include/ -L/opt/intel/openvino_2021/opencv/lib -ldl -lopencv_core -lopencv_dnn -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_video -o $(SAMPLE_EXC)


# .PHONY: lib
# lib: $(TARGET_LIB)
# $(TARGET_LIB): $(LIB_SRCS)
# 		$(CC) -shared -o $@ $^ $(CFLAGS) $(INCLUDE_PATH) $(LIB_PATH) ${LDFLAGS}

# .PHONY: sample
# sample: $(SAMPLE_EXC)
# $(SAMPLE_EXC):
# 		$(CC) $(SAMPLE_SRC) -I/opt/intel/openvino_2021/opencv/include/ -L/opt/intel/openvino_2021/opencv/lib -ldl -lopencv_core -lopencv_dnn -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_video -o $(SAMPLE_EXC)

.PHONY: clean
clean:
		-$(RM) $(TARGET_LIB) $(SAMPLE_EXC)
