# Makefile for resample_driver
# Run on the PYNQ board: make && cp libresample.so ../../

CXX = g++
CXXFLAGS = -O2 -shared -fPIC -std=c++11

TARGET = libresample.so
SRC    = resample_driver.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) -lm

clean:
	rm -f $(TARGET)
