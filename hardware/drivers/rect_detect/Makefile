CXX = g++
CXXFLAGS = -O2 -shared -fPIC -std=c++11
TARGET = librect_detect.so
SRC    = rect_detect_driver.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) -lm

clean:
	rm -f $(TARGET)
