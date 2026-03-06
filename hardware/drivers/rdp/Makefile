CXX = g++
CXXFLAGS = -O2 -shared -fPIC -std=c++11
TARGET = librdp.so
SRC    = rdp_driver.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) -lm

clean:
	rm -f $(TARGET)
