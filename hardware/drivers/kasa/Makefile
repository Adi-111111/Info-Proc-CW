CXX = g++
CXXFLAGS = -O2 -shared -fPIC -std=c++11
TARGET = libkasa.so
SRC    = kasa_driver.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) -lm

clean:
	rm -f $(TARGET)
