all: check check-cppyy

lib: clean
	$(CXX) -Wall -Wextra -pedantic -Wshadow -O3 -march=native -mtune=native unity_build.cpp -shared -fPIC -o libMocosHelper.so
debuglib: clean
	$(CXX) -Wall -Wextra -pedantic -Wshadow -O0 -g -D_GLIBCXX_DEBUG_PEDANTIC -D_GLIBCXX_DEBUG unity_build.cpp -shared -fPIC -o libMocosHelper.so
check:
	$(CXX) -std=c++17 -fsyntax-only -Wall -Wextra -pedantic -Wshadow unity_build.cpp
check-cppyy:
	$(CXX) -std=c++17 -fsyntax-only -DMOCOS_CPPYY -Wall -Wextra -pedantic -Wshadow unity_build.cpp
clean:
	rm libMocosHelper.so
test: *.cpp
	$(CXX) -Wall -Wextra -pedantic -Wshadow -O0 -g -D_GLIBCXX_DEBUG_PEDANTIC -D_GLIBCXX_DEBUG test.cpp -o test
