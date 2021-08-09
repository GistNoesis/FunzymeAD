testmnist: testmnist.cpp  funzyme/cpulayers.hpp
		clang  testmnist.cpp  -lstdc++ -lm \
		 																	-Rpass=enzyme -Xclang -load -Xclang /usr/local/lib/ClangEnzyme-12.so \
		  																			 -O2  -o bin/testmnist -fno-exceptions
