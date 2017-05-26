PREFIX=${PARSECDIR}/pkgs/apps_simd/swat/inst/${PARSECPLAT}

TARGET=SmithWaterman

#.PHONY: gpp
default:
	#g++ SmithWaterman.cpp pthread_funcs.cpp functions.cpp globals.cpp -o SmithWaterman -W -Wall -Wextra -lpthread -lrt
	$(CXX) $(CXXFLAGS) $(LDFLAGS) SmithWaterman.cpp pthread_funcs.cpp functions.cpp globals.cpp $(LIBS) -o $(TARGET)

#.PHONY: gppO3
#gppO3:
#	g++ SmithWaterman.cpp pthread_funcs.cpp functions.cpp globals.cpp -o SmithWaterman -W -Wall -Wextra -#lpthread -lrt -O3

#.PHONY: icpc
#icpc:
#	icpc SmithWaterman.cpp pthread_funcs.cpp functions.cpp globals.cpp -o SmithWaterman -W -Wall -Wextra -#lpthread -lrt

#.PHONY: icpcO3
#icpcO3:
#	icpc SmithWaterman.cpp pthread_funcs.cpp functions.cpp globals.cpp -o SmithWaterman -W -Wall -Wextra -#lpthread -lrt -O3


install:
	mkdir -p $(PREFIX)/bin
	cp -f $(TARGET) $(PREFIX)/bin/$(TARGET)