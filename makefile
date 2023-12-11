FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile

nbody: nbody.o compute.o
	gcc $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.cc planets.h config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $< 
compute.o: compute.cu config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $< 
clean:
	rm -f *.o nbody 
